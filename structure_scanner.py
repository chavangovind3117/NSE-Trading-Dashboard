"""
Structure Scanner — structure_scanner.py
=========================================
Scans price structure on your NSE watchlist every 2 hours.
Detects 3 things that institutions leave behind in price action:

1. LIQUIDITY SWEEP (Stop Hunt)
   Price spikes below a recent low to trigger retail stop-losses
   then immediately reverses back up. Institutions loaded there.
   Entry: Right after the reversal candle closes above swept level.

2. ORDER BLOCKS
   The last bearish candle before a strong bullish impulse move.
   Institutions placed unfilled buy orders there.
   Price returns to that zone = high probability support.

3. MARKET STRUCTURE BREAK (MSB)
   In uptrend: price breaks below the last Higher Low = trend change.
   In downtrend: price breaks above the last Lower High = trend change.
   Fires 2-5 candles before EMAs confirm — early warning system.

All three send Telegram alerts with AI trade plan.
All saved to nse_trading.db.

Run: python structure_scanner.py
Scans: Every 2 hours during market hours (9:15 AM – 3:30 PM IST)
"""

import yfinance as yf
import pandas as pd
import numpy as np
import requests
import sqlite3
import schedule
import time
import pytz
from groq import Groq
from datetime import datetime, timedelta
from config import TELEGRAM_TOKEN, CHAT_ID, GROQ_API_KEY, GROQ_MODEL

# ── CONFIG ────────────────────────────────────────────────────────────────────
IST     = pytz.timezone("Asia/Kolkata")
DB_PATH = "nse_trading.db"

WATCHLIST = {
    "Reliance":      "RELIANCE.NS",
    "TCS":           "TCS.NS",
    "HDFC Bank":     "HDFCBANK.NS",
    "Infosys":       "INFY.NS",
    "ICICI Bank":    "ICICIBANK.NS",
    "SBI":           "SBIN.NS",
    "Kotak Bank":    "KOTAKBANK.NS",
    "Bajaj Finance": "BAJFINANCE.NS",
    "Axis Bank":     "AXISBANK.NS",
    "Wipro":         "WIPRO.NS",
    "Maruti":        "MARUTI.NS",
    "Sun Pharma":    "SUNPHARMA.NS",
    "Titan":         "TITAN.NS",
    "L&T":           "LT.NS",
    "Tata Motors":   "TATAMOTORS.NS",
    "Bharti Airtel": "BHARTIARTL.NS",
    "Asian Paints":  "ASIANPAINT.NS",
    "Tech Mahindra": "TECHM.NS",
    "Zomato":        "ZOMATO.NS",
    "IRCTC":         "IRCTC.NS",
    "Dixon Tech":    "DIXON.NS",
    "NIFTY 50":      "^NSEI",
    "BANK NIFTY":    "^NSEBANK",
}

# Timeframes to scan
TIMEFRAMES = {
    "Daily":  {"period": "6mo", "interval": "1d"},
    "Weekly": {"period": "2y",  "interval": "1wk"},
}

# ── DATABASE ──────────────────────────────────────────────────────────────────
def init_db():
    conn = sqlite3.connect(DB_PATH)
    conn.execute("""CREATE TABLE IF NOT EXISTS structure_alerts (
        id          INTEGER PRIMARY KEY AUTOINCREMENT,
        timestamp   TEXT,
        stock       TEXT,
        ticker      TEXT,
        timeframe   TEXT,
        alert_type  TEXT,
        direction   TEXT,
        price       REAL,
        key_level   REAL,
        description TEXT,
        analysis    TEXT,
        alerted     INTEGER DEFAULT 1
    )""")
    conn.commit()
    conn.close()

def already_alerted(stock, alert_type, timeframe, hours=8):
    """Don't re-alert same setup within 8 hours."""
    conn     = sqlite3.connect(DB_PATH)
    cutoff   = (datetime.now(IST) - timedelta(hours=hours)).strftime("%Y-%m-%d %H:%M:%S")
    row      = conn.execute("""SELECT id FROM structure_alerts
        WHERE stock=? AND alert_type=? AND timeframe=? AND timestamp > ?""",
        (stock, alert_type, timeframe, cutoff)).fetchone()
    conn.close()
    return row is not None

def save_alert(stock, ticker, tf, atype, direction, price, key_level, desc, analysis):
    conn = sqlite3.connect(DB_PATH)
    conn.execute("""INSERT INTO structure_alerts
        (timestamp,stock,ticker,timeframe,alert_type,direction,price,key_level,description,analysis)
        VALUES (?,?,?,?,?,?,?,?,?,?)""",
        (datetime.now(IST).strftime("%Y-%m-%d %H:%M:%S"),
         stock, ticker, tf, atype, direction, price, key_level, desc, analysis))
    conn.commit()
    conn.close()

def load_structure_history(limit=50):
    conn = sqlite3.connect(DB_PATH)
    df   = pd.read_sql_query(
        "SELECT * FROM structure_alerts ORDER BY id DESC LIMIT ?",
        conn, params=(limit,))
    conn.close()
    return df

# ── TELEGRAM ──────────────────────────────────────────────────────────────────
def send(text: str):
    url = f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/sendMessage"
    for chunk in [text[i:i+4000] for i in range(0, len(text), 4000)]:
        try:
            requests.post(url, json={
                "chat_id": CHAT_ID, "text": chunk, "parse_mode": "HTML"
            }, timeout=15)
        except Exception as e:
            print(f"  Telegram error: {e}")
        time.sleep(0.5)

def ts():
    return datetime.now(IST).strftime("%d %b %Y %H:%M IST")

# ── GROQ AI ───────────────────────────────────────────────────────────────────
def ai(prompt: str, max_tokens=400) -> str:
    try:
        r = Groq(api_key=GROQ_API_KEY).chat.completions.create(
            model=GROQ_MODEL,
            messages=[
                {"role": "system", "content":
                 "You are an expert NSE India price action trader. "
                 "Specialise in institutional order flow and market structure. "
                 "Be specific with price levels. Concise and actionable."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.3, max_tokens=max_tokens)
        return r.choices[0].message.content.strip()
    except Exception as e:
        return f"AI unavailable: {e}"

# ── DATA FETCHING ─────────────────────────────────────────────────────────────
def fetch(ticker: str, period: str, interval: str) -> pd.DataFrame:
    try:
        df = yf.download(ticker, period=period, interval=interval,
                         progress=False, auto_adjust=True)
        if df.empty:
            return df
        df.columns = [c[0] if isinstance(c, tuple) else c for c in df.columns]
        df.dropna(inplace=True)
        return df
    except Exception:
        return pd.DataFrame()

def add_basic_indicators(df: pd.DataFrame) -> pd.DataFrame:
    if len(df) < 20:
        return df
    df = df.copy()
    df["EMA20"]  = df["Close"].ewm(span=20,  adjust=False).mean()
    df["EMA50"]  = df["Close"].ewm(span=50,  adjust=False).mean()
    df["EMA200"] = df["Close"].ewm(span=200, adjust=False).mean()
    delta = df["Close"].diff()
    gain  = delta.where(delta > 0, 0).rolling(14).mean()
    loss  = (-delta.where(delta < 0, 0)).rolling(14).mean()
    df["RSI"] = 100 - (100 / (1 + gain / loss))
    tr = pd.concat([
        df["High"] - df["Low"],
        (df["High"] - df["Close"].shift()).abs(),
        (df["Low"]  - df["Close"].shift()).abs()
    ], axis=1).max(axis=1)
    df["ATR"] = tr.rolling(14).mean()
    df["Body"]       = abs(df["Close"] - df["Open"])
    df["Upper_wick"] = df["High"] - df[["Close","Open"]].max(axis=1)
    df["Lower_wick"] = df[["Close","Open"]].min(axis=1) - df["Low"]
    df["Bullish"]    = df["Close"] > df["Open"]
    return df

# ── SWING POINT DETECTOR ──────────────────────────────────────────────────────
def find_swing_highs(df: pd.DataFrame, lookback: int = 3) -> pd.Series:
    """
    A swing high is a candle whose High is higher than the
    `lookback` candles on each side.
    """
    highs  = df["High"]
    result = pd.Series(False, index=df.index)
    for i in range(lookback, len(df) - lookback):
        window = highs.iloc[i-lookback:i+lookback+1]
        if highs.iloc[i] == window.max():
            result.iloc[i] = True
    return result

def find_swing_lows(df: pd.DataFrame, lookback: int = 3) -> pd.Series:
    """
    A swing low is a candle whose Low is lower than the
    `lookback` candles on each side.
    """
    lows   = df["Low"]
    result = pd.Series(False, index=df.index)
    for i in range(lookback, len(df) - lookback):
        window = lows.iloc[i-lookback:i+lookback+1]
        if lows.iloc[i] == window.min():
            result.iloc[i] = True
    return result

# ══════════════════════════════════════════════════════════════════════════════
# DETECTOR 1 — LIQUIDITY SWEEP
# ══════════════════════════════════════════════════════════════════════════════

def detect_liquidity_sweep(df: pd.DataFrame, stock: str, tf: str) -> dict | None:
    """
    Bullish sweep: price spikes BELOW a recent swing low (triggers retail stops)
    then closes BACK ABOVE that level in the same or next candle.
    This means: institutions swept the stops, collected liquidity, now going up.

    Bearish sweep: price spikes ABOVE a recent swing high then closes back below.
    Institutions distributed into retail FOMO buyers.

    Logic:
      1. Find the most significant swing low in last 20 bars
      2. Check if the last 1-2 candles dipped below it
      3. Check if price closed back above it (reversal confirmed)
      4. Confirm with RSI not making new lows (divergence = stronger signal)
      5. Volume should be above average on sweep candle
    """
    if len(df) < 25:
        return None

    df    = df.copy().reset_index(drop=True)
    n     = len(df)
    close = df["Close"]
    high  = df["High"]
    low   = df["Low"]
    atr   = df["ATR"].iloc[-1] if "ATR" in df.columns else (high - low).mean()
    rsi   = df["RSI"].iloc[-1] if "RSI" in df.columns else 50

    # ── Bullish sweep (most common and actionable) ────────
    # Find the lowest swing low in last 5-20 bars (excluding last 2)
    lookback_lows = low.iloc[-20:-2]
    if lookback_lows.empty:
        return None

    recent_swing_low = float(lookback_lows.min())
    swing_low_idx    = int(lookback_lows.idxmin())

    last_low   = float(low.iloc[-1])
    last_close = float(close.iloc[-1])
    prev_close = float(close.iloc[-2])
    last_high  = float(high.iloc[-1])

    # Condition 1: Last candle dipped below the swing low
    swept_below = last_low < recent_swing_low

    # Condition 2: Last candle closed BACK above the swing low (reversal)
    closed_back_above = last_close > recent_swing_low

    # Condition 3: The sweep was significant (at least 0.1x ATR below)
    sweep_depth = recent_swing_low - last_low
    significant_sweep = sweep_depth >= (atr * 0.1)

    # Condition 4: RSI not at extreme lows (avoiding genuine breakdowns)
    rsi_ok = rsi > 25

    # Condition 5: Candle has a visible lower wick (spike and reversal)
    lower_wick = float(df["Lower_wick"].iloc[-1]) if "Lower_wick" in df.columns else 0
    has_wick   = lower_wick > (sweep_depth * 0.3)

    if swept_below and closed_back_above and significant_sweep and rsi_ok:
        # Determine if sweep happened on previous candle too (stronger signal)
        prev_low   = float(low.iloc[-2])
        multi_bar  = prev_low < recent_swing_low

        # Check volume (higher volume on sweep = more conviction)
        vol_cur = float(df["Volume"].iloc[-1]) if "Volume" in df.columns else 0
        vol_avg = float(df["Volume"].tail(20).mean()) if "Volume" in df.columns else 0
        vol_ratio = vol_cur / vol_avg if vol_avg > 0 else 1.0

        return {
            "type":           "LIQUIDITY_SWEEP",
            "direction":      "BULLISH",
            "swept_level":    round(recent_swing_low, 2),
            "sweep_low":      round(last_low, 2),
            "current_price":  round(last_close, 2),
            "sweep_depth":    round(sweep_depth, 2),
            "sweep_pct":      round((sweep_depth / recent_swing_low) * 100, 3),
            "volume_ratio":   round(vol_ratio, 1),
            "rsi":            round(rsi, 1),
            "atr":            round(float(atr), 2),
            "multi_bar":      multi_bar,
            "stop_loss":      round(last_low - (atr * 0.5), 2),
            "target_1":       round(last_close + (sweep_depth * 2), 2),
            "target_2":       round(last_close + (sweep_depth * 4), 2),
        }

    # ── Bearish sweep ─────────────────────────────────────
    lookback_highs   = high.iloc[-20:-2]
    recent_swing_high = float(lookback_highs.max())
    last_high_val    = float(high.iloc[-1])
    last_close_val   = float(close.iloc[-1])

    swept_above    = last_high_val > recent_swing_high
    closed_back    = last_close_val < recent_swing_high
    sweep_height   = last_high_val - recent_swing_high
    sig_sweep_bear = sweep_height >= (atr * 0.1)
    rsi_ok_bear    = rsi < 75

    if swept_above and closed_back and sig_sweep_bear and rsi_ok_bear:
        vol_cur   = float(df["Volume"].iloc[-1]) if "Volume" in df.columns else 0
        vol_avg   = float(df["Volume"].tail(20).mean()) if "Volume" in df.columns else 0
        vol_ratio = vol_cur / vol_avg if vol_avg > 0 else 1.0
        return {
            "type":           "LIQUIDITY_SWEEP",
            "direction":      "BEARISH",
            "swept_level":    round(recent_swing_high, 2),
            "sweep_high":     round(last_high_val, 2),
            "current_price":  round(last_close_val, 2),
            "sweep_depth":    round(sweep_height, 2),
            "sweep_pct":      round((sweep_height / recent_swing_high) * 100, 3),
            "volume_ratio":   round(vol_ratio, 1),
            "rsi":            round(rsi, 1),
            "atr":            round(float(atr), 2),
            "stop_loss":      round(last_high_val + (float(atr) * 0.5), 2),
            "target_1":       round(last_close_val - (sweep_height * 2), 2),
            "target_2":       round(last_close_val - (sweep_height * 4), 2),
        }

    return None

# ══════════════════════════════════════════════════════════════════════════════
# DETECTOR 2 — ORDER BLOCKS
# ══════════════════════════════════════════════════════════════════════════════

def detect_order_block(df: pd.DataFrame, stock: str, tf: str) -> dict | None:
    """
    Bullish order block: The LAST BEARISH candle before a strong bullish impulse.
    When price returns to that candle's range, institutions have unfilled buy orders there.

    Identification:
      1. Find a 3+ candle bullish impulse (strong consecutive up move)
      2. The candle JUST before that impulse started = bullish order block
      3. Price has now returned to that candle's High-Low range
      4. This is the entry zone

    Bearish order block: Last bullish candle before strong bearish impulse.
    Price returning to that zone = distribution zone / resistance.
    """
    if len(df) < 30:
        return None

    df    = df.copy().reset_index(drop=True)
    n     = len(df)
    close = df["Close"]
    high  = df["High"]
    low   = df["Low"]
    atr   = float(df["ATR"].iloc[-1]) if "ATR" in df.columns else float((high-low).mean())
    rsi   = float(df["RSI"].iloc[-1]) if "RSI" in df.columns else 50
    ltp   = float(close.iloc[-1])

    # ── Scan for bullish order blocks ────────────────────
    # Look back through the last 30 candles for impulse moves
    impulse_threshold = atr * 0.8   # each impulse candle > 0.8x ATR

    for i in range(n - 6, 5, -1):
        # Check for 3+ consecutive bullish candles starting at i
        bullish_run = 0
        for j in range(i, min(i+5, n)):
            if (close.iloc[j] > df["Open"].iloc[j] and
                (close.iloc[j] - df["Open"].iloc[j]) > impulse_threshold * 0.5):
                bullish_run += 1
            else:
                break

        if bullish_run >= 3:
            # The candle just before this impulse is the order block
            ob_idx   = i - 1
            if ob_idx < 0: continue

            ob_high  = float(high.iloc[ob_idx])
            ob_low   = float(low.iloc[ob_idx])
            ob_open  = float(df["Open"].iloc[ob_idx])
            ob_close = float(close.iloc[ob_idx])
            ob_bearish = ob_close < ob_open  # should be bearish (red) candle

            if not ob_bearish: continue
            if (ob_high - ob_low) < atr * 0.2: continue  # too small

            # Check if current price is returning to (or inside) the order block
            in_ob_zone = ob_low <= ltp <= ob_high * 1.005
            near_ob    = ob_high < ltp <= ob_high * 1.02   # within 2% above

            if in_ob_zone or near_ob:
                # Check price hasn't already broken below the OB (invalidated)
                min_since_ob = float(low.iloc[ob_idx:].min())
                if min_since_ob < ob_low * 0.995:
                    continue  # OB invalidated by breach

                impulse_size = float(close.iloc[i + bullish_run - 1]) - float(close.iloc[i])
                return {
                    "type":          "ORDER_BLOCK",
                    "direction":     "BULLISH",
                    "ob_high":       round(ob_high, 2),
                    "ob_low":        round(ob_low, 2),
                    "ob_mid":        round((ob_high + ob_low) / 2, 2),
                    "current_price": round(ltp, 2),
                    "impulse_size":  round(impulse_size, 2),
                    "impulse_pct":   round((impulse_size / float(close.iloc[i])) * 100, 2),
                    "candles_ago":   n - ob_idx,
                    "rsi":           round(rsi, 1),
                    "atr":           round(atr, 2),
                    "stop_loss":     round(ob_low - atr * 0.3, 2),
                    "target_1":      round(ob_high + impulse_size * 0.5, 2),
                    "target_2":      round(ob_high + impulse_size, 2),
                    "in_zone":       in_ob_zone,
                }

    # ── Scan for bearish order blocks ────────────────────
    for i in range(n - 6, 5, -1):
        bearish_run = 0
        for j in range(i, min(i+5, n)):
            if (close.iloc[j] < df["Open"].iloc[j] and
                (df["Open"].iloc[j] - close.iloc[j]) > impulse_threshold * 0.5):
                bearish_run += 1
            else:
                break

        if bearish_run >= 3:
            ob_idx   = i - 1
            if ob_idx < 0: continue

            ob_high  = float(high.iloc[ob_idx])
            ob_low   = float(low.iloc[ob_idx])
            ob_open  = float(df["Open"].iloc[ob_idx])
            ob_close = float(close.iloc[ob_idx])
            ob_bullish = ob_close > ob_open

            if not ob_bullish: continue
            if (ob_high - ob_low) < atr * 0.2: continue

            # Price returning to bearish OB from below
            in_ob_zone = ob_low * 0.995 <= ltp <= ob_high
            near_ob    = ob_low * 0.98 <= ltp < ob_low

            if in_ob_zone or near_ob:
                max_since_ob = float(high.iloc[ob_idx:].max())
                if max_since_ob > ob_high * 1.005:
                    continue  # OB invalidated

                impulse_size = float(close.iloc[i]) - float(close.iloc[i + bearish_run - 1])
                return {
                    "type":          "ORDER_BLOCK",
                    "direction":     "BEARISH",
                    "ob_high":       round(ob_high, 2),
                    "ob_low":        round(ob_low, 2),
                    "ob_mid":        round((ob_high + ob_low) / 2, 2),
                    "current_price": round(ltp, 2),
                    "impulse_size":  round(impulse_size, 2),
                    "impulse_pct":   round((impulse_size / float(close.iloc[i])) * 100, 2),
                    "candles_ago":   n - ob_idx,
                    "rsi":           round(rsi, 1),
                    "atr":           round(atr, 2),
                    "stop_loss":     round(ob_high + atr * 0.3, 2),
                    "target_1":      round(ob_low - impulse_size * 0.5, 2),
                    "target_2":      round(ob_low - impulse_size, 2),
                    "in_zone":       in_ob_zone,
                }

    return None

# ══════════════════════════════════════════════════════════════════════════════
# DETECTOR 3 — MARKET STRUCTURE BREAK
# ══════════════════════════════════════════════════════════════════════════════

def detect_msb(df: pd.DataFrame, stock: str, tf: str) -> dict | None:
    """
    Market Structure Break — earliest possible trend change signal.

    In an UPTREND (higher highs + higher lows):
      A break BELOW the last Higher Low = trend flipping bearish.
      This fires before EMA crossovers, before MACD flips.

    In a DOWNTREND (lower highs + lower lows):
      A break ABOVE the last Lower High = trend flipping bullish.
      This fires before most indicators confirm.

    Steps:
      1. Identify current trend using last 10 swing points
      2. Find the most recent key level (last HL in uptrend / LH in downtrend)
      3. Check if current close has broken that level
      4. Confirm it's a decisive break (not just a wick)
    """
    if len(df) < 30:
        return None

    df    = df.copy().reset_index(drop=True)
    n     = len(df)
    close = df["Close"]
    high  = df["High"]
    low   = df["Low"]
    atr   = float(df["ATR"].iloc[-1]) if "ATR" in df.columns else float((high-low).mean())
    rsi   = float(df["RSI"].iloc[-1]) if "RSI" in df.columns else 50
    ltp   = float(close.iloc[-1])

    # Find swing highs and lows
    sh_mask = find_swing_highs(df, lookback=3)
    sl_mask = find_swing_lows(df,  lookback=3)

    sh_indices = [i for i in range(n) if sh_mask.iloc[i]]
    sl_indices = [i for i in range(n) if sl_mask.iloc[i]]

    # Need at least 3 swing points to determine trend
    if len(sh_indices) < 2 or len(sl_indices) < 2:
        return None

    # Recent swing points (last 4 of each)
    recent_sh = sh_indices[-4:] if len(sh_indices) >= 4 else sh_indices
    recent_sl = sl_indices[-4:] if len(sl_indices) >= 4 else sl_indices

    # ── Determine current trend ───────────────────────────
    # Uptrend: each recent swing high > previous, each recent swing low > previous
    sh_highs = [float(high.iloc[i]) for i in recent_sh]
    sl_lows  = [float(low.iloc[i])  for i in recent_sl]

    hh_count = sum(1 for i in range(1, len(sh_highs)) if sh_highs[i] > sh_highs[i-1])
    hl_count = sum(1 for i in range(1, len(sl_lows))  if sl_lows[i]  > sl_lows[i-1])
    lh_count = sum(1 for i in range(1, len(sh_highs)) if sh_highs[i] < sh_highs[i-1])
    ll_count = sum(1 for i in range(1, len(sl_lows))  if sl_lows[i]  < sl_lows[i-1])

    in_uptrend   = hh_count >= 1 and hl_count >= 1
    in_downtrend = lh_count >= 1 and ll_count >= 1

    if not in_uptrend and not in_downtrend:
        return None  # no clear trend — no MSB possible

    # ── Check for bearish MSB (uptrend breaking down) ─────
    if in_uptrend:
        # Last Higher Low is the key level to watch
        last_hl_idx   = recent_sl[-1]
        last_hl_price = float(low.iloc[last_hl_idx])

        # MSB: current close decisively BELOW last Higher Low
        broke_below = ltp < last_hl_price
        decisive    = (last_hl_price - ltp) >= (atr * 0.2)  # meaningful break

        if broke_below and decisive:
            # Make sure this is recent (happened in last 3 candles)
            recent_lows  = [float(close.iloc[i]) for i in range(max(0,n-4), n)]
            first_break  = next((n - 4 + i for i, c in enumerate(recent_lows)
                                 if c < last_hl_price), n-1)
            bars_ago     = n - 1 - first_break

            if bars_ago <= 3:
                ema200 = float(df["EMA200"].iloc[-1]) if "EMA200" in df.columns else ltp
                return {
                    "type":          "MARKET_STRUCTURE_BREAK",
                    "direction":     "BEARISH",
                    "broken_level":  round(last_hl_price, 2),
                    "current_price": round(ltp, 2),
                    "break_size":    round(last_hl_price - ltp, 2),
                    "break_pct":     round(((last_hl_price - ltp) / last_hl_price) * 100, 2),
                    "previous_trend":"UPTREND",
                    "bars_ago":      bars_ago,
                    "rsi":           round(rsi, 1),
                    "atr":           round(atr, 2),
                    "above_ema200":  ltp > ema200,
                    "stop_loss":     round(last_hl_price + atr * 0.5, 2),
                    "target_1":      round(ltp - (last_hl_price - ltp) * 1.5, 2),
                    "target_2":      round(ltp - (last_hl_price - ltp) * 3.0, 2),
                }

    # ── Check for bullish MSB (downtrend breaking up) ─────
    if in_downtrend:
        # Last Lower High is the key level to watch
        last_lh_idx   = recent_sh[-1]
        last_lh_price = float(high.iloc[last_lh_idx])

        broke_above = ltp > last_lh_price
        decisive    = (ltp - last_lh_price) >= (atr * 0.2)

        if broke_above and decisive:
            recent_closes = [float(close.iloc[i]) for i in range(max(0,n-4), n)]
            first_break   = next((n - 4 + i for i, c in enumerate(recent_closes)
                                  if c > last_lh_price), n-1)
            bars_ago      = n - 1 - first_break

            if bars_ago <= 3:
                ema200 = float(df["EMA200"].iloc[-1]) if "EMA200" in df.columns else ltp
                return {
                    "type":          "MARKET_STRUCTURE_BREAK",
                    "direction":     "BULLISH",
                    "broken_level":  round(last_lh_price, 2),
                    "current_price": round(ltp, 2),
                    "break_size":    round(ltp - last_lh_price, 2),
                    "break_pct":     round(((ltp - last_lh_price) / last_lh_price) * 100, 2),
                    "previous_trend":"DOWNTREND",
                    "bars_ago":      bars_ago,
                    "rsi":           round(rsi, 1),
                    "atr":           round(atr, 2),
                    "above_ema200":  ltp > ema200,
                    "stop_loss":     round(last_lh_price - atr * 0.5, 2),
                    "target_1":      round(ltp + (ltp - last_lh_price) * 1.5, 2),
                    "target_2":      round(ltp + (ltp - last_lh_price) * 3.0, 2),
                }

    return None

# ══════════════════════════════════════════════════════════════════════════════
# AI ANALYSIS BUILDERS
# ══════════════════════════════════════════════════════════════════════════════

def ai_sweep_analysis(stock, ticker, tf, result) -> str:
    d = result["direction"]
    return ai(f"""NSE India price action alert — Liquidity Sweep detected.

Stock: {stock} ({ticker}) | Timeframe: {tf}
Direction: {d}
Swept level: ₹{result['swept_level']:,.2f}
Sweep low/high: ₹{result.get('sweep_low') or result.get('sweep_high'):,.2f}
Current price: ₹{result['current_price']:,.2f}
Sweep depth: ₹{result['sweep_depth']:,.2f} ({result['sweep_pct']:.3f}%)
Volume on sweep: {result['volume_ratio']:.1f}x average
RSI: {result['rsi']}
ATR: ₹{result['atr']:,.2f}

A liquidity sweep means institutions drove price below/above a key level to
trigger retail stop-losses, collected that liquidity, then reversed.

Write a trading plan:
1. WHAT HAPPENED: One sentence explaining the sweep in simple terms
2. ENTRY: Specific entry price and method (limit or market)
3. STOP-LOSS: ₹{result['stop_loss']:,.2f} — confirm this is correct or adjust
4. TARGET 1: ₹{result['target_1']:,.2f} (1R)
5. TARGET 2: ₹{result['target_2']:,.2f} (2R)
6. CONFIRMATION NEEDED: One thing to confirm before entering
7. INVALIDATION: What would make this setup fail

Under 180 words. Be specific with NSE India context.""", max_tokens=500)


def ai_ob_analysis(stock, ticker, tf, result) -> str:
    d = result["direction"]
    return ai(f"""NSE India price action alert — Order Block detected.

Stock: {stock} ({ticker}) | Timeframe: {tf}
Direction: {d} Order Block
OB Range: ₹{result['ob_low']:,.2f} – ₹{result['ob_high']:,.2f}
OB Midpoint: ₹{result['ob_mid']:,.2f}
Current price: ₹{result['current_price']:,.2f}
{"Price IS inside OB zone" if result['in_zone'] else "Price approaching OB zone"}
Original impulse: {result['impulse_pct']:.1f}% ({result['candles_ago']} candles ago)
RSI: {result['rsi']}
ATR: ₹{result['atr']:,.2f}

An order block is where institutions placed bulk orders before a major move.
Price returning to this zone means unfilled institutional orders are being hit.

Write a trading plan:
1. WHAT IT MEANS: Why this OB zone is significant
2. ENTRY ZONE: ₹{result['ob_low']:,.2f} – ₹{result['ob_mid']:,.2f} (best entry range)
3. STOP-LOSS: ₹{result['stop_loss']:,.2f}
4. TARGETS: ₹{result['target_1']:,.2f} and ₹{result['target_2']:,.2f}
5. ZONE STRENGTH: Is this OB likely to hold? What makes it strong/weak?
6. INVALIDATION: Price closing below ₹{result['ob_low']:,.2f} invalidates the setup

Under 180 words.""", max_tokens=500)


def ai_msb_analysis(stock, ticker, tf, result) -> str:
    d   = result["direction"]
    prev = result["previous_trend"]
    return ai(f"""NSE India price action alert — Market Structure Break (MSB).

Stock: {stock} ({ticker}) | Timeframe: {tf}
Signal: {prev} just broke {d}
Broken level: ₹{result['broken_level']:,.2f} (this was the last {"Higher Low" if d=="BEARISH" else "Lower High"})
Current price: ₹{result['current_price']:,.2f}
Break size: ₹{result['break_size']:,.2f} ({result['break_pct']:.2f}%)
Bars since break: {result['bars_ago']}
RSI: {result['rsi']}
ATR: ₹{result['atr']:,.2f}
Price {"above" if result['above_ema200'] else "below"} EMA200

This is an EARLY trend change signal — fires before most indicators confirm.
The {prev.lower()} is breaking down/up at the structural level.

Write a trading plan:
1. SIGNIFICANCE: Why breaking ₹{result['broken_level']:,.2f} matters
2. IS IT REAL?: How to confirm this is a genuine break vs fake-out
3. ENTRY: Where to enter — now or wait for retest of broken level?
4. STOP-LOSS: ₹{result['stop_loss']:,.2f}
5. TARGETS: ₹{result['target_1']:,.2f} and ₹{result['target_2']:,.2f}
6. TIME SENSITIVITY: How quickly to act on this signal

Under 180 words. NSE India context.""", max_tokens=500)

# ══════════════════════════════════════════════════════════════════════════════
# TELEGRAM ALERT BUILDERS
# ══════════════════════════════════════════════════════════════════════════════

def send_sweep_alert(stock, ticker, tf, result, analysis):
    d     = result["direction"]
    emoji = "🟢⚡" if d == "BULLISH" else "🔴⚡"
    label = "BULLISH STOP HUNT" if d == "BULLISH" else "BEARISH STOP HUNT"

    msg = (
        f"{emoji} <b>LIQUIDITY SWEEP — {label}</b>\n"
        f"{'─'*35}\n"
        f"<b>Stock:</b>       {stock} ({ticker})\n"
        f"<b>Timeframe:</b>   {tf}\n"
        f"<b>Swept level:</b> ₹{result['swept_level']:,.2f}\n"
        f"<b>Swept to:</b>    ₹{result.get('sweep_low') or result.get('sweep_high'):,.2f}\n"
        f"<b>Now at:</b>      ₹{result['current_price']:,.2f}\n"
        f"<b>Volume:</b>      {result['volume_ratio']:.1f}x average\n"
        f"<b>RSI:</b>         {result['rsi']}\n\n"
        f"💡 <b>What happened:</b> Retail stops below ₹{result['swept_level']:,.2f} "
        f"were triggered. Institutions bought the dip and reversed price.\n\n"
        f"<b>📐 Quick trade plan:</b>\n"
        f"Entry:   ₹{result['current_price']:,.2f}\n"
        f"Stop:    ₹{result['stop_loss']:,.2f}\n"
        f"T1:      ₹{result['target_1']:,.2f}\n"
        f"T2:      ₹{result['target_2']:,.2f}\n\n"
        f"<b>🤖 AI Analysis:</b>\n{analysis}\n\n"
        f"{'─'*35}\n"
        f"⚠️ Educational only. Not financial advice.\n"
        f"🕐 {ts()}"
    )
    send(msg)


def send_ob_alert(stock, ticker, tf, result, analysis):
    d     = result["direction"]
    emoji = "🟩🏦" if d == "BULLISH" else "🟥🏦"
    label = "BULLISH ORDER BLOCK" if d == "BULLISH" else "BEARISH ORDER BLOCK"
    zone  = "INSIDE ZONE ✅" if result["in_zone"] else "APPROACHING ZONE"

    msg = (
        f"{emoji} <b>ORDER BLOCK — {label}</b>\n"
        f"{'─'*35}\n"
        f"<b>Stock:</b>       {stock} ({ticker})\n"
        f"<b>Timeframe:</b>   {tf}\n"
        f"<b>OB Zone:</b>     ₹{result['ob_low']:,.2f} – ₹{result['ob_high']:,.2f}\n"
        f"<b>OB Midpoint:</b> ₹{result['ob_mid']:,.2f}\n"
        f"<b>Current:</b>     ₹{result['current_price']:,.2f} [{zone}]\n"
        f"<b>Impulse was:</b> {result['impulse_pct']:.1f}% ({result['candles_ago']} candles ago)\n"
        f"<b>RSI:</b>         {result['rsi']}\n\n"
        f"💡 <b>What this means:</b> Institutions placed bulk orders in the "
        f"₹{result['ob_low']:,.2f}–{result['ob_high']:,.2f} zone. "
        f"Unfilled orders here act as {'support' if d=='BULLISH' else 'resistance'}.\n\n"
        f"<b>📐 Quick trade plan:</b>\n"
        f"Entry zone: ₹{result['ob_low']:,.2f} – ₹{result['ob_mid']:,.2f}\n"
        f"Stop:       ₹{result['stop_loss']:,.2f}\n"
        f"T1:         ₹{result['target_1']:,.2f}\n"
        f"T2:         ₹{result['target_2']:,.2f}\n\n"
        f"<b>🤖 AI Analysis:</b>\n{analysis}\n\n"
        f"{'─'*35}\n"
        f"⚠️ Educational only. Not financial advice.\n"
        f"🕐 {ts()}"
    )
    send(msg)


def send_msb_alert(stock, ticker, tf, result, analysis):
    d     = result["direction"]
    prev  = result["previous_trend"]
    emoji = "🔻📊" if d == "BEARISH" else "🔺📊"

    msg = (
        f"{emoji} <b>MARKET STRUCTURE BREAK — {d}</b>\n"
        f"{'─'*35}\n"
        f"<b>Stock:</b>         {stock} ({ticker})\n"
        f"<b>Timeframe:</b>     {tf}\n"
        f"<b>Previous trend:</b> {prev}\n"
        f"<b>Broken level:</b>  ₹{result['broken_level']:,.2f} "
        f"({'last Higher Low' if d=='BEARISH' else 'last Lower High'})\n"
        f"<b>Current price:</b> ₹{result['current_price']:,.2f}\n"
        f"<b>Break size:</b>    ₹{result['break_size']:,.2f} ({result['break_pct']:.2f}%)\n"
        f"<b>RSI:</b>           {result['rsi']}\n"
        f"<b>EMA200:</b>        {'Above ✅' if result['above_ema200'] else 'Below ⚠️'}\n\n"
        f"⚡ <b>Early warning:</b> This fires 2–5 candles before EMA crossovers confirm.\n\n"
        f"<b>📐 Quick trade plan:</b>\n"
        f"Stop:  ₹{result['stop_loss']:,.2f}\n"
        f"T1:    ₹{result['target_1']:,.2f}\n"
        f"T2:    ₹{result['target_2']:,.2f}\n\n"
        f"<b>🤖 AI Analysis:</b>\n{analysis}\n\n"
        f"{'─'*35}\n"
        f"⚠️ Educational only. Not financial advice.\n"
        f"🕐 {ts()}"
    )
    send(msg)

# ══════════════════════════════════════════════════════════════════════════════
# MAIN SCAN
# ══════════════════════════════════════════════════════════════════════════════

def scan_structure():
    now_ist = datetime.now(IST)

    # Market hours check
    if now_ist.weekday() >= 5:
        print(f"Weekend — skipping structure scan"); return

    h = now_ist.hour
    if not (8 <= h < 23):
        print(f"Outside hours ({now_ist.strftime('%H:%M')}) — skipping"); return

    print(f"\n{'='*55}")
    print(f"Structure Scan — {now_ist.strftime('%d %b %Y %H:%M IST')}")
    print(f"Scanning {len(WATCHLIST)} stocks × {len(TIMEFRAMES)} timeframes")
    print(f"{'='*55}")

    total_alerts = 0

    for stock_name, ticker in WATCHLIST.items():
        print(f"\n  {stock_name}...")

        for tf_name, tf_cfg in TIMEFRAMES.items():
            df = fetch(ticker, tf_cfg["period"], tf_cfg["interval"])
            if df.empty or len(df) < 25:
                continue

            df = add_basic_indicators(df)

            # ── 1. Liquidity Sweep ────────────────────────
            sweep = detect_liquidity_sweep(df, stock_name, tf_name)
            if sweep and not already_alerted(stock_name, "LIQUIDITY_SWEEP", tf_name):
                d = sweep["direction"]
                print(f"    ⚡ SWEEP {d} ({tf_name}) — swept ₹{sweep['swept_level']:,.2f}")
                try:
                    analysis = ai_sweep_analysis(stock_name, ticker, tf_name, sweep)
                except Exception as e:
                    analysis = f"AI unavailable: {e}"
                send_sweep_alert(stock_name, ticker, tf_name, sweep, analysis)
                save_alert(stock_name, ticker, tf_name, "LIQUIDITY_SWEEP", d,
                           sweep["current_price"], sweep["swept_level"],
                           f"Swept ₹{sweep['swept_level']:,.2f}, now ₹{sweep['current_price']:,.2f}",
                           analysis)
                total_alerts += 1
                time.sleep(2)

            # ── 2. Order Block ────────────────────────────
            ob = detect_order_block(df, stock_name, tf_name)
            if ob and not already_alerted(stock_name, "ORDER_BLOCK", tf_name):
                d = ob["direction"]
                zone_str = "INSIDE" if ob["in_zone"] else "NEAR"
                print(f"    🏦 ORDER BLOCK {d} ({tf_name}) — {zone_str} ₹{ob['ob_low']:,.2f}–{ob['ob_high']:,.2f}")
                try:
                    analysis = ai_ob_analysis(stock_name, ticker, tf_name, ob)
                except Exception as e:
                    analysis = f"AI unavailable: {e}"
                send_ob_alert(stock_name, ticker, tf_name, ob, analysis)
                save_alert(stock_name, ticker, tf_name, "ORDER_BLOCK", d,
                           ob["current_price"], ob["ob_mid"],
                           f"OB zone ₹{ob['ob_low']:,.2f}–{ob['ob_high']:,.2f}",
                           analysis)
                total_alerts += 1
                time.sleep(2)

            # ── 3. Market Structure Break ─────────────────
            msb = detect_msb(df, stock_name, tf_name)
            if msb and not already_alerted(stock_name, "MARKET_STRUCTURE_BREAK", tf_name):
                d = msb["direction"]
                print(f"    📊 MSB {d} ({tf_name}) — broke ₹{msb['broken_level']:,.2f}")
                try:
                    analysis = ai_msb_analysis(stock_name, ticker, tf_name, msb)
                except Exception as e:
                    analysis = f"AI unavailable: {e}"
                send_msb_alert(stock_name, ticker, tf_name, msb, analysis)
                save_alert(stock_name, ticker, tf_name, "MARKET_STRUCTURE_BREAK", d,
                           msb["current_price"], msb["broken_level"],
                           f"Broke ₹{msb['broken_level']:,.2f}, now ₹{msb['current_price']:,.2f}",
                           analysis)
                total_alerts += 1
                time.sleep(2)

    print(f"\n{'='*55}")
    print(f"Structure scan complete — {total_alerts} alerts sent")
    print(f"{'='*55}\n")


# ── STREAMLIT TAB ─────────────────────────────────────────────────────────────
def render_structure_tab():
    import streamlit as st
    st.subheader("🏗️ Structure Scanner")
    st.caption("Liquidity sweeps · Order blocks · Market structure breaks")

    col1, col2, col3 = st.columns(3)
    col1.metric("Liquidity Sweeps",  "Stop hunt detector")
    col2.metric("Order Blocks",      "Institutional zones")
    col3.metric("MSB",               "Early trend change")

    st.divider()

    try:
        hist = load_structure_history(100)
        if hist.empty:
            st.info("No structure alerts yet. Run structure_scanner.py to start.")
            st.code("python structure_scanner.py")
            return

        # Filter controls
        f1, f2, f3 = st.columns(3)
        filter_type  = f1.selectbox("Alert type", ["All","LIQUIDITY_SWEEP","ORDER_BLOCK","MARKET_STRUCTURE_BREAK"])
        filter_dir   = f2.selectbox("Direction",  ["All","BULLISH","BEARISH"])
        filter_stock = f3.selectbox("Stock",       ["All"] + sorted(hist["stock"].unique().tolist()))

        filtered = hist.copy()
        if filter_type  != "All": filtered = filtered[filtered["alert_type"] == filter_type]
        if filter_dir   != "All": filtered = filtered[filtered["direction"]  == filter_dir]
        if filter_stock != "All": filtered = filtered[filtered["stock"]      == filter_stock]

        # Summary
        c1,c2,c3,c4 = st.columns(4)
        c1.metric("Total alerts",  len(filtered))
        c2.metric("Sweeps",        len(filtered[filtered["alert_type"]=="LIQUIDITY_SWEEP"]))
        c3.metric("Order blocks",  len(filtered[filtered["alert_type"]=="ORDER_BLOCK"]))
        c4.metric("MSB",           len(filtered[filtered["alert_type"]=="MARKET_STRUCTURE_BREAK"]))

        st.divider()

        # Alert cards
        type_emoji = {
            "LIQUIDITY_SWEEP":        "⚡",
            "ORDER_BLOCK":            "🏦",
            "MARKET_STRUCTURE_BREAK": "📊",
        }
        for _, row in filtered.head(30).iterrows():
            icon  = type_emoji.get(row["alert_type"], "📌")
            color = "🟢" if row["direction"] == "BULLISH" else "🔴"
            label = row["alert_type"].replace("_"," ").title()
            with st.expander(
                f"{icon}{color} {label} — {row['stock']} ({row['timeframe']}) "
                f"@ ₹{row['price']:,.2f} · {row['timestamp'][:16]}"
            ):
                st.markdown(f"**Key level:** ₹{row['key_level']:,.2f}")
                st.markdown(f"**Description:** {row['description']}")
                if row["analysis"]:
                    st.markdown(
                        f'<div style="background:#1a1f2e;border-left:4px solid #6c63ff;'
                        f'border-radius:8px;padding:14px;margin:8px 0;font-size:13px;'
                        f'line-height:1.8;color:#e0e0e0">'
                        f'{row["analysis"].replace(chr(10),"<br>")}</div>',
                        unsafe_allow_html=True)

    except Exception as e:
        st.error(f"Error loading structure history: {e}")


# ── ENTRY POINT ───────────────────────────────────────────────────────────────
if __name__ == "__main__":
    now_ist = datetime.now(IST)
    print("="*55)
    print("NSE Structure Scanner")
    print(f"Time: {now_ist.strftime('%d %b %Y %H:%M IST')}")
    print("Detects: Liquidity Sweeps · Order Blocks · MSB")
    print("="*55)

    if "YOUR_" in TELEGRAM_TOKEN or "YOUR_" in CHAT_ID or "YOUR_" in GROQ_API_KEY:
        print("\n⚠️  Fill in TELEGRAM_TOKEN, CHAT_ID and GROQ_API_KEY at the top.\n")
        exit(1)

    init_db()

    print("\n1 — Scan now (test immediately)")
    print("2 — Start scheduler (every 2 hours during market hours)\n")
    choice = input("Enter 1 or 2: ").strip()

    if choice == "2":
        print("\nScheduler started.")
        print("Scans every 2 hours: 9:30, 11:30, 1:30, 3:30 PM IST")
        print("Press Ctrl+C to stop.\n")
        schedule.every().day.at("09:30").do(scan_structure)
        schedule.every().day.at("11:30").do(scan_structure)
        schedule.every().day.at("13:30").do(scan_structure)
        schedule.every().day.at("15:30").do(scan_structure)
        # Run once immediately
        scan_structure()
        while True:
            schedule.run_pending()
            time.sleep(60)
    else:
        scan_structure()
