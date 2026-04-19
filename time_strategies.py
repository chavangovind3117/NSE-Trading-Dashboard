"""
Time-Based Strategies — time_strategies.py
===========================================
Three strategies that exploit market time patterns:

1. OPENING RANGE BREAKOUT (ORB) — fires at 9:30 AM
   Tracks the high/low of first 15 minutes (9:15–9:30 AM).
   Alerts when price breaks out with volume in next 30 minutes.
   Predicts intraday direction with ~65% accuracy on NIFTY 50 stocks.

2. EXPIRY WEEK BIAS — fires every Monday morning
   NSE monthly expiry is last Thursday of the month.
   Detects max pain level from options OI and sends weekly bias.
   Price gravitates toward max pain — gives directional edge for the week.

3. POST-EARNINGS DRIFT (PEAD) — fires after quarterly results
   Stocks that beat earnings continue drifting in surprise direction for 30-60 days.
   Tracks stocks that reported strong results and monitors drift continuation.
   Academically proven anomaly that persists in Indian markets.

All three send Telegram alerts at the right time automatically.
All data saved to nse_trading.db.

Run: python time_strategies.py
Schedule:
  9:15 AM — Start ORB tracking (every weekday)
  9:30 AM — Send ORB breakout alerts (every weekday)
  9:00 AM Monday — Expiry week bias report
  6:00 PM daily — Check PEAD continuation signals
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
import calendar
from config import TELEGRAM_TOKEN, CHAT_ID, GROQ_API_KEY, GROQ_MODEL

# ── CONFIG ────────────────────────────────────────────────────────────────────
IST     = pytz.timezone("Asia/Kolkata")
DB_PATH = "nse_trading.db"

# Stocks for ORB — large caps with high liquidity only
ORB_WATCHLIST = {
    "NIFTY 50":      "^NSEI",
    "BANK NIFTY":    "^NSEBANK",
    "Reliance":      "RELIANCE.NS",
    "HDFC Bank":     "HDFCBANK.NS",
    "ICICI Bank":    "ICICIBANK.NS",
    "SBI":           "SBIN.NS",
    "TCS":           "TCS.NS",
    "Infosys":       "INFY.NS",
    "Bajaj Finance": "BAJFINANCE.NS",
    "Axis Bank":     "AXISBANK.NS",
    "Kotak Bank":    "KOTAKBANK.NS",
    "Tata Motors":   "TATAMOTORS.NS",
    "Maruti":        "MARUTI.NS",
    "L&T":           "LT.NS",
    "Titan":         "TITAN.NS",
}

# Stocks to track for PEAD — after earnings beats
PEAD_WATCHLIST = {
    "Reliance":      "RELIANCE.NS",
    "TCS":           "TCS.NS",
    "HDFC Bank":     "HDFCBANK.NS",
    "Infosys":       "INFY.NS",
    "ICICI Bank":    "ICICIBANK.NS",
    "SBI":           "SBIN.NS",
    "Bajaj Finance": "BAJFINANCE.NS",
    "Wipro":         "WIPRO.NS",
    "Asian Paints":  "ASIANPAINT.NS",
    "Titan":         "TITAN.NS",
    "Maruti":        "MARUTI.NS",
    "Sun Pharma":    "SUNPHARMA.NS",
    "Zomato":        "ZOMATO.NS",
    "Dixon Tech":    "DIXON.NS",
    "IRCTC":         "IRCTC.NS",
}

# ── DATABASE ──────────────────────────────────────────────────────────────────
def init_db():
    conn = sqlite3.connect(DB_PATH)
    conn.execute("""CREATE TABLE IF NOT EXISTS orb_data (
        id        INTEGER PRIMARY KEY AUTOINCREMENT,
        date      TEXT,
        stock     TEXT,
        ticker    TEXT,
        orb_high  REAL,
        orb_low   REAL,
        orb_size  REAL,
        breakout  TEXT,
        break_price REAL,
        target_1  REAL,
        target_2  REAL,
        stop_loss REAL,
        alerted   INTEGER DEFAULT 0
    )""")
    conn.execute("""CREATE TABLE IF NOT EXISTS expiry_bias (
        id         INTEGER PRIMARY KEY AUTOINCREMENT,
        date       TEXT,
        expiry_date TEXT,
        max_pain   REAL,
        nifty_close REAL,
        bias       TEXT,
        pcr        REAL,
        analysis   TEXT
    )""")
    conn.execute("""CREATE TABLE IF NOT EXISTS pead_tracker (
        id             INTEGER PRIMARY KEY AUTOINCREMENT,
        entry_date     TEXT,
        stock          TEXT,
        ticker         TEXT,
        result_date    TEXT,
        surprise_type  TEXT,
        entry_price    REAL,
        current_price  REAL,
        drift_pct      REAL,
        days_held      INTEGER,
        status         TEXT,
        last_updated   TEXT
    )""")
    conn.commit()
    conn.close()


def save_orb(date, stock, ticker, orb_high, orb_low):
    conn = sqlite3.connect(DB_PATH)
    conn.execute("""INSERT OR REPLACE INTO orb_data
        (date,stock,ticker,orb_high,orb_low,orb_size)
        VALUES (?,?,?,?,?,?)""",
        (date, stock, ticker, orb_high, orb_low, orb_high - orb_low))
    conn.commit(); conn.close()


def update_orb_breakout(date, stock, breakout, break_price, t1, t2, sl):
    conn = sqlite3.connect(DB_PATH)
    conn.execute("""UPDATE orb_data SET
        breakout=?, break_price=?, target_1=?, target_2=?, stop_loss=?, alerted=1
        WHERE date=? AND stock=?""",
        (breakout, break_price, t1, t2, sl, date, stock))
    conn.commit(); conn.close()


def orb_alerted_today(stock):
    conn  = sqlite3.connect(DB_PATH)
    today = datetime.now(IST).strftime("%Y-%m-%d")
    row   = conn.execute(
        "SELECT id FROM orb_data WHERE date=? AND stock=? AND alerted=1",
        (today, stock)).fetchone()
    conn.close()
    return row is not None


def save_expiry_bias(date, expiry_date, max_pain, nifty_close, bias, pcr, analysis):
    conn = sqlite3.connect(DB_PATH)
    conn.execute("""INSERT INTO expiry_bias
        (date,expiry_date,max_pain,nifty_close,bias,pcr,analysis)
        VALUES (?,?,?,?,?,?,?)""",
        (date, expiry_date, max_pain, nifty_close, bias, pcr, analysis))
    conn.commit(); conn.close()


def add_pead_stock(stock, ticker, result_date, surprise_type, entry_price):
    conn = sqlite3.connect(DB_PATH)
    today = datetime.now(IST).strftime("%Y-%m-%d")
    conn.execute("""INSERT INTO pead_tracker
        (entry_date,stock,ticker,result_date,surprise_type,entry_price,
         current_price,drift_pct,days_held,status,last_updated)
        VALUES (?,?,?,?,?,?,?,?,?,?,?)""",
        (today, stock, ticker, result_date, surprise_type, entry_price,
         entry_price, 0.0, 0, "ACTIVE", today))
    conn.commit(); conn.close()


def load_pead_active():
    conn = sqlite3.connect(DB_PATH)
    df   = pd.read_sql_query(
        "SELECT * FROM pead_tracker WHERE status='ACTIVE' ORDER BY entry_date DESC",
        conn)
    conn.close()
    return df


def update_pead(row_id, current_price, drift_pct, days_held, status):
    conn = sqlite3.connect(DB_PATH)
    conn.execute("""UPDATE pead_tracker SET
        current_price=?, drift_pct=?, days_held=?, status=?, last_updated=?
        WHERE id=?""",
        (current_price, drift_pct, days_held, status,
         datetime.now(IST).strftime("%Y-%m-%d"), row_id))
    conn.commit(); conn.close()


# ── TELEGRAM ──────────────────────────────────────────────────────────────────
def send(text: str):
    url = f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/sendMessage"
    for chunk in [text[i:i+4000] for i in range(0, len(text), 4000)]:
        try:
            requests.post(url, json={
                "chat_id": CHAT_ID, "text": chunk, "parse_mode": "HTML"
            }, timeout=15)
        except Exception as e:
            print(f"  Telegram: {e}")
        time.sleep(0.5)

def ts(): return datetime.now(IST).strftime("%d %b %Y %H:%M IST")

# ── GROQ ──────────────────────────────────────────────────────────────────────
def ai(prompt: str, max_tokens=400) -> str:
    try:
        r = Groq(api_key=GROQ_API_KEY).chat.completions.create(
            model=GROQ_MODEL,
            messages=[
                {"role": "system", "content": "You are an expert NSE India trader. Be specific, concise, actionable."},
                {"role": "user",   "content": prompt}
            ],
            temperature=0.3, max_tokens=max_tokens)
        return r.choices[0].message.content.strip()
    except Exception as e:
        return f"AI unavailable: {e}"


def fetch(ticker, period, interval):
    try:
        df = yf.download(ticker, period=period, interval=interval,
                         progress=False, auto_adjust=True)
        if df.empty: return df
        df.columns = [c[0] if isinstance(c, tuple) else c for c in df.columns]
        df.dropna(inplace=True)
        return df
    except Exception:
        return pd.DataFrame()


# ══════════════════════════════════════════════════════════════════════════════
# MODULE 1 — OPENING RANGE BREAKOUT (ORB)
# ══════════════════════════════════════════════════════════════════════════════

def get_nifty_direction() -> str:
    """Quick check if NIFTY is positive or negative today."""
    try:
        df = fetch("^NSEI", "2d", "1d")
        if df.empty or len(df) < 2: return "NEUTRAL"
        chg = float(df["Close"].iloc[-1]) - float(df["Close"].iloc[-2])
        return "POSITIVE" if chg > 0 else "NEGATIVE"
    except Exception:
        return "NEUTRAL"


def capture_orb():
    """
    Step 1 — called at 9:30 AM.
    Gets the 5-min data for today, finds the high/low of first 3 candles
    (9:15, 9:20, 9:25 = the 9:15–9:30 opening range), saves to DB.
    """
    now_ist = datetime.now(IST)
    today   = now_ist.strftime("%Y-%m-%d")
    print(f"\n[ORB] Capturing opening range — {now_ist.strftime('%H:%M IST')}")

    nifty_dir = get_nifty_direction()
    print(f"  NIFTY direction: {nifty_dir}")

    orb_ranges = {}
    for stock_name, ticker in ORB_WATCHLIST.items():
        try:
            df = fetch(ticker, "1d", "5m")
            if df.empty or len(df) < 3:
                continue

            # First 3 candles = 9:15, 9:20, 9:25 AM (opening range)
            orb_candles = df.head(3)
            orb_high    = float(orb_candles["High"].max())
            orb_low     = float(orb_candles["Low"].min())
            orb_size    = orb_high - orb_low

            save_orb(today, stock_name, ticker, orb_high, orb_low)
            orb_ranges[stock_name] = {
                "ticker":    ticker,
                "orb_high":  orb_high,
                "orb_low":   orb_low,
                "orb_size":  orb_size,
                "last_close": float(df["Close"].iloc[-1]),
            }
            print(f"  {stock_name}: ORB ₹{orb_low:,.2f} – ₹{orb_high:,.2f} (size ₹{orb_size:,.2f})")

        except Exception as e:
            print(f"  {stock_name} error: {e}")

    # Store in module-level dict for the 9:45/10:00 breakout check
    return orb_ranges, nifty_dir


# Shared state for ORB (populated at 9:30, checked at 9:45 and 10:00)
_orb_ranges   = {}
_nifty_dir    = "NEUTRAL"


def check_orb_breakouts():
    """
    Step 2 — called at 9:45 AM and 10:00 AM.
    Checks if price has broken out of the ORB range with volume.
    Sends Telegram alert for each breakout.
    """
    global _orb_ranges, _nifty_dir

    if not _orb_ranges:
        print("[ORB] No ORB ranges captured yet — run capture first")
        return

    today   = datetime.now(IST).strftime("%Y-%m-%d")
    alerted = 0

    for stock_name, orb in _orb_ranges.items():
        if orb_alerted_today(stock_name):
            continue

        try:
            df = fetch(orb["ticker"], "1d", "5m")
            if df.empty or len(df) < 5:
                continue

            # Use candles after the opening range (from 9:30 onwards)
            post_orb = df.iloc[3:]
            if post_orb.empty:
                continue

            latest_close  = float(post_orb["Close"].iloc[-1])
            latest_high   = float(post_orb["High"].iloc[-1])
            latest_low    = float(post_orb["Low"].iloc[-1])
            latest_vol    = float(post_orb["Volume"].iloc[-1]) if "Volume" in post_orb.columns else 0
            avg_vol       = float(df["Volume"].mean()) if "Volume" in df.columns else 1
            vol_ratio     = latest_vol / avg_vol if avg_vol > 0 else 1.0

            orb_high = orb["orb_high"]
            orb_low  = orb["orb_low"]
            orb_size = orb["orb_size"]

            # Bullish breakout: close above ORB high with volume
            if latest_close > orb_high and vol_ratio > 1.2:
                breakout    = "BULLISH"
                break_price = latest_close
                t1 = round(orb_high + orb_size,       2)
                t2 = round(orb_high + orb_size * 2,   2)
                sl = round(orb_low,                    2)

            # Bearish breakdown: close below ORB low with volume
            elif latest_close < orb_low and vol_ratio > 1.2:
                breakout    = "BEARISH"
                break_price = latest_close
                t1 = round(orb_low  - orb_size,       2)
                t2 = round(orb_low  - orb_size * 2,   2)
                sl = round(orb_high,                   2)
            else:
                continue

            # Filter: only alert if NIFTY direction matches (avoid counter-trend)
            if breakout == "BULLISH" and _nifty_dir == "NEGATIVE":
                print(f"  {stock_name}: BULLISH ORB but NIFTY negative — skipping")
                continue
            if breakout == "BEARISH" and _nifty_dir == "POSITIVE":
                print(f"  {stock_name}: BEARISH ORB but NIFTY positive — skipping")
                continue

            update_orb_breakout(today, stock_name, breakout,
                                 break_price, t1, t2, sl)

            # AI analysis
            analysis = ai(
                f"NSE ORB breakout alert.\n"
                f"Stock: {stock_name} | Time: {datetime.now(IST).strftime('%H:%M IST')}\n"
                f"ORB range: ₹{orb_low:,.2f} – ₹{orb_high:,.2f} (size ₹{orb_size:,.2f})\n"
                f"Breakout: {breakout} at ₹{break_price:,.2f}\n"
                f"Volume: {vol_ratio:.1f}x average\n"
                f"NIFTY today: {_nifty_dir}\n\n"
                f"1. CONVICTION: Is this a strong or weak ORB breakout? Why?\n"
                f"2. ENTRY: Enter now or wait for retest of ORB level?\n"
                f"3. HOLD TIME: Intraday only or can hold overnight?\n"
                f"4. CAUTION: One thing that could make this fail\n\n"
                f"Under 120 words.")

            emoji = "🟢" if breakout == "BULLISH" else "🔴"
            direction_word = "ABOVE" if breakout == "BULLISH" else "BELOW"
            send(
                f"{emoji}⏰ <b>ORB BREAKOUT — {breakout}</b>\n"
                f"{'─'*32}\n"
                f"<b>Stock:</b>     {stock_name}\n"
                f"<b>Time:</b>      {datetime.now(IST).strftime('%H:%M IST')}\n"
                f"<b>ORB Range:</b> ₹{orb_low:,.2f} – ₹{orb_high:,.2f}\n"
                f"<b>Broke {direction_word}:</b> ₹{break_price:,.2f}\n"
                f"<b>ORB size:</b>  ₹{orb_size:,.2f}\n"
                f"<b>Volume:</b>    {vol_ratio:.1f}x avg\n"
                f"<b>NIFTY:</b>     {_nifty_dir}\n\n"
                f"<b>📐 Trade plan:</b>\n"
                f"Entry: ₹{break_price:,.2f}\n"
                f"Stop:  ₹{sl:,.2f} (opposite end of ORB)\n"
                f"T1:    ₹{t1:,.2f} (1x ORB size)\n"
                f"T2:    ₹{t2:,.2f} (2x ORB size)\n\n"
                f"<b>🤖 AI:</b>\n{analysis}\n\n"
                f"⚠️ Educational only.\n🕐 {ts()}"
            )

            print(f"  ✅ ORB alert: {stock_name} {breakout} at ₹{break_price:,.2f}")
            alerted += 1
            time.sleep(2)

        except Exception as e:
            print(f"  {stock_name} check error: {e}")

    print(f"  ORB check complete — {alerted} breakout alerts")


def run_orb_capture():
    """Called at 9:30 AM — captures the opening range."""
    global _orb_ranges, _nifty_dir
    now = datetime.now(IST)
    if now.weekday() >= 5: return
    _orb_ranges, _nifty_dir = capture_orb()


def run_orb_check():
    """Called at 9:45 AM and 10:00 AM — checks for breakouts."""
    now = datetime.now(IST)
    if now.weekday() >= 5: return
    check_orb_breakouts()


# ══════════════════════════════════════════════════════════════════════════════
# MODULE 2 — EXPIRY WEEK BIAS
# ══════════════════════════════════════════════════════════════════════════════

def get_monthly_expiry_date() -> datetime:
    """
    NSE monthly expiry = last Thursday of the month.
    If today is past expiry, get next month's expiry.
    """
    now     = datetime.now(IST)
    year    = now.year
    month   = now.month

    def last_thursday(y, m):
        last_day = calendar.monthrange(y, m)[1]
        for day in range(last_day, 0, -1):
            if datetime(y, m, day).weekday() == 3:  # Thursday = 3
                return datetime(y, m, day)

    expiry = last_thursday(year, month)

    # If we're past this month's expiry, use next month
    if now.date() > expiry.date():
        if month == 12:
            expiry = last_thursday(year + 1, 1)
        else:
            expiry = last_thursday(year, month + 1)

    return expiry


def is_expiry_week() -> bool:
    """True if current week contains the monthly expiry."""
    now    = datetime.now(IST)
    expiry = get_monthly_expiry_date()
    # Same week = within 7 days of expiry
    days_to_expiry = (expiry.date() - now.date()).days
    return 0 <= days_to_expiry <= 6


def fetch_pcr_and_max_pain() -> dict:
    """
    Fetch NIFTY options OI from NSE to calculate:
    - Put-Call Ratio (PCR)
    - Max Pain level (strike where most options expire worthless)

    NSE provides this via their option chain API.
    """
    result = {"pcr": None, "max_pain": None, "total_call_oi": 0, "total_put_oi": 0}

    try:
        session = requests.Session()
        session.headers.update({
            "User-Agent":      "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36",
            "Accept":          "application/json, text/plain, */*",
            "Referer":         "https://www.nseindia.com/",
            "Accept-Language": "en-US,en;q=0.9",
        })

        # Warm up session
        session.get("https://www.nseindia.com/", timeout=10)
        time.sleep(1.5)

        # Fetch option chain
        r = session.get(
            "https://www.nseindia.com/api/option-chain-indices?symbol=NIFTY",
            timeout=20)

        if r.status_code != 200:
            print(f"  Options API: {r.status_code}")
            return result

        data   = r.json()
        records = data.get("records", {})
        oc_data = records.get("data", [])

        if not oc_data:
            return result

        # Calculate PCR and max pain
        total_call_oi = 0
        total_put_oi  = 0
        pain_data     = {}  # strike -> total pain for longs

        for item in oc_data:
            strike = item.get("strikePrice", 0)
            ce     = item.get("CE", {})
            pe     = item.get("PE", {})

            call_oi = ce.get("openInterest", 0) or 0
            put_oi  = pe.get("openInterest", 0) or 0

            total_call_oi += call_oi
            total_put_oi  += put_oi

            if strike > 0:
                pain_data[strike] = {"call_oi": call_oi, "put_oi": put_oi}

        # PCR
        if total_call_oi > 0:
            result["pcr"]           = round(total_put_oi / total_call_oi, 2)
        result["total_call_oi"] = total_call_oi
        result["total_put_oi"]  = total_put_oi

        # Max pain: strike where total loss for option buyers is maximum
        # = where most call + put OI is concentrated (options expire worthless)
        strikes = sorted(pain_data.keys())
        if strikes:
            pain_at_strike = {}
            for s in strikes:
                call_pain = sum(
                    pain_data[k]["call_oi"] * max(0, k - s)
                    for k in strikes if k > s)
                put_pain  = sum(
                    pain_data[k]["put_oi"] * max(0, s - k)
                    for k in strikes if k < s)
                pain_at_strike[s] = call_pain + put_pain

            # Max pain = strike with MINIMUM total pain (writers' sweet spot)
            result["max_pain"] = min(pain_at_strike, key=pain_at_strike.get)

        print(f"  PCR: {result['pcr']} | Max Pain: ₹{result['max_pain']:,.0f}")
        return result

    except Exception as e:
        print(f"  Options fetch error: {e}")
        return result


def run_expiry_week_bias():
    """
    Runs every Monday at 9:00 AM.
    If it's expiry week, sends full bias report.
    If not expiry week, sends a brief heads-up on days to expiry.
    """
    now_ist = datetime.now(IST)
    if now_ist.weekday() >= 5: return  # safety check

    today       = now_ist.strftime("%Y-%m-%d")
    expiry_date = get_monthly_expiry_date()
    days_left   = (expiry_date.date() - now_ist.date()).days

    print(f"\n[EXPIRY] Checking expiry week bias — {now_ist.strftime('%H:%M IST')}")
    print(f"  Next expiry: {expiry_date.strftime('%d %b %Y')} ({days_left} days)")

    # Get current NIFTY level
    nifty_df    = fetch("^NSEI", "5d", "1d")
    nifty_close = float(nifty_df["Close"].iloc[-1]) if not nifty_df.empty else 0

    # Get options data
    options_data = fetch_pcr_and_max_pain()
    max_pain     = options_data.get("max_pain")
    pcr          = options_data.get("pcr")

    # ── Determine bias ────────────────────────────────────
    if max_pain and nifty_close:
        gap_to_pain  = max_pain - nifty_close
        gap_pct      = (gap_to_pain / nifty_close) * 100

        if gap_to_pain > nifty_close * 0.005:    # max pain > 0.5% above = bullish pull
            bias       = "BULLISH"
            bias_reason = f"Max pain ₹{max_pain:,.0f} is {gap_pct:+.1f}% ABOVE current price — price tends to drift UP"
        elif gap_to_pain < -nifty_close * 0.005: # max pain > 0.5% below = bearish pull
            bias       = "BEARISH"
            bias_reason = f"Max pain ₹{max_pain:,.0f} is {gap_pct:+.1f}% BELOW current price — price tends to drift DOWN"
        else:
            bias       = "NEUTRAL"
            bias_reason = f"Max pain ₹{max_pain:,.0f} very close to current price — no strong expiry bias"
    else:
        bias       = "UNKNOWN"
        bias_reason = "Options data unavailable — using PCR only"

    # PCR-based override
    if pcr:
        if pcr > 1.3 and bias != "BULLISH":
            bias = "BULLISH (PCR extreme fear)"
        elif pcr < 0.7 and bias != "BEARISH":
            bias = "BEARISH (PCR extreme greed)"

    # ── AI analysis ───────────────────────────────────────
    analysis = ai(
        f"NSE expiry week analysis — {expiry_date.strftime('%d %b %Y')}\n"
        f"Days to expiry: {days_left}\n"
        f"NIFTY: ₹{nifty_close:,.0f}\n"
        f"Max Pain: {'₹'+f'{max_pain:,.0f}' if max_pain else 'unavailable'}\n"
        f"PCR: {pcr if pcr else 'unavailable'}\n"
        f"Bias: {bias}\n"
        f"Reason: {bias_reason}\n\n"
        f"Explain for NSE India trader:\n"
        f"1. WEEK BIAS: {bias} — what this means for NIFTY this week\n"
        f"2. MAX PAIN STRATEGY: How to trade toward max pain if valid\n"
        f"3. KEY LEVELS: Support and resistance for expiry week\n"
        f"4. WHAT TO WATCH: Specific trigger that confirms or invalidates bias\n"
        f"5. OPTION SELLERS VIEW: What premium sellers are positioned for\n\n"
        f"Under 200 words. Specific to NSE India expiry dynamics.",
        max_tokens=600)

    # ── Send alert ────────────────────────────────────────
    expiry_week = is_expiry_week()
    header = "🗓️⚠️ <b>EXPIRY WEEK BIAS REPORT</b>" if expiry_week else "🗓️ <b>WEEKLY EXPIRY UPDATE</b>"

    emoji_bias = {"BULLISH":"🟢","BEARISH":"🔴","NEUTRAL":"⚪","UNKNOWN":"❓"}
    b_emoji    = next((v for k,v in emoji_bias.items() if k in bias), "⚪")

    msg = (
        f"{header}\n{'─'*35}\n"
        f"<b>Expiry date:</b>  {expiry_date.strftime('%d %b %Y')} ({days_left} days)\n"
        f"<b>NIFTY close:</b>  ₹{nifty_close:,.0f}\n"
    )
    if max_pain:
        msg += f"<b>Max Pain:</b>     ₹{max_pain:,.0f}\n"
    if pcr:
        msg += f"<b>PCR:</b>          {pcr:.2f} "
        msg += ("(fear zone ↑)" if pcr > 1.2 else "(greed zone ↓)" if pcr < 0.8 else "(neutral)")
        msg += "\n"

    msg += (
        f"\n{b_emoji} <b>WEEK BIAS: {bias}</b>\n"
        f"{bias_reason}\n\n"
        f"<b>🤖 AI Analysis:</b>\n{analysis}\n\n"
        f"{'─'*35}\n"
        f"⚠️ Educational only. Not financial advice.\n"
        f"🕐 {ts()}"
    )
    send(msg)

    # Save to DB
    save_expiry_bias(today, expiry_date.strftime("%Y-%m-%d"),
                     max_pain or 0, nifty_close, bias, pcr or 0, analysis)

    print(f"  ✅ Expiry bias sent: {bias}")


# ══════════════════════════════════════════════════════════════════════════════
# MODULE 3 — POST-EARNINGS ANNOUNCEMENT DRIFT (PEAD)
# ══════════════════════════════════════════════════════════════════════════════

def add_pead_manually(stock: str, ticker: str, result_date: str,
                      surprise_type: str, entry_price: float):
    """
    Call this manually when you see a strong earnings result.
    surprise_type: "BEAT" or "MISS"
    entry_price: price 3-5 days after results (after initial spike settles)

    Example:
      add_pead_manually("Reliance", "RELIANCE.NS", "2026-04-15", "BEAT", 2850.0)
    """
    add_pead_stock(stock, ticker, result_date, surprise_type, entry_price)
    print(f"Added PEAD tracker: {stock} ({surprise_type}) from ₹{entry_price:,.2f}")


def check_pead_continuation():
    """
    Runs daily at 6 PM.
    Checks all active PEAD positions:
    - Is the drift continuing?
    - Has price reached 2R target?
    - Has it been 60 days? (exit — drift effect fades)
    - Is it still above entry for BEATs? (if not, may be failing)
    Sends update if significant change.
    """
    now_ist = datetime.now(IST)
    if now_ist.weekday() >= 5: return

    active = load_pead_active()
    if active.empty:
        print("[PEAD] No active PEAD positions to track")
        return

    print(f"\n[PEAD] Checking {len(active)} active PEAD positions...")
    updates = []

    for _, row in active.iterrows():
        ticker = row["ticker"]
        stock  = row["stock"]

        try:
            df = fetch(ticker, "5d", "1d")
            if df.empty: continue

            current_price = float(df["Close"].iloc[-1])
            entry_price   = float(row["entry_price"])
            drift_pct     = ((current_price - entry_price) / entry_price) * 100
            entry_date    = datetime.strptime(row["entry_date"], "%Y-%m-%d")
            days_held     = (now_ist.date() - entry_date.date()).days

            surprise = row["surprise_type"]
            status   = "ACTIVE"

            # BEAT: expect upward drift
            # MISS: expect downward drift
            expected_positive = surprise == "BEAT"

            # Exit conditions
            if days_held >= 60:
                status = "EXPIRED"  # PEAD effect fades after 60 days
            elif expected_positive and drift_pct >= 15:
                status = "TARGET_HIT"
            elif not expected_positive and drift_pct <= -15:
                status = "TARGET_HIT"
            elif expected_positive and drift_pct <= -8:
                status = "FAILED"   # thesis broken
            elif not expected_positive and drift_pct >= 8:
                status = "FAILED"

            update_pead(int(row["id"]), current_price, round(drift_pct, 2),
                        days_held, status)

            updates.append({
                "stock":    stock,
                "surprise": surprise,
                "entry":    entry_price,
                "current":  current_price,
                "drift":    drift_pct,
                "days":     days_held,
                "status":   status,
            })

            print(f"  {stock}: ₹{current_price:,.2f} ({drift_pct:+.1f}%) "
                  f"Day {days_held} — {status}")

        except Exception as e:
            print(f"  {stock} error: {e}")

    if not updates:
        return

    # Send daily PEAD summary
    active_updates = [u for u in updates if u["status"] == "ACTIVE"]
    hit_updates    = [u for u in updates if u["status"] in ("TARGET_HIT","FAILED","EXPIRED")]

    summary = "<b>📈 PEAD TRACKER — Daily Update</b>\n{'─'*32}\n\n"

    if active_updates:
        summary += "<b>Active positions:</b>\n"
        for u in active_updates:
            arrow = "▲" if u["drift"] >= 0 else "▼"
            emoji = "🟢" if (u["surprise"]=="BEAT" and u["drift"]>0) or \
                           (u["surprise"]=="MISS" and u["drift"]<0) else "🔴"
            summary += (f"{emoji} {u['stock']} ({u['surprise']}) "
                       f"₹{u['current']:,.2f} {arrow} {u['drift']:+.1f}% "
                       f"(Day {u['days']})\n")

    if hit_updates:
        summary += "\n<b>Closed today:</b>\n"
        for u in hit_updates:
            summary += (f"{'✅' if 'TARGET' in u['status'] else '❌'} "
                       f"{u['stock']} — {u['status']} "
                       f"({u['drift']:+.1f}% in {u['days']} days)\n")

    summary = summary.replace("{'─'*32}", "─"*32)
    summary += f"\n⚠️ Educational only.\n🕐 {ts()}"
    send(summary)

    print(f"  PEAD summary sent — {len(active_updates)} active, {len(hit_updates)} closed")


def show_pead_instructions():
    """Print instructions for adding new PEAD positions."""
    print("\n" + "="*55)
    print("HOW TO USE THE PEAD TRACKER")
    print("="*55)
    print("""
When you see a stock beat earnings (revenue UP, PAT UP vs estimates):

Wait 3-5 days after results for the initial spike to settle.
Then add it to the tracker:

  from time_strategies import add_pead_manually
  add_pead_manually(
      stock="Reliance",
      ticker="RELIANCE.NS",
      result_date="2026-04-15",
      surprise_type="BEAT",   # or "MISS" for negative surprise
      entry_price=2850.0      # today's price, 3-5 days after results
  )

The tracker will:
  - Monitor drift daily for up to 60 days
  - Alert when drift reaches +15% (BEAT) or -15% (MISS)
  - Alert if thesis breaks (BEAT stock falling -8%)
  - Send daily summary at 6 PM

Signs of a strong BEAT worth tracking:
  - Revenue > estimates by 5%+
  - PAT (profit) > estimates by 8%+
  - Management raises guidance
  - Stock gaps up 3%+ on results day
  - Volume on results day > 3x average
""")
    print("="*55)


# ══════════════════════════════════════════════════════════════════════════════
# STREAMLIT TAB
# ══════════════════════════════════════════════════════════════════════════════

def render_time_strategies_tab():
    import streamlit as st
    st.subheader("⏰ Time-Based Strategies")
    st.caption("Opening Range Breakout · Expiry Week Bias · Post-Earnings Drift")

    t1, t2, t3 = st.tabs(["ORB", "Expiry Week", "PEAD Tracker"])

    # ── ORB Tab ───────────────────────────────────────────
    with t1:
        st.markdown("**Opening Range Breakout — Today's Data**")
        st.caption("Tracks first 15 min range (9:15–9:30 AM) then alerts on breakout")

        try:
            conn  = sqlite3.connect(DB_PATH)
            today = datetime.now(IST).strftime("%Y-%m-%d")
            orb   = pd.read_sql_query(
                "SELECT * FROM orb_data WHERE date=? ORDER BY stock",
                conn, params=(today,))
            conn.close()

            if not orb.empty:
                def color_orb(row):
                    if row["breakout"] == "BULLISH":
                        return ["background-color:#1a3a2a;color:#2ecc71"] * len(row)
                    elif row["breakout"] == "BEARISH":
                        return ["background-color:#3a1a1a;color:#e74c3c"] * len(row)
                    return [""] * len(row)

                disp = orb[["stock","orb_low","orb_high","orb_size",
                             "breakout","break_price","target_1","stop_loss"]].copy()
                st.dataframe(disp.style.apply(color_orb, axis=1),
                             use_container_width=True, hide_index=True)
            else:
                st.info("No ORB data today. ORB captures at 9:30 AM on weekdays.")
                st.caption("Run time_strategies.py → option 2 to start scheduler")
        except Exception:
            st.info("No ORB data yet.")

        st.divider()
        st.markdown("**How ORB works**")
        st.markdown("""
- At **9:30 AM** — system records the high and low of first 15 minutes
- At **9:45 AM** — checks if price broke above (bullish) or below (bearish) that range
- Alert fires only if volume confirms (>1.2x average)
- Alert fires only if NIFTY direction matches breakout direction
- **Target** = 1x and 2x the ORB range size
- **Stop** = opposite end of the ORB range
        """)

    # ── Expiry Week Tab ───────────────────────────────────
    with t2:
        st.markdown("**Expiry Week Analysis**")

        expiry_date  = get_monthly_expiry_date()
        now_ist      = datetime.now(IST)
        days_to_exp  = (expiry_date.date() - now_ist.date()).days
        in_exp_week  = is_expiry_week()

        c1, c2, c3 = st.columns(3)
        c1.metric("Next expiry",    expiry_date.strftime("%d %b %Y"))
        c2.metric("Days remaining", days_to_exp)
        c3.metric("Expiry week",    "YES ⚠️" if in_exp_week else "No")

        if in_exp_week:
            st.warning("⚠️ This is expiry week — max pain dynamics are active")

        st.divider()
        st.markdown("**Recent expiry bias reports**")
        try:
            conn = sqlite3.connect(DB_PATH)
            eb   = pd.read_sql_query(
                "SELECT date,expiry_date,nifty_close,max_pain,pcr,bias FROM expiry_bias ORDER BY id DESC LIMIT 10",
                conn)
            conn.close()
            if not eb.empty:
                def color_eb(row):
                    if "BULLISH" in str(row["bias"]):
                        return ["background-color:#1a3a2a;color:#2ecc71"] * len(row)
                    elif "BEARISH" in str(row["bias"]):
                        return ["background-color:#3a1a1a;color:#e74c3c"] * len(row)
                    return [""] * len(row)
                st.dataframe(eb.style.apply(color_eb, axis=1),
                             use_container_width=True, hide_index=True)
            else:
                st.info("No expiry bias reports yet. Runs every Monday at 9 AM.")
        except Exception:
            st.info("No expiry data yet.")

        st.divider()
        st.markdown("**How max pain works**")
        st.markdown("""
- **Max pain** = the strike price where the most options expire worthless
- NSE price tends to gravitate toward max pain during expiry week
- If max pain is **above** current price → bullish bias for the week
- If max pain is **below** current price → bearish bias for the week
- Works because option writers (usually institutions) have incentive to pin price at max pain
        """)

    # ── PEAD Tab ──────────────────────────────────────────
    with t3:
        st.markdown("**Post-Earnings Announcement Drift (PEAD)**")
        st.caption("Tracks stocks with earnings surprises for 30–60 day drift")

        # Add new position
        with st.expander("➕ Add new PEAD position"):
            col1, col2 = st.columns(2)
            with col1:
                p_stock    = st.selectbox("Stock", list(PEAD_WATCHLIST.keys()), key="pead_stock")
                p_type     = st.radio("Surprise type", ["BEAT","MISS"], horizontal=True)
                p_date     = st.date_input("Result date", key="pead_date")
            with col2:
                p_price    = st.number_input("Entry price (₹)", min_value=1.0, value=100.0, key="pead_price")
                st.caption("Enter price 3–5 days after results (after initial move settles)")
            if st.button("Add to PEAD Tracker", type="primary"):
                add_pead_stock(p_stock, PEAD_WATCHLIST[p_stock],
                               str(p_date), p_type, p_price)
                st.success(f"Added {p_stock} PEAD tracker ({p_type} @ ₹{p_price:,.2f})")
                st.rerun()

        # Active positions
        st.markdown("**Active PEAD positions**")
        active = load_pead_active()
        if not active.empty:
            def color_pead(row):
                drift = float(row["drift_pct"])
                surprise = str(row["surprise_type"])
                winning = (surprise=="BEAT" and drift>0) or (surprise=="MISS" and drift<0)
                if winning:
                    return ["background-color:#1a3a2a;color:#2ecc71"] * len(row)
                else:
                    return ["background-color:#3a1a1a;color:#e74c3c"] * len(row)
            disp = active[["stock","surprise_type","entry_price",
                           "current_price","drift_pct","days_held","status"]].copy()
            disp.columns = ["Stock","Surprise","Entry ₹","Current ₹","Drift %","Days","Status"]
            st.dataframe(disp.style.apply(color_pead, axis=1),
                         use_container_width=True, hide_index=True)
        else:
            st.info("No active PEAD positions. Add one using the form above.")

        st.divider()
        st.markdown("**Signs of a strong earnings beat worth tracking**")
        st.markdown("""
- Revenue beats estimate by **5%+**
- PAT (net profit) beats estimate by **8%+**
- Management **raises guidance** for next quarter
- Stock gaps up **3%+** on results day with high volume
- Enter **3–5 days after** results (not on the day — too volatile)
- Exit at **+15% drift**, **60 days**, or if stock reverses **-8%** from entry
        """)


# ── ENTRY POINT ───────────────────────────────────────────────────────────────
if __name__ == "__main__":
    now_ist = datetime.now(IST)
    print("="*55)
    print("NSE Time-Based Strategies")
    print(f"Time: {now_ist.strftime('%d %b %Y %H:%M IST')}")
    print("Modules: ORB · Expiry Week · PEAD")
    print("="*55)

    if "YOUR_" in TELEGRAM_TOKEN or "YOUR_" in CHAT_ID or "YOUR_" in GROQ_API_KEY:
        print("\n⚠️  Fill in TELEGRAM_TOKEN, CHAT_ID and GROQ_API_KEY at the top.\n")
        exit(1)

    init_db()

    print("\n1 — Run ORB scan now (test)")
    print("2 — Run expiry week bias now (test)")
    print("3 — Check PEAD positions now")
    print("4 — Show PEAD instructions")
    print("5 — Start full scheduler (auto-runs everything)\n")

    choice = input("Enter 1–5: ").strip()

    if choice == "1":
        run_orb_capture()
        time.sleep(2)
        run_orb_check()

    elif choice == "2":
        run_expiry_week_bias()

    elif choice == "3":
        check_pead_continuation()

    elif choice == "4":
        show_pead_instructions()

    elif choice == "5":
        print("\nFull scheduler started:")
        print("  9:30 AM — ORB capture")
        print("  9:45 AM — ORB breakout check")
        print("  10:00 AM — ORB breakout check (second pass)")
        print("  9:00 AM Monday — Expiry week bias")
        print("  6:00 PM daily — PEAD continuation check")
        print("\nPress Ctrl+C to stop.\n")

        # ORB — weekdays
        for day in ["monday","tuesday","wednesday","thursday","friday"]:
            getattr(schedule.every(), day).at("09:30").do(run_orb_capture)
            getattr(schedule.every(), day).at("09:45").do(run_orb_check)
            getattr(schedule.every(), day).at("10:00").do(run_orb_check)
            getattr(schedule.every(), day).at("18:00").do(check_pead_continuation)

        # Expiry bias — every Monday
        schedule.every().monday.at("09:00").do(run_expiry_week_bias)

        # Run ORB capture immediately if during market hours
        h = now_ist.hour
        if 9 <= h <= 10 and now_ist.weekday() < 5:
            print("Market hours — running ORB capture now...")
            run_orb_capture()
            time.sleep(60)
            run_orb_check()

        while True:
            schedule.run_pending()
            time.sleep(30)
    else:
        run_orb_capture()
