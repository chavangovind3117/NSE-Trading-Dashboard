"""
NSE Multi-Timeframe Confluence Checker
=======================================
Checks Daily + Weekly + Monthly signals for every stock.
Only alerts when ALL THREE timeframes agree on direction.

This means:
  STRONG BUY  = Daily BUY + Weekly BUY + Monthly BUY
  STRONG SELL = Daily SELL + Weekly SELL + Monthly SELL
  NO ALERT    = any disagreement between timeframes

These are the highest quality signals your system produces.
Expect 1-3 alerts per week — not every day.

Run this in a separate terminal:
  python confluence_checker.py

It scans every 4 hours during market hours and sends
Telegram alerts only on full confluence.
"""

import yfinance as yf
import pandas as pd
import requests
import schedule
import time
import sqlite3
from groq import Groq
from datetime import datetime
import pytz
from config import TELEGRAM_TOKEN, CHAT_ID, GROQ_API_KEY, GROQ_MODEL

# ── CONFIG ────────────────────────────────────────────────────────────────────
IST    = pytz.timezone("Asia/Kolkata")
DB_PATH = "nse_trading.db"

# ── Watchlist ─────────────────────────────────────────────────────────────────
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
    "ONGC":          "ONGC.NS",
    "Zomato":        "ZOMATO.NS",
    "IRCTC":         "IRCTC.NS",
    "Dixon Tech":    "DIXON.NS",
}

# Timeframe configs
TIMEFRAMES = {
    "Daily":   {"period": "6mo",  "interval": "1d"},
    "Weekly":  {"period": "3y",   "interval": "1wk"},
    "Monthly": {"period": "10y",  "interval": "1mo"},
}

# ── Database ──────────────────────────────────────────────────────────────────
def init_db():
    conn = sqlite3.connect(DB_PATH)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS confluence_alerts (
            id          INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp   TEXT,
            stock       TEXT,
            ticker      TEXT,
            direction   TEXT,
            daily_sig   TEXT,
            weekly_sig  TEXT,
            monthly_sig TEXT,
            price       REAL,
            confidence  INTEGER,
            analysis    TEXT
        )
    """)
    conn.commit()
    conn.close()


def save_confluence(stock, ticker, direction, d_sig, w_sig, m_sig, price, confidence, analysis):
    conn = sqlite3.connect(DB_PATH)
    conn.execute("""
        INSERT INTO confluence_alerts
        (timestamp,stock,ticker,direction,daily_sig,weekly_sig,monthly_sig,price,confidence,analysis)
        VALUES (?,?,?,?,?,?,?,?,?,?)
    """, (datetime.now(IST).strftime("%Y-%m-%d %H:%M:%S"),
          stock, ticker, direction, d_sig, w_sig, m_sig, price, confidence, analysis))
    conn.commit()
    conn.close()


def already_alerted_today(stock, direction):
    """Avoid duplicate alerts for same stock+direction on same day."""
    conn = sqlite3.connect(DB_PATH)
    today = datetime.now(IST).strftime("%Y-%m-%d")
    row = conn.execute("""
        SELECT id FROM confluence_alerts
        WHERE stock=? AND direction=? AND timestamp LIKE ?
        ORDER BY id DESC LIMIT 1
    """, (stock, direction, f"{today}%")).fetchone()
    conn.close()
    return row is not None


# ── Indicators ────────────────────────────────────────────────────────────────
def compute_indicators(df: pd.DataFrame) -> pd.DataFrame:
    if len(df) < 20:
        return df
    df = df.copy()

    # EMAs
    df["EMA20"]  = df["Close"].ewm(span=20,  adjust=False).mean()
    df["EMA50"]  = df["Close"].ewm(span=50,  adjust=False).mean()
    df["EMA200"] = df["Close"].ewm(span=200, adjust=False).mean()

    # RSI
    delta = df["Close"].diff()
    gain  = delta.where(delta > 0, 0).rolling(14).mean()
    loss  = (-delta.where(delta < 0, 0)).rolling(14).mean()
    df["RSI"] = 100 - (100 / (1 + gain / loss))

    # MACD
    fast          = df["Close"].ewm(span=12, adjust=False).mean()
    slow          = df["Close"].ewm(span=26, adjust=False).mean()
    df["MACD"]        = fast - slow
    df["MACD_Signal"] = df["MACD"].ewm(span=9, adjust=False).mean()
    df["MACD_Hist"]   = df["MACD"] - df["MACD_Signal"]

    # ATR
    tr = pd.concat([
        df["High"] - df["Low"],
        (df["High"] - df["Close"].shift()).abs(),
        (df["Low"]  - df["Close"].shift()).abs()
    ], axis=1).max(axis=1)
    df["ATR"] = tr.rolling(14).mean()

    # Bollinger Bands
    sma           = df["Close"].rolling(20).mean()
    std           = df["Close"].rolling(20).std()
    df["BB_Upper"] = sma + 2 * std
    df["BB_Lower"] = sma - 2 * std
    df["BB_Mid"]   = sma

    return df


def get_stats(df: pd.DataFrame) -> dict:
    if df.empty or len(df) < 2:
        return {}
    c = df["Close"]
    lat, prev = float(c.iloc[-1]), float(c.iloc[-2])
    chg = lat - prev

    def gv(col):
        return float(df[col].iloc[-1]) if col in df.columns and pd.notna(df[col].iloc[-1]) else None

    return {
        "latest":    lat,
        "prev":      prev,
        "chg":       chg,
        "chg_pct":   (chg / prev) * 100,
        "high52":    float(c.tail(52).max()),
        "low52":     float(c.tail(52).min()),
        "rsi":       gv("RSI"),
        "ema20":     gv("EMA20"),
        "ema50":     gv("EMA50"),
        "ema200":    gv("EMA200"),
        "atr":       gv("ATR"),
        "macd":      gv("MACD"),
        "macd_sig":  gv("MACD_Signal"),
        "macd_hist": gv("MACD_Hist"),
        "bb_upper":  gv("BB_Upper"),
        "bb_lower":  gv("BB_Lower"),
        "vol_cur":   float(df["Volume"].iloc[-1]) if "Volume" in df.columns else None,
        "vol_avg":   float(df["Volume"].tail(20).mean()) if "Volume" in df.columns else None,
    }


# ── Single timeframe signal ───────────────────────────────────────────────────
def timeframe_signal(s: dict) -> tuple:
    """
    Returns (signal, score, reasons)
    signal: 'BUY' | 'SELL' | 'NEUTRAL'
    score:  -5 to +5
    """
    if not s:
        return "NEUTRAL", 0, []

    score   = 0
    reasons = []
    ltp     = s.get("latest", 0)
    e20, e50, e200 = s.get("ema20"), s.get("ema50"), s.get("ema200")
    rsi     = s.get("rsi")
    mh      = s.get("macd_hist")
    vc, va  = s.get("vol_cur"), s.get("vol_avg")

    # ── EMA alignment (most important — weighted 2x) ──
    if e20 and e50 and e200:
        if ltp > e20 > e50 > e200:
            score += 2; reasons.append("Full bullish EMA stack (price>EMA20>50>200)")
        elif ltp > e20 and e20 > e50:
            score += 1; reasons.append("Bullish EMA20/50 alignment")
        elif ltp < e20 < e50 < e200:
            score -= 2; reasons.append("Full bearish EMA stack (price<EMA20<50<200)")
        elif ltp < e20 and e20 < e50:
            score -= 1; reasons.append("Bearish EMA20/50 alignment")

    # ── RSI ──
    if rsi:
        if rsi < 30:
            score += 2; reasons.append(f"RSI deeply oversold ({rsi:.0f})")
        elif rsi < 45:
            score += 1; reasons.append(f"RSI in buy zone ({rsi:.0f})")
        elif rsi > 70:
            score -= 2; reasons.append(f"RSI overbought ({rsi:.0f})")
        elif rsi > 60:
            score -= 1; reasons.append(f"RSI elevated ({rsi:.0f})")

    # ── MACD histogram ──
    if mh is not None:
        if mh > 0:
            score += 1; reasons.append("MACD histogram positive")
        else:
            score -= 1; reasons.append("MACD histogram negative")

    # ── Price vs 52W levels ──
    h52, l52 = s.get("high52"), s.get("low52")
    if h52 and ltp >= h52 * 0.95:
        score += 1; reasons.append("Near 52-period high (strength)")
    if l52 and ltp <= l52 * 1.05:
        score -= 1; reasons.append("Near 52-period low (weakness)")

    # ── Volume confirmation ──
    if vc and va and va > 0:
        ratio = vc / va
        if ratio > 1.5:
            score += 1; reasons.append(f"Volume {ratio:.1f}x above average")
        elif ratio < 0.5:
            score -= 1; reasons.append("Below average volume")

    # ── Classify ──
    if score >= 3:
        return "BUY",  score, reasons
    elif score <= -3:
        return "SELL", score, reasons
    else:
        return "NEUTRAL", score, reasons


# ── Fetch one timeframe ───────────────────────────────────────────────────────
def fetch_timeframe(ticker: str, period: str, interval: str) -> pd.DataFrame:
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


# ── Full confluence check for one stock ──────────────────────────────────────
def check_confluence(stock_name: str, ticker: str) -> dict | None:
    """
    Returns confluence result dict if all 3 TFs agree, else None.
    """
    results = {}

    for tf_name, cfg in TIMEFRAMES.items():
        df = fetch_timeframe(ticker, cfg["period"], cfg["interval"])
        if df.empty or len(df) < 20:
            return None  # can't assess without data
        df = compute_indicators(df)
        s  = get_stats(df)
        signal, score, reasons = timeframe_signal(s)
        results[tf_name] = {
            "signal":  signal,
            "score":   score,
            "reasons": reasons,
            "stats":   s,
        }

    d = results["Daily"]["signal"]
    w = results["Weekly"]["signal"]
    m = results["Monthly"]["signal"]

    # All three must agree AND none can be NEUTRAL
    if d == w == m and d != "NEUTRAL":
        # Calculate confidence 0-100
        total_score = abs(results["Daily"]["score"]) + \
                      abs(results["Weekly"]["score"]) + \
                      abs(results["Monthly"]["score"])
        confidence = min(100, int((total_score / 15) * 100))

        return {
            "stock":       stock_name,
            "ticker":      ticker,
            "direction":   d,
            "confidence":  confidence,
            "daily":       results["Daily"],
            "weekly":      results["Weekly"],
            "monthly":     results["Monthly"],
            "price":       results["Daily"]["stats"].get("latest", 0),
        }
    return None


# ── AI deep analysis for confirmed confluence ─────────────────────────────────
def ai_confluence_analysis(result: dict) -> str:
    client = Groq(api_key=GROQ_API_KEY)

    d, w, m = result["daily"], result["weekly"], result["monthly"]
    direction = result["direction"]
    stock     = result["stock"]
    price     = result["price"]

    def fmt_reasons(r): return "\n".join([f"  • {x}" for x in r])

    prompt = f"""You are an expert NSE India technical analyst.

A RARE multi-timeframe confluence has been detected:

Stock: {stock} ({result['ticker']})
Current Price: ₹{price:,.2f}
Direction: {direction} (confirmed on ALL 3 timeframes)
Confidence: {result['confidence']}%

DAILY signals (score {d['score']:+d}):
{fmt_reasons(d['reasons'])}

WEEKLY signals (score {w['score']:+d}):
{fmt_reasons(w['reasons'])}

MONTHLY signals (score {m['score']:+d}):
{fmt_reasons(m['reasons'])}

This is a high-conviction setup. Write a specific trading plan:

1. WHY THIS MATTERS: One sentence on why all-timeframe confluence is significant
2. ENTRY IDEA: Specific entry strategy (e.g. buy now / wait for pullback to ₹___)
3. STOP LOSS: Where to put stop-loss and why (use ATR or key level)
4. TARGETS: Two price targets with reasoning
5. TIMEFRAME: How long to hold this trade (days/weeks/months)
6. ONE RISK: The single biggest thing that could invalidate this setup

Be very specific with price levels. Under 200 words."""

    r = client.chat.completions.create(
        model=GROQ_MODEL,
        messages=[
            {"role": "system", "content": "You are a professional NSE India trader. Give specific, actionable trade plans."},
            {"role": "user",   "content": prompt}
        ],
        temperature=0.3,
        max_tokens=500,
    )
    return r.choices[0].message.content.strip()


# ── Telegram ──────────────────────────────────────────────────────────────────
def send_telegram(text: str):
    url = f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/sendMessage"
    chunks = [text[i:i+4000] for i in range(0, len(text), 4000)]
    for chunk in chunks:
        try:
            requests.post(url, json={
                "chat_id":    CHAT_ID,
                "text":       chunk,
                "parse_mode": "HTML"
            }, timeout=15)
        except Exception as e:
            print(f"Telegram error: {e}")
        time.sleep(0.5)


def send_confluence_alert(result: dict, ai_analysis: str):
    direction  = result["direction"]
    stock      = result["stock"]
    ticker     = result["ticker"]
    price      = result["price"]
    confidence = result["confidence"]

    emoji = "🟢" if direction == "BUY" else "🔴"
    label = "STRONG BUY" if direction == "BUY" else "STRONG SELL"

    d_r = "\n".join([f"  • {x}" for x in result["daily"]["reasons"][:3]])
    w_r = "\n".join([f"  • {x}" for x in result["weekly"]["reasons"][:3]])
    m_r = "\n".join([f"  • {x}" for x in result["monthly"]["reasons"][:3]])

    atr = result["daily"]["stats"].get("atr")
    atr_str = f"₹{atr:.2f}" if atr else "N/A"

    msg = (
        f"{emoji}{emoji} <b>CONFLUENCE ALERT — {label}</b> {emoji}{emoji}\n"
        f"{'─'*35}\n"
        f"<b>Stock:</b>      {stock} ({ticker})\n"
        f"<b>Price:</b>      ₹{price:,.2f}\n"
        f"<b>Confidence:</b> {confidence}%\n"
        f"<b>Signal:</b>     D={direction} | W={direction} | M={direction}\n"
        f"<b>ATR (daily):</b> {atr_str}\n"
        f"{'─'*35}\n\n"
        f"<b>📅 Daily reasons:</b>\n{d_r}\n\n"
        f"<b>📆 Weekly reasons:</b>\n{w_r}\n\n"
        f"<b>📅 Monthly reasons:</b>\n{m_r}\n\n"
        f"{'─'*35}\n"
        f"<b>🤖 AI Trade Plan:</b>\n\n{ai_analysis}\n\n"
        f"{'─'*35}\n"
        f"⚠️ Educational only. Not financial advice.\n"
        f"🕐 {datetime.now(IST).strftime('%d %b %Y %H:%M IST')}"
    )
    send_telegram(msg)


# ── Main scan ─────────────────────────────────────────────────────────────────
def run_confluence_scan():
    now_ist  = datetime.now(IST)
    time_str = now_ist.strftime("%H:%M:%S")

    # Only during market hours Mon-Fri
    if now_ist.weekday() >= 5:
        print(f"[{time_str}] Weekend — skipping")
        return

    hour, minute = now_ist.hour, now_ist.minute
    in_market = (hour > 9 or (hour == 9 and minute >= 15)) and \
                (hour < 15 or (hour == 15 and minute <= 30))

    if not in_market:
        print(f"[{time_str}] Outside market hours — skipping")
        return

    print(f"\n[{time_str}] Starting confluence scan ({len(WATCHLIST)} stocks)...")
    found = 0

    for stock_name, ticker in WATCHLIST.items():
        print(f"  Checking {stock_name}...", end=" ")
        try:
            result = check_confluence(stock_name, ticker)

            if result:
                direction = result["direction"]

                # Skip if already alerted today
                if already_alerted_today(stock_name, direction):
                    print(f"CONFLUENCE {direction} (already alerted today)")
                    continue

                print(f"✅ CONFLUENCE {direction} — confidence {result['confidence']}%")

                # Get AI analysis
                try:
                    ai_analysis = ai_confluence_analysis(result)
                except Exception as e:
                    ai_analysis = f"AI analysis unavailable: {e}"

                # Send alert
                send_confluence_alert(result, ai_analysis)

                # Save to DB
                d = result["daily"]; w = result["weekly"]; m = result["monthly"]
                save_confluence(
                    stock_name, ticker, direction,
                    d["signal"], w["signal"], m["signal"],
                    result["price"], result["confidence"], ai_analysis
                )

                found += 1
                time.sleep(2)  # brief pause between alerts

            else:
                print("no confluence")

        except Exception as e:
            print(f"ERROR: {e}")

    if found == 0:
        print(f"[{time_str}] Scan complete — no confluence found across {len(WATCHLIST)} stocks")
    else:
        print(f"[{time_str}] Scan complete — {found} confluence alert(s) sent")


# ── Streamlit tab helper (import this in app.py) ──────────────────────────────
def get_confluence_for_stock(ticker: str) -> dict | None:
    """
    Call this from app.py to show confluence in the dashboard.
    Returns result dict or None.
    """
    return check_confluence(ticker, ticker)


def render_confluence_tab(stock_name: str, ticker: str):
    """
    Drop this into app.py's tab section.
    Renders a full confluence view for the currently selected stock.
    """
    import streamlit as st

    st.subheader(f"⚡ Multi-Timeframe Confluence — {stock_name}")
    st.caption("Checks Daily + Weekly + Monthly. High-conviction signals only.")

    if st.button("Check Confluence Now", type="primary", key="conf_btn"):
        cols = st.columns(3)
        tf_results = {}

        for i, (tf_name, cfg) in enumerate(TIMEFRAMES.items()):
            with cols[i]:
                with st.spinner(f"Checking {tf_name}..."):
                    df = fetch_timeframe(ticker, cfg["period"], cfg["interval"])
                    if df.empty or len(df) < 20:
                        st.warning(f"{tf_name}: No data")
                        continue
                    df = compute_indicators(df)
                    s  = get_stats(df)
                    signal, score, reasons = timeframe_signal(s)
                    tf_results[tf_name] = {"signal": signal, "score": score, "reasons": reasons, "stats": s}

                    # Display
                    color = {"BUY": "🟢", "SELL": "🔴", "NEUTRAL": "🟡"}.get(signal, "⚪")
                    st.markdown(f"### {color} {tf_name}")
                    st.markdown(f"**Signal:** {signal} (score {score:+d})")
                    rsi = s.get("rsi")
                    st.metric("RSI",    f"{rsi:.1f}" if rsi else "—")
                    st.metric("Price",  f"₹{s.get('latest',0):,.2f}")
                    st.metric("Change", f"{s.get('chg_pct',0):+.2f}%")
                    st.markdown("**Reasons:**")
                    for r in reasons[:4]:
                        st.markdown(f"• {r}")

        st.divider()

        if len(tf_results) == 3:
            signals = [tf_results[tf]["signal"] for tf in ["Daily","Weekly","Monthly"]]
            d, w, m = signals[0], signals[1], signals[2]

            if d == w == m and d != "NEUTRAL":
                total = sum(abs(tf_results[tf]["score"]) for tf in tf_results)
                confidence = min(100, int((total / 15) * 100))

                if d == "BUY":
                    st.success(f"✅ FULL CONFLUENCE — STRONG BUY | Confidence: {confidence}%")
                else:
                    st.error(f"❌ FULL CONFLUENCE — STRONG SELL | Confidence: {confidence}%")

                st.markdown(f"**All three timeframes agree: {d}**")
                st.progress(confidence / 100)

                atr = tf_results["Daily"]["stats"].get("atr")
                price = tf_results["Daily"]["stats"].get("latest", 0)
                if atr:
                    if d == "BUY":
                        st.info(
                            f"📐 **Suggested trade plan:**\n\n"
                            f"Entry: ₹{price:,.2f} (current price)\n"
                            f"Stop-loss: ₹{price - 1.5*atr:,.2f} (1.5x ATR below)\n"
                            f"Target 1: ₹{price + 1.5*atr:,.2f} (1R)\n"
                            f"Target 2: ₹{price + 3.0*atr:,.2f} (2R)"
                        )
                    else:
                        st.info(
                            f"📐 **Suggested trade plan:**\n\n"
                            f"Entry: ₹{price:,.2f} (current price)\n"
                            f"Stop-loss: ₹{price + 1.5*atr:,.2f} (1.5x ATR above)\n"
                            f"Target 1: ₹{price - 1.5*atr:,.2f} (1R)\n"
                            f"Target 2: ₹{price - 3.0*atr:,.2f} (2R)"
                        )

            elif len(set(signals)) == 1 and signals[0] == "NEUTRAL":
                st.warning("⚪ All timeframes neutral — no clear direction. Wait for a setup to form.")
            else:
                # Show partial confluence
                st.warning(
                    f"⚡ Partial confluence only\n\n"
                    f"Daily: **{d}** | Weekly: **{w}** | Monthly: **{m}**\n\n"
                    f"Timeframes disagree — this is a lower quality signal. "
                    f"Wait until at least 2 of 3 agree before considering a trade."
                )

                # Show which TFs agree
                if d == w: st.info("Daily + Weekly agree — watch monthly for confirmation")
                elif w == m: st.info("Weekly + Monthly agree — wait for daily to confirm")
                elif d == m: st.info("Daily + Monthly agree — wait for weekly to confirm")

        # History of past confluence alerts for this stock
        st.divider()
        st.subheader("Past confluence alerts for this stock")
        try:
            conn = sqlite3.connect(DB_PATH)
            hist = pd.read_sql_query(
                "SELECT timestamp, direction, confidence, daily_sig, weekly_sig, monthly_sig, price, analysis "
                "FROM confluence_alerts WHERE stock=? ORDER BY id DESC LIMIT 10",
                conn, params=(stock_name,)
            )
            conn.close()
            if not hist.empty:
                for _, row in hist.iterrows():
                    icon = "🟢" if row["direction"] == "BUY" else "🔴"
                    with st.expander(f"{icon} {row['direction']} — ₹{row['price']:,.2f} — {row['timestamp']}"):
                        st.markdown(f"**Confidence:** {row['confidence']}%")
                        st.markdown(f"D: {row['daily_sig']} | W: {row['weekly_sig']} | M: {row['monthly_sig']}")
                        st.markdown(row["analysis"])
            else:
                st.caption("No past confluence alerts for this stock yet.")
        except Exception:
            st.caption("No history yet.")


# ── Entry point ───────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("=" * 55)
    print("NSE Multi-Timeframe Confluence Checker")
    print(f"Stocks:   {len(WATCHLIST)}")
    print("Schedule: Every 4 hours during market hours")
    print("Alerts:   Only when Daily + Weekly + Monthly all agree")
    print("=" * 55)

    if "YOUR_" in TELEGRAM_TOKEN or "YOUR_" in CHAT_ID or "YOUR_" in GROQ_API_KEY:
        print("\n⚠️  Fill in TELEGRAM_TOKEN, CHAT_ID and GROQ_API_KEY at the top.\n")
        exit(1)

    init_db()

    # Schedule every 4 hours
    schedule.every(4).hours.do(run_confluence_scan)

    # Run immediately on start
    print("\nPress ENTER to run a test scan now")
    print("Or wait — auto-runs every 4 hours during market hours\n")

    import threading

    def wait_enter():
        input()
        print("Running confluence scan now...\n")
        run_confluence_scan()

    t = threading.Thread(target=wait_enter, daemon=True)
    t.start()

    while True:
        schedule.run_pending()
        time.sleep(60)
