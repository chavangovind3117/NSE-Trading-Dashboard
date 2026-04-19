"""
Global Overnight Signal — Pre-Open NIFTY Predictor
====================================================
Fires at 8:45 AM IST every weekday — 30 minutes before NSE opens.

Reads 5 global markets overnight:
  1. SGX NIFTY futures (direct NIFTY proxy)
  2. S&P 500 / NASDAQ (US market sentiment)
  3. Brent Crude oil (15% of NIFTY weight via energy stocks)
  4. USD/INR (FII flow indicator)
  5. Hang Seng / Nikkei (Asian market sentiment)

AI synthesises all five → sends Telegram with:
  • Gap prediction (points + direction)
  • Confidence % 
  • Sector bias for the day
  • What to watch at 9:15 AM open
  • Suggested intraday bias

Run in a separate terminal:
  python global_signal.py

Sits alongside your other scripts — all share the same DB.
"""

import yfinance as yf
import requests
import schedule
import time
import sqlite3
import pytz
from groq import Groq
from datetime import datetime, timedelta
import pandas as pd
from config import TELEGRAM_TOKEN, CHAT_ID, GROQ_API_KEY, GROQ_MODEL

# ── CONFIG ────────────────────────────────────────────────────────────────────

IST    = pytz.timezone("Asia/Kolkata")
DB_PATH = "nse_trading.db"

# ── Global market tickers ─────────────────────────────────────────────────────
GLOBAL_TICKERS = {
    # Direct NIFTY proxy
    "SGX_NIFTY":   "^NSEI",          # Use NIFTY itself for overnight ref
    # US markets
    "SP500":       "^GSPC",
    "NASDAQ":      "^IXIC",
    "DOW":         "^DJI",
    # Commodities
    "BRENT_CRUDE": "BZ=F",
    "GOLD":        "GC=F",
    # Currency
    "USDINR":      "INR=X",
    # Asian markets
    "HANG_SENG":   "^HSI",
    "NIKKEI":      "^N225",
    "SGX_NIFTY_F": "^NSEBANK",       # BANKNIFTY as sector proxy
    # Volatility
    "VIX":         "^VIX",
    "INDIA_VIX":   "^INBX",
}

# ── DB ────────────────────────────────────────────────────────────────────────
def init_db():
    conn = sqlite3.connect(DB_PATH)
    conn.execute("""CREATE TABLE IF NOT EXISTS global_signals (
        id          INTEGER PRIMARY KEY AUTOINCREMENT,
        date        TEXT,
        timestamp   TEXT,
        sp500_chg   REAL,
        nasdaq_chg  REAL,
        crude_chg   REAL,
        usdinr_chg  REAL,
        hangseng_chg REAL,
        nikkei_chg  REAL,
        vix_level   REAL,
        prediction  TEXT,
        confidence  INTEGER,
        direction   TEXT,
        analysis    TEXT
    )""")
    conn.commit()
    conn.close()

def save_signal(data: dict):
    conn = sqlite3.connect(DB_PATH)
    conn.execute("""INSERT INTO global_signals
        (date,timestamp,sp500_chg,nasdaq_chg,crude_chg,usdinr_chg,
         hangseng_chg,nikkei_chg,vix_level,prediction,confidence,direction,analysis)
        VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?)""",
        (datetime.now(IST).strftime("%Y-%m-%d"),
         datetime.now(IST).strftime("%Y-%m-%d %H:%M:%S"),
         data.get("sp500_chg",0), data.get("nasdaq_chg",0),
         data.get("crude_chg",0), data.get("usdinr_chg",0),
         data.get("hangseng_chg",0), data.get("nikkei_chg",0),
         data.get("vix",0), data.get("prediction",""),
         data.get("confidence",50), data.get("direction","NEUTRAL"),
         data.get("analysis","")))
    conn.commit()
    conn.close()

def load_recent_signals(days=5):
    conn = sqlite3.connect(DB_PATH)
    df = pd.read_sql_query(
        "SELECT * FROM global_signals ORDER BY id DESC LIMIT ?",
        conn, params=(days,))
    conn.close()
    return df

# ── Fetch one ticker — last 2 days to get overnight change ───────────────────
def fetch_change(ticker: str) -> tuple:
    """Returns (latest_price, pct_change, prev_close)"""
    try:
        df = yf.download(ticker, period="5d", interval="1d",
                         progress=False, auto_adjust=True)
        if df.empty or len(df) < 2:
            return None, None, None
        df.columns = [c[0] if isinstance(c, tuple) else c for c in df.columns]
        latest = float(df["Close"].iloc[-1])
        prev   = float(df["Close"].iloc[-2])
        chg    = ((latest - prev) / prev) * 100
        return latest, chg, prev
    except Exception as e:
        print(f"  Error fetching {ticker}: {e}")
        return None, None, None

def fetch_all_globals() -> dict:
    """Fetch all global market data and return structured dict."""
    print("  Fetching global markets...")
    data = {}

    # US Markets
    sp_price, sp_chg, _    = fetch_change("^GSPC")
    nq_price, nq_chg, _    = fetch_change("^IXIC")
    dj_price, dj_chg, _    = fetch_change("^DJI")
    data["sp500"]     = sp_price; data["sp500_chg"]  = sp_chg  or 0
    data["nasdaq"]    = nq_price; data["nasdaq_chg"] = nq_chg  or 0
    data["dow"]       = dj_price; data["dow_chg"]    = dj_chg  or 0

    # Crude oil
    cr_price, cr_chg, _    = fetch_change("BZ=F")
    data["crude"]     = cr_price; data["crude_chg"]  = cr_chg  or 0

    # Gold
    gd_price, gd_chg, _    = fetch_change("GC=F")
    data["gold"]      = gd_price; data["gold_chg"]   = gd_chg  or 0

    # USD/INR
    fx_price, fx_chg, _    = fetch_change("INR=X")
    data["usdinr"]    = fx_price; data["usdinr_chg"] = fx_chg  or 0

    # Asian markets
    hs_price, hs_chg, _    = fetch_change("^HSI")
    nk_price, nk_chg, _    = fetch_change("^N225")
    data["hangseng"]  = hs_price; data["hangseng_chg"] = hs_chg or 0
    data["nikkei"]    = nk_price; data["nikkei_chg"]   = nk_chg or 0

    # VIX (US fear gauge — correlates with India VIX)
    vx_price, vx_chg, _    = fetch_change("^VIX")
    data["vix"]       = vx_price; data["vix_chg"]   = vx_chg  or 0

    # Previous NIFTY close for reference
    nf_price, nf_chg, nf_prev = fetch_change("^NSEI")
    data["nifty_prev"]  = nf_prev or 0
    data["nifty_last"]  = nf_price or 0

    return data

# ── Score each input ──────────────────────────────────────────────────────────
def score_globals(data: dict) -> tuple:
    """
    Returns (score, confidence, direction, bullish_factors, bearish_factors)
    Score: -10 to +10
    """
    score     = 0
    bullish   = []
    bearish   = []

    sp  = data.get("sp500_chg",  0)
    nq  = data.get("nasdaq_chg", 0)
    cr  = data.get("crude_chg",  0)
    fx  = data.get("usdinr_chg", 0)
    hs  = data.get("hangseng_chg",0)
    nk  = data.get("nikkei_chg", 0)
    vix = data.get("vix", 0)
    vix_chg = data.get("vix_chg", 0)

    # ── S&P 500 (weight: 3x — strongest NIFTY correlator) ──
    if sp > 1.0:
        score += 3; bullish.append(f"S&P 500 strong +{sp:.1f}%")
    elif sp > 0.3:
        score += 2; bullish.append(f"S&P 500 positive +{sp:.1f}%")
    elif sp > 0:
        score += 1; bullish.append(f"S&P 500 flat-positive +{sp:.1f}%")
    elif sp < -1.0:
        score -= 3; bearish.append(f"S&P 500 fell sharply {sp:.1f}%")
    elif sp < -0.3:
        score -= 2; bearish.append(f"S&P 500 negative {sp:.1f}%")
    elif sp < 0:
        score -= 1; bearish.append(f"S&P 500 flat-negative {sp:.1f}%")

    # ── NASDAQ (tech-heavy — relevant for IT stocks) ──
    if nq > 1.0:
        score += 1; bullish.append(f"NASDAQ surged +{nq:.1f}% (positive for IT stocks)")
    elif nq < -1.0:
        score -= 1; bearish.append(f"NASDAQ fell {nq:.1f}% (negative for IT stocks)")

    # ── Crude oil (negative for India = positive for NIFTY) ──
    if cr < -1.5:
        score += 2; bullish.append(f"Crude dropped {cr:.1f}% — positive for India (lower import bill)")
    elif cr < -0.5:
        score += 1; bullish.append(f"Crude eased {cr:.1f}%")
    elif cr > 1.5:
        score -= 2; bearish.append(f"Crude surged +{cr:.1f}% — negative for India (inflation pressure)")
    elif cr > 0.5:
        score -= 1; bearish.append(f"Crude up {cr:.1f}%")

    # ── USD/INR (rupee weakening = bearish for NIFTY via FII outflows) ──
    if fx < -0.3:
        score += 2; bullish.append(f"Rupee strengthened {fx:.2f}% — FII flows positive")
    elif fx < 0:
        score += 1; bullish.append(f"Rupee stable-to-strong")
    elif fx > 0.3:
        score -= 2; bearish.append(f"Rupee weakened +{fx:.2f}% — FII outflow risk")
    elif fx > 0:
        score -= 1; bearish.append(f"Rupee slight weakness")

    # ── Asian markets (same session momentum) ──
    asian_avg = (hs + nk) / 2 if (hs and nk) else (hs or nk or 0)
    if asian_avg > 0.8:
        score += 1; bullish.append(f"Asian markets positive (Hang Seng {hs:+.1f}%, Nikkei {nk:+.1f}%)")
    elif asian_avg < -0.8:
        score -= 1; bearish.append(f"Asian markets weak (Hang Seng {hs:+.1f}%, Nikkei {nk:+.1f}%)")

    # ── VIX (US fear gauge — high VIX = bad for risk assets) ──
    if vix:
        if vix < 15:
            score += 1; bullish.append(f"VIX low at {vix:.1f} — low global fear")
        elif vix > 25:
            score -= 2; bearish.append(f"VIX elevated at {vix:.1f} — global risk-off")
        elif vix > 20:
            score -= 1; bearish.append(f"VIX above 20 at {vix:.1f} — some fear")

    if vix_chg and vix_chg > 10:
        score -= 1; bearish.append(f"VIX spiked +{vix_chg:.1f}% overnight — fear rising")
    elif vix_chg and vix_chg < -10:
        score += 1; bullish.append(f"VIX fell {vix_chg:.1f}% — fear subsiding")

    # ── Classify ──
    score = max(-10, min(10, score))
    if score >= 4:
        direction  = "BULLISH"
        confidence = min(95, 50 + score * 5)
    elif score <= -4:
        direction  = "BEARISH"
        confidence = min(95, 50 + abs(score) * 5)
    elif score >= 2:
        direction  = "MILDLY BULLISH"
        confidence = 55
    elif score <= -2:
        direction  = "MILDLY BEARISH"
        confidence = 55
    else:
        direction  = "NEUTRAL"
        confidence = 45

    return score, confidence, direction, bullish, bearish

# ── Gap estimate ──────────────────────────────────────────────────────────────
def estimate_gap(score: int, nifty_prev: float) -> tuple:
    """Estimate NIFTY gap in points based on score and S&P move."""
    if nifty_prev <= 0:
        nifty_prev = 24000  # fallback
    # Rough historical: 1% S&P move ≈ 0.6% NIFTY move next day
    # Score of +5 ≈ 0.5% expected move
    estimated_pct = score * 0.12  # conservative
    gap_points    = nifty_prev * (estimated_pct / 100)
    return round(gap_points), round(estimated_pct, 2)

# ── AI deep analysis ──────────────────────────────────────────────────────────
def ai_overnight_analysis(data: dict, score: int, confidence: int,
                           direction: str, bullish: list, bearish: list) -> str:
    client = Groq(api_key=GROQ_API_KEY)

    bull_text = "\n".join([f"  + {b}" for b in bullish])
    bear_text = "\n".join([f"  - {b}" for b in bearish])
    gap, gap_pct = estimate_gap(score, data.get("nifty_prev", 24000))

    prompt = f"""You are an expert NSE India market analyst writing a pre-open briefing.

Today: {datetime.now(IST).strftime('%A, %d %B %Y')}
Time: 8:45 AM IST — 30 minutes before NSE opens

OVERNIGHT GLOBAL DATA:
S&P 500:    {data.get('sp500_chg',0):+.2f}%  (closed at {data.get('sp500',0):,.0f})
NASDAQ:     {data.get('nasdaq_chg',0):+.2f}%
Crude Oil:  {data.get('crude_chg',0):+.2f}%  (at ${data.get('crude',0):.1f}/bbl)
USD/INR:    {data.get('usdinr_chg',0):+.3f}%  (at ₹{data.get('usdinr',0):.2f})
Hang Seng:  {data.get('hangseng_chg',0):+.2f}%
Nikkei:     {data.get('nikkei_chg',0):+.2f}%
US VIX:     {data.get('vix',0):.1f} ({data.get('vix_chg',0):+.1f}% change)
Gold:       {data.get('gold_chg',0):+.2f}%

PREVIOUS NIFTY CLOSE: ₹{data.get('nifty_prev',0):,.0f}
SIGNAL SCORE: {score:+d}/10
DIRECTION: {direction} (confidence {confidence}%)

BULLISH FACTORS:
{bull_text if bull_text else "  None significant"}

BEARISH FACTORS:
{bear_text if bear_text else "  None significant"}

Write a sharp pre-open briefing in EXACTLY this format:

🎯 OPENING BIAS: {direction}
📍 EXPECTED GAP: {'+' if gap >= 0 else ''}{gap} points ({'+' if gap_pct >= 0 else ''}{gap_pct}%)

📊 KEY LEVELS TO WATCH AT 9:15 AM:
• Support: ₹___ (explain why)
• Resistance: ₹___ (explain why)
• Breakout above ₹___: bullish signal for day
• Break below ₹___: bearish signal for day

🏭 SECTOR BIAS TODAY:
• Buy bias: [sector] — [one reason linked to overnight data]
• Avoid bias: [sector] — [one reason linked to overnight data]

⚡ FIRST 15 MINUTES STRATEGY:
[2 sentences on what to watch at open and when to act]

🎯 INTRADAY BIAS: [LONG / SHORT / WAIT AND WATCH] with one sentence explanation

Keep under 200 words. Be specific with NIFTY price levels."""

    r = client.chat.completions.create(
        model=GROQ_MODEL,
        messages=[
            {"role": "system", "content": "You are a professional NSE India pre-market analyst. Be specific with price levels and actionable."},
            {"role": "user",   "content": prompt}
        ],
        temperature=0.3,
        max_tokens=700,
    )
    return r.choices[0].message.content.strip()

# ── Telegram ──────────────────────────────────────────────────────────────────
def send(text: str):
    url    = f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/sendMessage"
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

# ── Build and send the full signal ───────────────────────────────────────────
def run_global_signal():
    now_ist  = datetime.now(IST)
    time_str = now_ist.strftime("%I:%M %p IST")
    date_str = now_ist.strftime("%A, %d %B %Y")

    # Skip weekends
    if now_ist.weekday() >= 5:
        print(f"[{time_str}] Weekend — skipping global signal")
        return

    print(f"\n{'='*55}")
    print(f"Global Overnight Signal — {date_str}")
    print(f"{'='*55}")

    # ── 1. Fetch all global data ──────────────────────────
    data = fetch_all_globals()

    # ── 2. Score ──────────────────────────────────────────
    score, confidence, direction, bullish, bearish = score_globals(data)
    gap, gap_pct = estimate_gap(score, data.get("nifty_prev", 24000))

    print(f"Score: {score:+d}/10 | Direction: {direction} | Confidence: {confidence}%")
    print(f"Estimated gap: {gap:+d} points ({gap_pct:+.2f}%)")

    # ── 3. AI analysis ────────────────────────────────────
    print("Generating AI analysis...")
    try:
        analysis = ai_overnight_analysis(data, score, confidence,
                                         direction, bullish, bearish)
    except Exception as e:
        analysis = f"AI unavailable: {e}"
        print(f"AI error: {e}")

    # ── 4. Build Telegram message ─────────────────────────
    dir_emoji = {"BULLISH":"🟢","BEARISH":"🔴","MILDLY BULLISH":"🟡",
                 "MILDLY BEARISH":"🟠","NEUTRAL":"⚪"}.get(direction,"⚪")

    # Global snapshot table
    def row(name, val, chg, invert=False):
        if chg is None: return f"{name}: N/A"
        arr = "▲" if chg >= 0 else "▼"
        sentiment = ""
        if invert:  # crude, USD/INR — negative change = good for India
            sentiment = "✓" if chg < -0.3 else ("✗" if chg > 0.3 else "")
        else:
            sentiment = "✓" if chg > 0.3 else ("✗" if chg < -0.3 else "")
        return f"{name:<12} {arr} {chg:+.2f}%  {sentiment}"

    snapshot = (
        f"<code>"
        f"{row('S&P 500',   data.get('sp500'),   data.get('sp500_chg'))}\n"
        f"{row('NASDAQ',    data.get('nasdaq'),  data.get('nasdaq_chg'))}\n"
        f"{row('Crude Oil', data.get('crude'),   data.get('crude_chg'), invert=True)}\n"
        f"{row('USD/INR',   data.get('usdinr'),  data.get('usdinr_chg'), invert=True)}\n"
        f"{row('Hang Seng', data.get('hangseng'),data.get('hangseng_chg'))}\n"
        f"{row('Nikkei',    data.get('nikkei'),  data.get('nikkei_chg'))}\n"
        f"{row('US VIX',    data.get('vix'),     data.get('vix_chg'), invert=True)}\n"
        f"{row('Gold',      data.get('gold'),    data.get('gold_chg'))}"
        f"</code>"
    )

    bull_lines = "\n".join([f"  ✓ {b}" for b in bullish]) if bullish else "  None"
    bear_lines = "\n".join([f"  ✗ {b}" for b in bearish]) if bearish else "  None"

    msg1 = (
        f"{dir_emoji} <b>GLOBAL OVERNIGHT SIGNAL</b>\n"
        f"{'─'*32}\n"
        f"📅 {date_str} | {time_str}\n"
        f"🎯 Direction: <b>{direction}</b>\n"
        f"📊 Confidence: <b>{confidence}%</b>\n"
        f"📍 Est. NIFTY gap: <b>{'+' if gap >= 0 else ''}{gap} pts ({gap_pct:+.2f}%)</b>\n"
        f"{'─'*32}\n\n"
        f"<b>Global Markets:</b>\n{snapshot}\n\n"
        f"<b>Bullish factors:</b>\n{bull_lines}\n\n"
        f"<b>Bearish factors:</b>\n{bear_lines}"
    )

    msg2 = f"<b>🤖 AI Pre-Open Analysis:</b>\n\n{analysis}"

    msg3 = (
        f"{'─'*32}\n"
        f"⏰ NSE opens in 30 minutes (9:15 AM)\n"
        f"📌 Morning briefing follows at 9:00 AM\n"
        f"⚠️ Educational only. Not financial advice."
    )

    # ── 5. Send ───────────────────────────────────────────
    send(msg1)
    time.sleep(1)
    send(msg2)
    time.sleep(1)
    send(msg3)

    # ── 6. Save to DB ─────────────────────────────────────
    data["prediction"] = f"{gap:+d} points"
    data["confidence"] = confidence
    data["direction"]  = direction
    data["analysis"]   = analysis
    save_signal(data)

    print(f"✓ Global signal sent at {time_str}")
    print(f"{'='*55}\n")

# ── Streamlit helper — call from app.py ──────────────────────────────────────
def render_global_signal_tab():
    """Add this to app.py for a Global Signal tab in the dashboard."""
    import streamlit as st

    st.subheader("🌍 Global Overnight Signal")
    st.caption("Pre-open NIFTY predictor based on overnight global markets")

    col1, col2 = st.columns([2, 1])

    with col1:
        if st.button("Fetch Global Signal Now", type="primary"):
            with st.spinner("Reading global markets..."):
                data = fetch_all_globals()
                score, conf, direction, bullish, bearish = score_globals(data)
                gap, gap_pct = estimate_gap(score, data.get("nifty_prev", 24000))

                dir_color = {"BULLISH":"🟢","BEARISH":"🔴",
                             "MILDLY BULLISH":"🟡","MILDLY BEARISH":"🟠",
                             "NEUTRAL":"⚪"}.get(direction,"⚪")

                st.markdown(f"### {dir_color} {direction}")
                st.markdown(f"**Confidence:** {conf}% | **Est. gap:** {gap:+d} pts ({gap_pct:+.2f}%)")
                st.progress(conf / 100)

                m1, m2, m3 = st.columns(3)
                m1.metric("S&P 500",   f"{data.get('sp500_chg',0):+.2f}%")
                m2.metric("Crude Oil", f"{data.get('crude_chg',0):+.2f}%")
                m3.metric("USD/INR",   f"{data.get('usdinr_chg',0):+.3f}%")

                m4, m5, m6 = st.columns(3)
                m4.metric("NASDAQ",    f"{data.get('nasdaq_chg',0):+.2f}%")
                m5.metric("Hang Seng", f"{data.get('hangseng_chg',0):+.2f}%")
                m6.metric("VIX",       f"{data.get('vix',0):.1f}")

                if bullish:
                    st.success("**Bullish factors:** " + " · ".join(bullish))
                if bearish:
                    st.error("**Bearish factors:** " + " · ".join(bearish))

                # AI analysis
                try:
                    analysis = ai_overnight_analysis(data, score, conf,
                                                     direction, bullish, bearish)
                    st.markdown(
                        f'<div style="background:#1a1f2e;border-left:4px solid #6c63ff;'
                        f'border-radius:8px;padding:16px;margin:10px 0;font-size:14px;'
                        f'line-height:1.8;color:#e0e0e0">'
                        f'{analysis.replace(chr(10),"<br>")}</div>',
                        unsafe_allow_html=True)
                except Exception as e:
                    st.error(f"AI error: {e}")

    with col2:
        st.markdown("**Recent signals**")
        try:
            hist = load_recent_signals(7)
            if not hist.empty:
                for _, row in hist.iterrows():
                    icon = "🟢" if "BULL" in str(row["direction"]) else \
                           "🔴" if "BEAR" in str(row["direction"]) else "⚪"
                    st.markdown(
                        f"{icon} **{row['date']}** — {row['direction']} "
                        f"({row['confidence']}%)")
            else:
                st.caption("No history yet.")
        except Exception:
            st.caption("No history yet.")

# ── Main ──────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("=" * 55)
    print("NSE Global Overnight Signal Service")
    print("Fires: 8:45 AM IST every weekday")
    print("=" * 55)

    if "YOUR_" in TELEGRAM_TOKEN or "YOUR_" in CHAT_ID or "YOUR_" in GROQ_API_KEY:
        print("\n⚠️  Fill in TELEGRAM_TOKEN, CHAT_ID and GROQ_API_KEY at the top.\n")
        exit(1)

    init_db()

    # Schedule at 8:45 AM IST every weekday
    schedule.every().monday.at("08:45").do(run_global_signal)
    schedule.every().tuesday.at("08:45").do(run_global_signal)
    schedule.every().wednesday.at("08:45").do(run_global_signal)
    schedule.every().thursday.at("08:45").do(run_global_signal)
    schedule.every().friday.at("08:45").do(run_global_signal)

    print("\nPress ENTER to run a test signal RIGHT NOW")
    print("Or wait — fires automatically at 8:45 AM IST every weekday\n")

    import threading
    def wait_enter():
        input()
        print("Running global signal now...\n")
        run_global_signal()

    t = threading.Thread(target=wait_enter, daemon=True)
    t.start()

    while True:
        schedule.run_pending()
        time.sleep(30)
