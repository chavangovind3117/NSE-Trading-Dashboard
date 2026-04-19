"""
Sentiment Tracker — sentiment_tracker.py
=======================================
Market sentiment dashboard for NSE India.

Features:
- Put-Call Ratio extremes
- India VIX spike / crush
- Market breadth advance / decline ratio
- Contrarian alert signals

Run standalone: python sentiment_tracker.py
"""

import requests
import sqlite3
import time
import pytz
import pandas as pd
import numpy as np
import yfinance as yf
from groq import Groq
from datetime import datetime, timedelta
from config import TELEGRAM_TOKEN, CHAT_ID, GROQ_API_KEY, GROQ_MODEL

IST = pytz.timezone("Asia/Kolkata")
DB_PATH = "nse_trading.db"

SENTIMENT_WATCHLIST = {
    "Reliance":      "RELIANCE.NS",
    "HDFC Bank":     "HDFCBANK.NS",
    "ICICI Bank":    "ICICIBANK.NS",
    "TCS":           "TCS.NS",
    "Infosys":       "INFY.NS",
    "SBI":           "SBIN.NS",
    "Bajaj Finance": "BAJFINANCE.NS",
    "Kotak Bank":    "KOTAKBANK.NS",
    "Axis Bank":     "AXISBANK.NS",
    "Titan":         "TITAN.NS",
    "Asian Paints":  "ASIANPAINT.NS",
    "Maruti":        "MARUTI.NS",
    "L&T":           "LT.NS",
}

CHROME_HEADERS = {
    "User-Agent":      "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36",
    "Accept":          "application/json, text/plain, */*",
    "Accept-Language": "en-US,en;q=0.9",
    "Accept-Encoding": "gzip, deflate, br",
    "Referer":         "https://www.nseindia.com/",
    "Origin":          "https://www.nseindia.com",
    "sec-fetch-dest":  "empty",
    "sec-fetch-mode":  "cors",
    "sec-fetch-site":  "same-origin",
}


# ── DB SETUP ──────────────────────────────────────────────────────────────────

def init_db():
    conn = sqlite3.connect(DB_PATH)
    conn.execute("""CREATE TABLE IF NOT EXISTS sentiment_history (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        timestamp TEXT,
        pcr REAL,
        vix REAL,
        vix_sma20 REAL,
        vix_delta_pct REAL,
        advancers INTEGER,
        decliners INTEGER,
        breadth_ratio REAL,
        signal TEXT,
        note TEXT
    )""")
    conn.commit()
    conn.close()


def save_sentiment(pcr, vix, vix_sma20, vix_delta_pct, advancers,
                   decliners, breadth_ratio, signal, note):
    conn = sqlite3.connect(DB_PATH)
    conn.execute("""INSERT INTO sentiment_history (
        timestamp, pcr, vix, vix_sma20, vix_delta_pct,
        advancers, decliners, breadth_ratio, signal, note)
        VALUES (?,?,?,?,?,?,?,?,?,?)""",
        (datetime.now(IST).strftime("%Y-%m-%d %H:%M:%S"), pcr, vix,
         vix_sma20, vix_delta_pct, advancers, decliners,
         breadth_ratio, signal, note))
    conn.commit()
    conn.close()


def load_sentiment_history(limit=20):
    conn = sqlite3.connect(DB_PATH)
    df = pd.read_sql_query(
        "SELECT * FROM sentiment_history ORDER BY id DESC LIMIT ?", conn,
        params=(limit,))
    conn.close()
    return df


# ── NSE HELPERS ─────────────────────────────────────────────────────────────

def get_nse_session() -> requests.Session:
    s = requests.Session()
    s.headers.update(CHROME_HEADERS)
    try:
        s.get("https://www.nseindia.com/", timeout=15)
        time.sleep(1.5)
        s.get("https://www.nseindia.com/market-data/live-equity-market", timeout=10)
        time.sleep(1.0)
    except Exception:
        pass
    return s


def nse_get(url: str, retries=3):
    for attempt in range(retries):
        try:
            s = get_nse_session()
            r = s.get(url, timeout=20)
            if r.status_code == 200:
                return r.json()
            time.sleep(2)
        except Exception:
            time.sleep(2)
    return None


def send_telegram(text: str):
    if not TELEGRAM_TOKEN or not CHAT_ID:
        return
    url = f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/sendMessage"
    payload = {
        "chat_id": CHAT_ID,
        "text": text,
        "parse_mode": "HTML"
    }
    try:
        requests.post(url, json=payload, timeout=15)
    except Exception:
        pass


def ai(prompt: str, max_tokens=400) -> str:
    if not GROQ_API_KEY or not GROQ_MODEL:
        return "AI unavailable: missing GROQ API config."
    try:
        r = Groq(api_key=GROQ_API_KEY).chat.completions.create(
            model=GROQ_MODEL,
            messages=[
                {"role": "system", "content": "You are an expert NSE India market analyst. Provide concise, actionable sentiment insights."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.3, max_tokens=max_tokens)
        return r.choices[0].message.content.strip()
    except Exception as e:
        return f"AI unavailable: {e}"


def sentiment_ai_summary(pcr, vix, vix_delta_pct, breadth_ratio, advancers, decliners, signal, note):
    prompt = (
        "Summarize the current NSE market sentiment in 3 short bullet points. "
        f"PCR is {pcr if pcr is not None else 'unavailable'}, India VIX is {vix if vix is not None else 'unavailable'}, "
        f"VIX delta vs 20d is {f'{vix_delta_pct:+.2f}%' if vix_delta_pct is not None else 'unavailable'}, breadth ratio is {breadth_ratio if breadth_ratio is not None else 'unavailable'}, "
        f"advancers {advancers if advancers is not None else 'unavailable'} and decliners {decliners if decliners is not None else 'unavailable'}. "
        f"The computed contrarian signal is {signal}. Note: {note}."
    )
    return ai(prompt, max_tokens=300)


# ── SENTIMENT DATA ──────────────────────────────────────────────────────────

def fetch_pcr():
    url = "https://www.nseindia.com/api/option-chain-indices?symbol=NIFTY"
    data = nse_get(url)
    if not data:
        return None

    records = data.get("records", {})
    oc_data = records.get("data", [])
    total_call_oi = 0
    total_put_oi = 0

    for item in oc_data:
        ce = item.get("CE", {})
        pe = item.get("PE", {})
        total_call_oi += int(ce.get("openInterest", 0) or 0)
        total_put_oi += int(pe.get("openInterest", 0) or 0)

    if total_call_oi <= 0:
        return None

    pcr = round(total_put_oi / total_call_oi, 2)
    return {
        "pcr": pcr,
        "total_call_oi": total_call_oi,
        "total_put_oi": total_put_oi,
    }


def fetch_india_vix():
    try:
        df = yf.download("^INDIAVIX", period="60d", interval="1d",
                         progress=False, auto_adjust=True)
        if df.empty:
            raise ValueError("No VIX data")
        close = df["Close"].dropna()
        if len(close) < 20:
            raise ValueError("Insufficient VIX history")

        current = float(close.iloc[-1])
        sma20 = float(close.tail(20).mean())
        delta_pct = round((current - sma20) / sma20 * 100, 2)
        return {
            "vix": round(current, 2),
            "sma20": round(sma20, 2),
            "delta_pct": delta_pct,
            "history": close.tail(20).reset_index().rename(columns={"Close": "VIX"}),
        }
    except Exception:
        return None


def fetch_market_breadth():
    try:
        tickers = list(SENTIMENT_WATCHLIST.values())
        df = yf.download(tickers, period="10d", interval="1d",
                         progress=False, auto_adjust=True)
        if df.empty:
            return None

        close_df = df["Close"] if isinstance(df.columns, pd.MultiIndex) else df
        if close_df.empty or len(close_df) < 2:
            return None

        returns = close_df.pct_change().iloc[-1].dropna()
        advancers = int((returns > 0).sum())
        decliners = int((returns < 0).sum())
        total = advancers + decliners
        breadth_ratio = round(advancers / total, 2) if total > 0 else None

        return {
            "advancers": advancers,
            "decliners": decliners,
            "breadth_ratio": breadth_ratio,
            "return_series": returns.sort_values(ascending=False).reset_index(
                names=["ticker"]).rename(columns={0: "return"}),
        }
    except Exception:
        return None


def compute_contrarian_signal(pcr, vix_delta_pct, breadth_ratio):
    if pcr is None or vix_delta_pct is None or breadth_ratio is None:
        return "INSUFFICIENT DATA", "Waiting for all sentiment inputs."

    notes = []
    signal = "NEUTRAL"

    if pcr >= 1.3:
        notes.append("PCR extreme fear")
    elif pcr <= 0.7:
        notes.append("PCR extreme greed")

    if vix_delta_pct >= 15:
        notes.append("India VIX spike")
    elif vix_delta_pct <= -15:
        notes.append("India VIX crush")

    if breadth_ratio <= 0.35:
        notes.append("weak breadth")
    elif breadth_ratio >= 0.65:
        notes.append("strong breadth")

    if "PCR extreme fear" in notes and "India VIX spike" in notes and breadth_ratio <= 0.4:
        signal = "CONTRARIAN BULLISH"
        note = "Extreme fear + VIX spike + weak breadth suggests a bounce setup."
    elif "PCR extreme greed" in notes and "India VIX crush" in notes and breadth_ratio >= 0.6:
        signal = "CONTRARIAN BEARISH"
        note = "Extreme greed + VIX crush + strong breadth suggests a reversal risk."
    elif "PCR extreme fear" in notes and breadth_ratio <= 0.4:
        signal = "CONTRARIAN BULLISH"
        note = "High PCR fear with weak breadth is a contrarian buy signal."
    elif "PCR extreme greed" in notes and breadth_ratio >= 0.6:
        signal = "CONTRARIAN BEARISH"
        note = "Low PCR greed with strong breadth is a contrarian sell signal."
    elif "India VIX spike" in notes and breadth_ratio <= 0.45:
        signal = "CONTRARIAN BULLISH"
        note = "VIX spike with narrow breadth favors contrarian bullish caution."
    elif "India VIX crush" in notes and breadth_ratio >= 0.55:
        signal = "CONTRARIAN BEARISH"
        note = "VIX crush with broad advance bias favors contrarian caution."
    else:
        signal = "NEUTRAL"
        note = "No strong contrarian edge at this time."

    return signal, note


def summarize_sentiment():
    pcr_data = fetch_pcr()
    vix_data = fetch_india_vix()
    breadth_data = fetch_market_breadth()

    pcr = pcr_data["pcr"] if pcr_data else None
    vix = vix_data["vix"] if vix_data else None
    vix_sma20 = vix_data["sma20"] if vix_data else None
    vix_delta_pct = vix_data["delta_pct"] if vix_data else None
    advancers = breadth_data["advancers"] if breadth_data else None
    decliners = breadth_data["decliners"] if breadth_data else None
    breadth_ratio = breadth_data["breadth_ratio"] if breadth_data else None

    signal, note = compute_contrarian_signal(pcr, vix_delta_pct, breadth_ratio)
    save_sentiment(pcr, vix, vix_sma20, vix_delta_pct,
                   advancers, decliners, breadth_ratio, signal, note)

    analysis = sentiment_ai_summary(pcr, vix, vix_delta_pct,
                                    breadth_ratio, advancers, decliners,
                                    signal, note)

    send_telegram(
        f"<b>📡 Market Sentiment Update</b>\n"
        f"PCR: {pcr if pcr is not None else 'N/A'}\n"
        f"India VIX: {vix if vix is not None else 'N/A'} "
        f"({f'{vix_delta_pct:+.2f}% vs 20d' if vix_delta_pct is not None else 'N/A'})\n"
        f"Breadth: {breadth_ratio if breadth_ratio is not None else 'N/A'}\n"
        f"Advancers / Decliners: {advancers if advancers is not None else 'N/A'} / {decliners if decliners is not None else 'N/A'}\n"
        f"Signal: {signal}\n"
        f"{note}\n"
    )

    return {
        "pcr": pcr,
        "vix": vix,
        "vix_sma20": vix_sma20,
        "vix_delta_pct": vix_delta_pct,
        "advancers": advancers,
        "decliners": decliners,
        "breadth_ratio": breadth_ratio,
        "signal": signal,
        "note": note,
        "analysis": analysis,
        "pcr_data": pcr_data,
        "vix_history": vix_data["history"] if vix_data else None,
        "breadth_returns": breadth_data["return_series"] if breadth_data else None,
    }


# ── STREAMLIT TAB ──────────────────────────────────────────────────────────────

def render_sentiment_tracker_tab():
    import streamlit as st

    st.subheader("📡 Market Sentiment Tracker")
    st.caption("PCR extremes · India VIX spike/crush · breadth ratio · contrarian alerts")

    if st.button("Refresh sentiment"):
        st.session_state["sentiment_refresh"] = True

    refresh = st.session_state.get("sentiment_refresh", False)
    if refresh or "sentiment_data" not in st.session_state:
        with st.spinner("Fetching sentiment data…"):
            st.session_state["sentiment_data"] = summarize_sentiment()
            st.session_state["sentiment_refresh"] = False

    data = st.session_state.get("sentiment_data", {})
    pcr = data.get("pcr")
    vix = data.get("vix")
    vix_sma20 = data.get("vix_sma20")
    vix_delta_pct = data.get("vix_delta_pct")
    advancers = data.get("advancers")
    decliners = data.get("decliners")
    breadth_ratio = data.get("breadth_ratio")
    signal = data.get("signal")
    note = data.get("note")

    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Put/Call Ratio", f"{pcr:.2f}" if pcr is not None else "N/A",
                  "Extreme" if pcr and (pcr >= 1.3 or pcr <= 0.7) else "Normal")
    with col2:
        subtitle = f"{vix:.2f}" if vix is not None else "N/A"
        detail = f"{vix_delta_pct:+.2f}% vs 20d" if vix_delta_pct is not None else ""
        st.metric("India VIX", subtitle, detail)
    with col3:
        st.metric("Advancers", f"{advancers}" if advancers is not None else "N/A",
                  "Breadth" if breadth_ratio is not None else "")
    with col4:
        st.metric("Breadth ratio", f"{breadth_ratio:.2f}" if breadth_ratio is not None else "N/A",
                  signal)

    st.markdown(f"**Contrarian signal:** {signal}")
    st.info(note)

    st.markdown("---")
    st.markdown("### What this means")

    if pcr is None:
        st.warning("Put/Call ratio could not be fetched from NSE India.")
    else:
        st.markdown(
            f"- PCR is <b>{pcr:.2f}</b>. "
            f"{('High fear' if pcr >= 1.3 else 'High greed' if pcr <= 0.7 else 'Neutral market sentiment')}.")

    if vix is None:
        st.warning("India VIX data unavailable. Check ticker ^INDIAVIX in yfinance.")
    else:
        st.markdown(
            f"- India VIX is <b>{vix:.2f}</b>, {vix_delta_pct:+.2f}% vs 20-day average.")

    if breadth_ratio is None:
        st.warning("Breadth ratio unavailable because market breadth data could not be calculated.")
    else:
        st.markdown(
            f"- Advance/decline ratio is <b>{breadth_ratio:.2f}</b>.")

    st.markdown("---")

    st.markdown("### Recent sentiment history")
    history = load_sentiment_history(10)
    if history.empty:
        st.info("No historical sentiment data yet. Refresh to generate the first record.")
    else:
        def style_row(row):
            if row["Signal"] == "CONTRARIAN BULLISH":
                return ["background-color:#1a3a2a;color:#2ecc71"] * len(row)
            if row["Signal"] == "CONTRARIAN BEARISH":
                return ["background-color:#3a1a1a;color:#e74c3c"] * len(row)
            return [""] * len(row)

        disp = history[["timestamp", "pcr", "vix", "vix_delta_pct",
                        "advancers", "decliners", "breadth_ratio", "signal"]].copy()
        disp.columns = ["Time", "PCR", "VIX", "VIX % vs 20d", "Advancers",
                        "Decliners", "Breadth", "Signal"]
        st.dataframe(disp.style.apply(style_row, axis=1), use_container_width=True)

    if data.get("vix_history") is not None:
        st.markdown("---")
        st.markdown("### India VIX last 20 sessions")
        st.line_chart(data["vix_history"].set_index("Date")["VIX"])

    if data.get("breadth_returns") is not None:
        st.markdown("---")
        st.markdown("### Latest breadth sample returns")
        st.dataframe(data["breadth_returns"].head(10), use_container_width=True)


# ── CLI SUPPORT ───────────────────────────────────────────────────────────────

def print_summary():
    result = summarize_sentiment()
    print("\nMarket Sentiment Summary")
    print("------------------------")
    print(f"PCR: {result['pcr']}")
    print(f"India VIX: {result['vix']} ({result['vix_delta_pct']}% vs 20d)")
    print(f"Advancers: {result['advancers']}, Decliners: {result['decliners']}")
    print(f"Breadth ratio: {result['breadth_ratio']}")
    print(f"Signal: {result['signal']}")
    print(f"Note: {result['note']}\n")


if __name__ == "__main__":
    init_db()
    print("Sentiment Tracker — NSE India")
    print("1) Run a sentiment summary")
    print("2) Show recent history")
    choice = input("Choose 1 or 2: ").strip()
    if choice == "1":
        print_summary()
    elif choice == "2":
        df = load_sentiment_history(20)
        if df.empty:
            print("No history found.")
        else:
            print(df.to_string(index=False))
    else:
        print("Invalid choice.")
