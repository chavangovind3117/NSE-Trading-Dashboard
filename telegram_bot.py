"""
NSE Telegram Alert Bot
======================
Run this SEPARATELY from Streamlit — it works in the background.

Setup:
1. Message @BotFather on Telegram → /newbot → copy the token
2. Message @userinfobot on Telegram → copy your chat_id
3. Fill in TELEGRAM_TOKEN and CHAT_ID below
4. Run: python telegram_bot.py

It will scan your watchlist every hour during market hours
and send you a Telegram message when a signal triggers.
"""

import yfinance as yf
import pandas as pd
import requests
import schedule
import time
from groq import Groq
from datetime import datetime
import sqlite3
import os
from config import TELEGRAM_TOKEN, CHAT_ID, GROQ_API_KEY, GROQ_MODEL

# ── CONFIG — loaded from environment / .env ──────────────────────────────────

# Stocks to monitor
WATCHLIST = {
    "NIFTY 50 Index":  "^NSEI",
    "BANK NIFTY":      "^NSEBANK",
    "Reliance":        "RELIANCE.NS",
    "TCS":             "TCS.NS",
    "HDFC Bank":       "HDFCBANK.NS",
    "Infosys":         "INFY.NS",
    "ICICI Bank":      "ICICIBANK.NS",
    "Kotak Bank":      "KOTAKBANK.NS",
    "SBI":             "SBIN.NS",
    "Bajaj Finance":   "BAJFINANCE.NS",
    "Wipro":           "WIPRO.NS",
    "Bharti Airtel":   "BHARTIARTL.NS",
    "Asian Paints":    "ASIANPAINT.NS",
    "HUL":             "HINDUNILVR.NS",
    "Axis Bank":       "AXISBANK.NS",
    "Maruti":          "MARUTI.NS",
    "Sun Pharma":      "SUNPHARMA.NS",
    "Titan":           "TITAN.NS",
    "L&T":             "LT.NS",
    "Tata Motors":     "TATAMOTORS.NS",
    "ONGC":            "ONGC.NS",
    "Power Grid":      "POWERGRID.NS",
    "NTPC":            "NTPC.NS",
    "Adani Ports":     "ADANIPORTS.NS",
    "Tech Mahindra":   "TECHM.NS",
    "Zomato":          "ZOMATO.NS",
    "IRCTC":           "IRCTC.NS",
    "Paytm":           "PAYTM.NS",
    "Nykaa":           "NYKAA.NS",
    "Dixon Tech":      "DIXON.NS",
}

# Alert conditions — edit to match your strategy
ALERT_CONDITIONS = {
    "rsi_oversold":   lambda s: s.get("rsi") and s["rsi"] < 35,
    "rsi_overbought": lambda s: s.get("rsi") and s["rsi"] > 65,
    "bullish_ema":    lambda s: s.get("ema20") and s.get("ema50") and s["latest"] > s["ema20"] > s["ema50"],
    "bearish_ema":    lambda s: s.get("ema20") and s.get("ema50") and s["latest"] < s["ema20"] < s["ema50"],
    "near_52w_high":  lambda s: s.get("high52") and s["latest"] >= s["high52"] * 0.98,
    "near_52w_low":   lambda s: s.get("low52")  and s["latest"] <= s["low52"]  * 1.02,
}

DB_PATH = "nse_trading.db"

# ── Telegram ──────────────────────────────────────────────────────────────────
def send_telegram(message: str):
    url = f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/sendMessage"
    payload = {"chat_id": CHAT_ID, "text": message, "parse_mode": "HTML"}
    try:
        r = requests.post(url, json=payload, timeout=10)
        if r.status_code == 200:
            print(f"[{now()}] Telegram sent ✓")
        else:
            print(f"[{now()}] Telegram error: {r.text}")
    except Exception as e:
        print(f"[{now()}] Telegram failed: {e}")

def now():
    return datetime.now().strftime("%H:%M:%S")

def today():
    return datetime.now().strftime("%d %b %Y")

# ── Indicators ────────────────────────────────────────────────────────────────
def compute_indicators(df: pd.DataFrame) -> pd.DataFrame:
    if len(df) < 20:
        return df
    df = df.copy()
    df["EMA20"]  = df["Close"].ewm(span=20, adjust=False).mean()
    df["EMA50"]  = df["Close"].ewm(span=50, adjust=False).mean()
    df["EMA200"] = df["Close"].ewm(span=200, adjust=False).mean()
    delta = df["Close"].diff()
    gain  = delta.where(delta > 0, 0).rolling(14).mean()
    loss  = (-delta.where(delta < 0, 0)).rolling(14).mean()
    df["RSI"] = 100 - (100 / (1 + gain / loss))
    tr = pd.concat([df["High"]-df["Low"],
                    (df["High"]-df["Close"].shift()).abs(),
                    (df["Low"]-df["Close"].shift()).abs()], axis=1).max(axis=1)
    df["ATR"] = tr.rolling(14).mean()
    return df

def get_stats(df: pd.DataFrame) -> dict:
    if df.empty or len(df) < 2:
        return {}
    c = df["Close"]
    lat, prev = float(c.iloc[-1]), float(c.iloc[-2])
    chg = lat - prev
    def gv(col): return float(df[col].iloc[-1]) if col in df.columns and pd.notna(df[col].iloc[-1]) else None
    return {
        "latest": lat, "change": chg, "change_pct": (chg/prev)*100,
        "high52": float(c.tail(252).max()) if len(c)>=252 else float(c.max()),
        "low52":  float(c.tail(252).min()) if len(c)>=252 else float(c.min()),
        "rsi":    gv("RSI"), "ema20": gv("EMA20"), "ema50": gv("EMA50"),
        "ema200": gv("EMA200"), "atr": gv("ATR"),
    }

# ── Groq quick signal ─────────────────────────────────────────────────────────
def get_ai_signal(stock_name, ticker, stats):
    try:
        client = Groq(api_key=GROQ_API_KEY)
        def fmt(v): return f"₹{v:,.2f}" if v else "N/A"
        prompt = f"""NSE stock alert analysis. Be very brief.

Stock: {stock_name} ({ticker})
Price: ₹{stats.get('latest',0):,.2f} ({stats.get('change_pct',0):+.2f}%)
RSI: {f"{stats['rsi']:.1f}" if stats.get('rsi') else 'N/A'}
EMA20: {fmt(stats.get('ema20'))} | EMA50: {fmt(stats.get('ema50'))}
52W High: {fmt(stats.get('high52'))} | 52W Low: {fmt(stats.get('low52'))}

In 3 lines max:
Line 1: SIGNAL: BUY / SELL / HOLD / WATCH
Line 2: One sentence on why
Line 3: Key level to watch"""

        r = client.chat.completions.create(
            model=GROQ_MODEL,
            messages=[{"role":"user","content":prompt}],
            temperature=0.3, max_tokens=150
        )
        return r.choices[0].message.content.strip()
    except Exception as e:
        return f"AI unavailable: {e}"

# ── Save alert to DB ──────────────────────────────────────────────────────────
def save_alert(stock, ticker, alert_type, price, message):
    try:
        conn = sqlite3.connect(DB_PATH)
        conn.execute("""CREATE TABLE IF NOT EXISTS alerts (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp TEXT, stock TEXT, ticker TEXT,
            alert_type TEXT, price REAL, message TEXT)""")
        conn.execute("INSERT INTO alerts (timestamp,stock,ticker,alert_type,price,message) VALUES (?,?,?,?,?,?)",
            (datetime.now().strftime("%Y-%m-%d %H:%M:%S"), stock, ticker, alert_type, price, message))
        conn.commit(); conn.close()
    except Exception as e:
        print(f"DB save error: {e}")

# ── Market hours check ────────────────────────────────────────────────────────
def is_market_hours():
    now_t = datetime.now()
    # NSE: Mon-Fri, 9:15 AM - 3:30 PM IST
    if now_t.weekday() >= 5:  # Saturday, Sunday
        return False
    hour, minute = now_t.hour, now_t.minute
    market_open  = hour > 9  or (hour == 9  and minute >= 15)
    market_close = hour < 15 or (hour == 15 and minute <= 30)
    return market_open and market_close

# ── Main scan ─────────────────────────────────────────────────────────────────
def scan_and_alert():
    if not is_market_hours():
        print(f"[{now()}] Market closed — skipping scan")
        return

    print(f"[{now()}] Starting watchlist scan…")
    alerts_triggered = []

    for stock_name, ticker in WATCHLIST.items():
        try:
            df = yf.download(ticker, period="6mo", interval="1d",
                             progress=False, auto_adjust=True)
            if df.empty:
                continue
            df.columns = [c[0] if isinstance(c, tuple) else c for c in df.columns]
            df.dropna(inplace=True)
            df = compute_indicators(df)
            stats = get_stats(df)

            triggered = []
            for condition_name, condition_fn in ALERT_CONDITIONS.items():
                if condition_fn(stats):
                    triggered.append(condition_name)

            if triggered:
                ai_signal = get_ai_signal(stock_name, ticker, stats)
                condition_labels = {
                    "rsi_oversold":   "RSI Oversold (<35)",
                    "rsi_overbought": "RSI Overbought (>65)",
                    "bullish_ema":    "Bullish EMA alignment",
                    "bearish_ema":    "Bearish EMA alignment",
                    "near_52w_high":  "Near 52-week high",
                    "near_52w_low":   "Near 52-week low",
                }
                conditions_text = "\n".join([f"  • {condition_labels.get(c,c)}" for c in triggered])
                message = (
                    f"🔔 <b>NSE Alert — {stock_name}</b>\n"
                    f"💰 ₹{stats['latest']:,.2f} ({stats['change_pct']:+.2f}%)\n"
                    f"📊 RSI: {stats['rsi']:.1f if stats.get('rsi') else 'N/A'}\n\n"
                    f"<b>Conditions triggered:</b>\n{conditions_text}\n\n"
                    f"<b>AI View:</b>\n{ai_signal}\n\n"
                    f"🕐 {today()} {now()} IST"
                )
                alerts_triggered.append((stock_name, ticker, triggered[0], stats["latest"], message))
                print(f"[{now()}] ALERT: {stock_name} — {', '.join(triggered)}")

        except Exception as e:
            print(f"[{now()}] Error scanning {stock_name}: {e}")

    # Send all alerts
    if alerts_triggered:
        for stock_name, ticker, alert_type, price, message in alerts_triggered:
            send_telegram(message)
            save_alert(stock_name, ticker, alert_type, price, message)
            time.sleep(1)  # avoid Telegram rate limit
    else:
        print(f"[{now()}] Scan complete — no alerts triggered")

def morning_briefing():
    if not is_market_hours():
        return
    msg = (
        f"🌅 <b>Good morning! NSE Market Briefing</b>\n"
        f"📅 {today()}\n\n"
        f"Market opens at 9:15 AM IST\n"
        f"Running first watchlist scan now…\n\n"
        f"Stocks monitored: {', '.join(WATCHLIST.keys())}"
    )
    send_telegram(msg)
    scan_and_alert()

# ── Scheduler ─────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("=" * 50)
    print("NSE Telegram Alert Bot started")
    print(f"Monitoring: {', '.join(WATCHLIST.keys())}")
    print("Scanning every hour during market hours (9:15 AM - 3:30 PM IST)")
    print("=" * 50)

    # Validate config
    if "YOUR_" in TELEGRAM_TOKEN or "YOUR_" in CHAT_ID or "YOUR_" in GROQ_API_KEY:
        print("\n⚠️  ERROR: Please fill in TELEGRAM_TOKEN, CHAT_ID, and GROQ_API_KEY at the top of this file.")
        print("See setup instructions in the comment at the top.\n")
        exit(1)

    # Morning briefing at 9:10 AM every weekday
    schedule.every().monday.at("09:10").do(morning_briefing)
    schedule.every().tuesday.at("09:10").do(morning_briefing)
    schedule.every().wednesday.at("09:10").do(morning_briefing)
    schedule.every().thursday.at("09:10").do(morning_briefing)
    schedule.every().friday.at("09:10").do(morning_briefing)

    # Scan every hour
    schedule.every().hour.do(scan_and_alert)

    # Run first scan immediately
    scan_and_alert()

    print("\nBot is running. Press Ctrl+C to stop.\n")
    while True:
        schedule.run_pending()
        time.sleep(60)
