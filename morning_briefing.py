"""
NSE Morning Market Briefing
============================
Runs every day at 9:00 AM IST automatically.
Scans your full watchlist, analyses NIFTY direction,
finds top opportunities + stocks to avoid,
then sends a beautifully formatted Telegram message.

Setup:
  1. Fill in your credentials below (same as telegram_bot.py)
  2. Run:  python morning_briefing.py
  3. Keep it running — it fires at 9:00 AM every weekday

Run it alongside your dashboard and telegram_bot.py
in a third VS Code terminal.
"""

import yfinance as yf
import pandas as pd
import requests
import schedule
import time
from groq import Groq
from datetime import datetime
import pytz
from config import TELEGRAM_TOKEN, CHAT_ID, GROQ_API_KEY, GROQ_MODEL

# ── CONFIG — loaded from environment / .env ─────────────────────────────────
IST = pytz.timezone("Asia/Kolkata")

# ── Watchlist — edit freely ───────────────────────────────────────────────────
WATCHLIST = {
    # Indices
    "NIFTY 50":      "^NSEI",
    "BANK NIFTY":    "^NSEBANK",
    # Large caps
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
}

# Sectors for rotation check
SECTORS = {
    "IT":      ["TCS.NS", "INFY.NS", "WIPRO.NS", "TECHM.NS"],
    "Banking": ["HDFCBANK.NS", "ICICIBANK.NS", "SBIN.NS", "KOTAKBANK.NS", "AXISBANK.NS"],
    "Auto":    ["MARUTI.NS", "TATAMOTORS.NS"],
    "Pharma":  ["SUNPHARMA.NS"],
    "Energy":  ["RELIANCE.NS", "ONGC.NS"],
    "Infra":   ["LT.NS"],
    "FMCG":    ["HINDUNILVR.NS"],
}

# ── Indicators ────────────────────────────────────────────────────────────────
def compute(df):
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
    fast = df["Close"].ewm(span=12, adjust=False).mean()
    slow = df["Close"].ewm(span=26, adjust=False).mean()
    df["MACD"] = fast - slow
    df["MACD_Signal"] = df["MACD"].ewm(span=9, adjust=False).mean()
    df["MACD_Hist"] = df["MACD"] - df["MACD_Signal"]
    return df


def stats(df):
    if df.empty or len(df) < 2:
        return None
    c = df["Close"]
    lat, prev = float(c.iloc[-1]), float(c.iloc[-2])
    chg = lat - prev
    w = df.tail(5)["Close"]
    wk_chg = ((float(w.iloc[-1]) - float(w.iloc[0])) / float(w.iloc[0])) * 100 if len(w) >= 5 else 0
    def gv(col): return float(df[col].iloc[-1]) if col in df.columns and pd.notna(df[col].iloc[-1]) else None
    return {
        "latest": lat, "prev": prev,
        "chg": chg, "chg_pct": (chg / prev) * 100,
        "wk_chg": wk_chg,
        "high52": float(c.tail(252).max()) if len(c) >= 252 else float(c.max()),
        "low52":  float(c.tail(252).min()) if len(c) >= 252 else float(c.min()),
        "rsi":    gv("RSI"), "ema20": gv("EMA20"),
        "ema50":  gv("EMA50"), "ema200": gv("EMA200"),
        "atr":    gv("ATR"),
        "macd_hist": gv("MACD_Hist"),
        "vol_cur": float(df["Volume"].iloc[-1]) if "Volume" in df.columns else None,
        "vol_avg": float(df["Volume"].tail(20).mean()) if "Volume" in df.columns else None,
    }


def fetch(ticker, period="6mo", interval="1d"):
    try:
        df = yf.download(ticker, period=period, interval=interval,
                         progress=False, auto_adjust=True)
        if df.empty:
            return pd.DataFrame()
        df.columns = [c[0] if isinstance(c, tuple) else c for c in df.columns]
        df.dropna(inplace=True)
        return df
    except Exception:
        return pd.DataFrame()


# ── Scoring system ────────────────────────────────────────────────────────────
def score_stock(s):
    """Returns score -10 to +10 and list of reasons."""
    if not s:
        return 0, []
    score = 0
    bullish, bearish = [], []

    # RSI
    rsi = s.get("rsi")
    if rsi:
        if rsi < 35:
            score += 2; bullish.append(f"RSI oversold ({rsi:.0f})")
        elif rsi < 50:
            score += 1; bullish.append(f"RSI neutral-low ({rsi:.0f})")
        elif rsi > 70:
            score -= 2; bearish.append(f"RSI overbought ({rsi:.0f})")
        elif rsi > 60:
            score -= 1

    # EMA alignment
    ltp, e20, e50, e200 = s.get("latest",0), s.get("ema20"), s.get("ema50"), s.get("ema200")
    if e20 and e50:
        if ltp > e20 > e50:
            score += 2; bullish.append("Bullish EMA stack")
        elif ltp < e20 < e50:
            score -= 2; bearish.append("Bearish EMA stack")
    if e200:
        if ltp > e200:
            score += 1; bullish.append("Above EMA200")
        else:
            score -= 1; bearish.append("Below EMA200")

    # MACD
    mh = s.get("macd_hist")
    if mh:
        if mh > 0:
            score += 1; bullish.append("MACD bullish")
        else:
            score -= 1; bearish.append("MACD bearish")

    # Volume
    vc, va = s.get("vol_cur"), s.get("vol_avg")
    if vc and va and va > 0:
        ratio = vc / va
        if ratio > 1.5:
            score += 1; bullish.append(f"High volume ({ratio:.1f}x)")
        elif ratio < 0.5:
            score -= 1; bearish.append("Low volume")

    # Near 52W levels
    h52, l52 = s.get("high52"), s.get("low52")
    if h52 and ltp >= h52 * 0.98:
        score += 1; bullish.append("Near 52W high")
    if l52 and ltp <= l52 * 1.03:
        score -= 1; bearish.append("Near 52W low")

    # Weekly momentum
    wk = s.get("wk_chg", 0)
    if wk > 3:
        score += 1; bullish.append(f"Strong week (+{wk:.1f}%)")
    elif wk < -3:
        score -= 1; bearish.append(f"Weak week ({wk:.1f}%)")

    return max(-10, min(10, score)), bullish, bearish


# ── Sector analysis ───────────────────────────────────────────────────────────
def analyse_sectors():
    sector_scores = {}
    for sector, tickers in SECTORS.items():
        changes = []
        for ticker in tickers:
            df = fetch(ticker, "5d", "1d")
            if not df.empty:
                df = compute(df)
                s = stats(df)
                if s:
                    changes.append(s["chg_pct"])
        if changes:
            avg = sum(changes) / len(changes)
            sector_scores[sector] = avg
    return sector_scores


# ── AI briefing generator ─────────────────────────────────────────────────────
def generate_briefing(nifty_stats, banknifty_stats, scan_results, sector_scores):
    client = Groq(api_key=GROQ_API_KEY)

    # Build scan summary
    scan_text = ""
    for name, s, score, bull, bear in scan_results:
        if not s:
            continue
        rsi_str = f"{s['rsi']:.0f}" if s.get("rsi") else "N/A"
        ema_str = "Bullish" if (s.get("ema20") and s.get("ema50") and s["latest"] > s["ema20"] > s["ema50"]) else \
                  "Bearish" if (s.get("ema20") and s.get("ema50") and s["latest"] < s["ema20"] < s["ema50"]) else "Mixed"
        scan_text += f"{name}: ₹{s['latest']:,.0f} ({s['chg_pct']:+.1f}%) | RSI {rsi_str} | EMA {ema_str} | Score {score:+d}\n"

    sector_text = " | ".join([f"{k}: {v:+.1f}%" for k, v in sorted(sector_scores.items(), key=lambda x: -x[1])])

    nifty_str = f"₹{nifty_stats['latest']:,.0f} ({nifty_stats['chg_pct']:+.1f}%)" if nifty_stats else "N/A"
    bank_str  = f"₹{banknifty_stats['latest']:,.0f} ({banknifty_stats['chg_pct']:+.1f}%)" if banknifty_stats else "N/A"

    prompt = f"""You are an expert NSE India market analyst. Today is {datetime.now(IST).strftime('%A, %d %B %Y')}.

Generate a sharp, actionable pre-market briefing for an NSE trader.

MARKET OVERVIEW:
- NIFTY 50:   {nifty_str}
- BANK NIFTY: {bank_str}

SECTOR PERFORMANCE (last close):
{sector_text}

WATCHLIST SCAN (scored -10 bearish to +10 bullish):
{scan_text}

Write the briefing in this EXACT format — no deviation:

🌅 MARKET MOOD: [one punchy sentence on overall market tone — bullish/bearish/mixed and why]

📊 NIFTY VIEW: [2 sentences — key level to watch, likely direction today]

🏆 TOP 3 OPPORTUNITIES:
1. [Stock] — [specific reason in one line, entry idea, key level]
2. [Stock] — [specific reason in one line, entry idea, key level]
3. [Stock] — [specific reason in one line, entry idea, key level]

🚫 AVOID TODAY:
1. [Stock] — [reason in one line]
2. [Stock] — [reason in one line]

🔥 HOT SECTOR: [sector name] — [one sentence why]
❄️ WEAK SECTOR: [sector name] — [one sentence why]

⚡ ONE KEY THING TO WATCH: [single most important thing for today — level, event, or condition]

Keep it under 300 words. Be specific with price levels. NSE India context only."""

    r = client.chat.completions.create(
        model=GROQ_MODEL,
        messages=[
            {"role": "system", "content": "You are a professional NSE India market analyst writing a daily pre-market briefing. Be sharp, specific, and actionable."},
            {"role": "user",   "content": prompt}
        ],
        temperature=0.4,
        max_tokens=800,
    )
    return r.choices[0].message.content.strip()


# ── Telegram sender ───────────────────────────────────────────────────────────
def send(text):
    url = f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/sendMessage"
    # Telegram has a 4096 char limit — split if needed
    chunks = [text[i:i+4000] for i in range(0, len(text), 4000)]
    for chunk in chunks:
        try:
            r = requests.post(url, json={
                "chat_id": CHAT_ID,
                "text": chunk,
                "parse_mode": "HTML"
            }, timeout=15)
            if r.status_code != 200:
                print(f"Telegram error: {r.text}")
        except Exception as e:
            print(f"Telegram send failed: {e}")
        time.sleep(0.5)


def send_chunk(title, body):
    send(f"<b>{title}</b>\n\n{body}")


# ── Main briefing function ────────────────────────────────────────────────────
def morning_briefing():
    now_ist = datetime.now(IST)
    date_str = now_ist.strftime("%A, %d %B %Y")
    time_str = now_ist.strftime("%I:%M %p IST")

    print(f"\n{'='*55}")
    print(f"Morning Briefing — {date_str} {time_str}")
    print(f"{'='*55}")

    # ── 1. Send opening header immediately ────────────────
    send(
        f"🌅 <b>NSE Morning Briefing</b>\n"
        f"📅 {date_str} | {time_str}\n"
        f"{'─'*30}\n"
        f"Scanning {len(WATCHLIST)} stocks... please wait ⏳"
    )

    # ── 2. Fetch NIFTY & BANK NIFTY ───────────────────────
    print("Fetching indices...")
    nifty_df = fetch("^NSEI");     nifty_df = compute(nifty_df) if not nifty_df.empty else nifty_df
    bank_df  = fetch("^NSEBANK");  bank_df  = compute(bank_df)  if not bank_df.empty  else bank_df
    nifty_s  = stats(nifty_df)
    bank_s   = stats(bank_df)

    # ── 3. Scan full watchlist ─────────────────────────────
    print("Scanning watchlist...")
    scan_results = []
    for name, ticker in WATCHLIST.items():
        if ticker in ["^NSEI", "^NSEBANK"]:
            continue
        df = fetch(ticker)
        if df.empty:
            continue
        df = compute(df)
        s  = stats(df)
        sc, bull, bear = score_stock(s)
        scan_results.append((name, s, sc, bull, bear))
        print(f"  {name}: score {sc:+d}")

    # Sort by score
    scan_results.sort(key=lambda x: x[2], reverse=True)

    # ── 4. Sector analysis ────────────────────────────────
    print("Analysing sectors...")
    sector_scores = analyse_sectors()

    # ── 5. Generate AI briefing ───────────────────────────
    print("Generating AI briefing...")
    try:
        briefing = generate_briefing(nifty_s, bank_s, scan_results, sector_scores)
    except Exception as e:
        briefing = f"AI briefing unavailable: {e}"
        print(f"AI error: {e}")

    # ── 6. Build & send the full message ──────────────────
    # Indices block
    def arrow(v): return "▲" if v >= 0 else "▼"
    nifty_line = f"NIFTY 50:    ₹{nifty_s['latest']:>10,.0f}  {arrow(nifty_s['chg_pct'])} {nifty_s['chg_pct']:+.2f}%" if nifty_s else "NIFTY 50: N/A"
    bank_line  = f"BANK NIFTY:  ₹{bank_s['latest']:>10,.0f}  {arrow(bank_s['chg_pct'])} {bank_s['chg_pct']:+.2f}%"  if bank_s  else "BANK NIFTY: N/A"

    indices_block = (
        f"<b>📊 Index Snapshot</b>\n"
        f"<code>{nifty_line}\n{bank_line}</code>"
    )

    # Sector block
    sorted_sectors = sorted(sector_scores.items(), key=lambda x: -x[1])
    sector_lines = "\n".join([
        f"{'🟢' if v >= 0 else '🔴'} {k}: {v:+.2f}%"
        for k, v in sorted_sectors
    ])
    sector_block = f"<b>🏭 Sector Snapshot</b>\n{sector_lines}"

    # Watchlist block — top 5 bullish + top 3 bearish
    bullish_stocks = [r for r in scan_results if r[2] > 0][:5]
    bearish_stocks = [r for r in scan_results if r[2] < 0][-3:]

    wl_lines = "<b>🟢 Bullish setups:</b>\n"
    for name, s, sc, bull, bear in bullish_stocks:
        if s:
            rsi_str = f"RSI {s['rsi']:.0f}" if s.get("rsi") else ""
            reasons = ", ".join(bull[:2])
            wl_lines += f"• <b>{name}</b> ₹{s['latest']:,.0f} ({s['chg_pct']:+.1f}%) | {rsi_str} | {reasons}\n"

    wl_lines += "\n<b>🔴 Bearish / avoid:</b>\n"
    for name, s, sc, bull, bear in reversed(bearish_stocks):
        if s:
            reasons = ", ".join(bear[:2])
            wl_lines += f"• <b>{name}</b> ₹{s['latest']:,.0f} ({s['chg_pct']:+.1f}%) | {reasons}\n"

    # Send all parts
    send(indices_block)
    time.sleep(1)
    send(sector_block)
    time.sleep(1)
    send(wl_lines)
    time.sleep(1)
    send(f"<b>🤖 AI Pre-Market Analysis</b>\n\n{briefing}")
    time.sleep(1)

    # Footer
    send(
        f"{'─'*30}\n"
        f"⏰ Market opens: <b>9:15 AM IST</b>\n"
        f"📌 Next briefing: <b>Tomorrow 9:00 AM</b>\n"
        f"⚠️ For educational use only. Not financial advice."
    )

    print(f"\n✓ Morning briefing sent at {time_str}")
    print(f"{'='*55}\n")


# ── Weekend checker ───────────────────────────────────────────────────────────
def is_weekday():
    return datetime.now(IST).weekday() < 5  # Mon=0 … Fri=4


def run_if_weekday():
    if is_weekday():
        morning_briefing()
    else:
        print(f"[{datetime.now(IST).strftime('%H:%M')}] Weekend — no briefing today")


# ── Scheduler ─────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("=" * 55)
    print("NSE Morning Briefing Service")
    print(f"Scheduled: 9:00 AM IST every weekday")
    print(f"Stocks:    {len(WATCHLIST)}")
    print(f"Sectors:   {len(SECTORS)}")
    print("=" * 55)

    # Validate credentials
    if "YOUR_" in TELEGRAM_TOKEN or "YOUR_" in CHAT_ID or "YOUR_" in GROQ_API_KEY:
        print("\n⚠️  Fill in TELEGRAM_TOKEN, CHAT_ID and GROQ_API_KEY at the top of this file.\n")
        exit(1)

    # Schedule 9:00 AM IST daily
    schedule.every().day.at("09:00").do(run_if_weekday)

    # Ask if user wants to test immediately
    print("\nPress ENTER to send a test briefing NOW")
    print("Or wait — it will auto-fire at 9:00 AM IST every weekday\n")
    import threading

    def wait_for_enter():
        input()
        print("Sending test briefing...\n")
        morning_briefing()

    t = threading.Thread(target=wait_for_enter, daemon=True)
    t.start()

    while True:
        schedule.run_pending()
        time.sleep(30)
