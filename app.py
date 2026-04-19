import streamlit as st
import yfinance as yf
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from groq import Groq
from datetime import datetime
import os
import numpy as np
import sqlite3

# ── Optional modules ────────────────────────────────────────────────────────
CONFLUENCE_OK = False
GLOBAL_OK = False
SMART_OK = False
TIME_STRATEGIES_OK = False
SENTIMENT_OK = False

try:
    from confluence_checker import render_confluence_tab, init_db as init_confluence_db
    CONFLUENCE_OK = True
except ImportError:
    CONFLUENCE_OK = False

try:
    from global_signal import render_global_signal_tab, init_db as init_global_db
    GLOBAL_OK = True
except ImportError:
    GLOBAL_OK = False

try:
    from smart_money import render_smart_money_tab, init_db as init_smart_db
    SMART_OK = True
except ImportError:
    SMART_OK = False

try:
    from time_strategies import render_time_strategies_tab, init_db as init_time_db
    TIME_STRATEGIES_OK = True
except ImportError:
    TIME_STRATEGIES_OK = False

try:
    from sentiment_tracker import render_sentiment_tracker_tab, init_db as init_sentiment_db
    SENTIMENT_OK = True
except ImportError:
    SENTIMENT_OK = False

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="NSE Trading Dashboard",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
<style>
.signal-buy   { color:#2ecc71; font-weight:700; font-size:20px; }
.signal-sell  { color:#e74c3c; font-weight:700; font-size:20px; }
.signal-hold  { color:#f39c12; font-weight:700; font-size:20px; }
.signal-watch { color:#3498db; font-weight:700; font-size:20px; }
.ai-box {
    background:#1a1f2e;
    border-left:4px solid #6c63ff;
    border-radius:8px;
    padding:16px 20px;
    margin:10px 0;
    font-size:14px;
    line-height:1.8;
    color:#e0e0e0;
}
</style>
""", unsafe_allow_html=True)

# ── Constants ─────────────────────────────────────────────────────────────────
NSE_STOCKS = {
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

TIMEFRAME_CONFIG = {
    "Intraday (5m)":  {"period":"1d",  "interval":"5m",  "label":"5-minute intraday"},
    "Intraday (15m)": {"period":"5d",  "interval":"15m", "label":"15-minute intraday"},
    "Daily":          {"period":"6mo", "interval":"1d",  "label":"daily"},
    "Weekly":         {"period":"2y",  "interval":"1wk", "label":"weekly"},
    "Monthly":        {"period":"5y",  "interval":"1mo", "label":"monthly"},
    "Yearly":         {"period":"max", "interval":"3mo", "label":"quarterly/yearly"},
}

GROQ_MODELS = {
    "Llama 3.3 70B (Best)":   "llama-3.3-70b-versatile",
    "Llama 3.1 8B (Fastest)": "llama-3.1-8b-instant",
    "Mixtral 8x7B":           "mixtral-8x7b-32768",
}

DB_PATH = "nse_trading.db"

# ── SQLite — all tables ───────────────────────────────────────────────────────
def init_db():
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute("""CREATE TABLE IF NOT EXISTS analysis_history (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        timestamp TEXT, stock TEXT, ticker TEXT,
        timeframe TEXT, signal TEXT, price REAL, analysis TEXT)""")
    c.execute("""CREATE TABLE IF NOT EXISTS trade_journal (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        date TEXT, stock TEXT, ticker TEXT, direction TEXT,
        entry REAL, exit REAL, qty INTEGER,
        pnl REAL, pnl_pct REAL, notes TEXT, outcome TEXT)""")
    c.execute("""CREATE TABLE IF NOT EXISTS confluence_alerts (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        timestamp TEXT, stock TEXT, ticker TEXT, direction TEXT,
        daily_sig TEXT, weekly_sig TEXT, monthly_sig TEXT,
        price REAL, confidence INTEGER, analysis TEXT)""")
    conn.commit()
    conn.close()

def save_analysis(stock, ticker, timeframe, signal, price, analysis):
    conn = sqlite3.connect(DB_PATH)
    conn.execute(
        "INSERT INTO analysis_history (timestamp,stock,ticker,timeframe,signal,price,analysis) VALUES (?,?,?,?,?,?,?)",
        (datetime.now().strftime("%Y-%m-%d %H:%M:%S"), stock, ticker, timeframe, signal, price, analysis))
    conn.commit(); conn.close()

def load_history(limit=100):
    conn = sqlite3.connect(DB_PATH)
    df = pd.read_sql_query(
        "SELECT * FROM analysis_history ORDER BY id DESC LIMIT ?", conn, params=(limit,))
    conn.close(); return df

def save_trade(stock, ticker, direction, entry, exit_p, qty, notes):
    pnl     = (exit_p - entry) * qty if direction == "BUY" else (entry - exit_p) * qty
    pnl_pct = ((exit_p - entry) / entry * 100) if direction == "BUY" else ((entry - exit_p) / entry * 100)
    outcome = "WIN" if pnl > 0 else "LOSS"
    conn = sqlite3.connect(DB_PATH)
    conn.execute(
        "INSERT INTO trade_journal (date,stock,ticker,direction,entry,exit,qty,pnl,pnl_pct,notes,outcome) VALUES (?,?,?,?,?,?,?,?,?,?,?)",
        (datetime.now().strftime("%Y-%m-%d"), stock, ticker, direction,
         entry, exit_p, qty, round(pnl, 2), round(pnl_pct, 2), notes, outcome))
    conn.commit(); conn.close()

def load_trades():
    conn = sqlite3.connect(DB_PATH)
    df = pd.read_sql_query("SELECT * FROM trade_journal ORDER BY id DESC", conn)
    conn.close(); return df

def delete_trade(trade_id):
    conn = sqlite3.connect(DB_PATH)
    conn.execute("DELETE FROM trade_journal WHERE id=?", (trade_id,))
    conn.commit(); conn.close()

init_db()
if CONFLUENCE_OK:
    init_confluence_db()
if SENTIMENT_OK:
    init_sentiment_db()
if TIME_STRATEGIES_OK:
    init_time_db()

# ── Data fetching ─────────────────────────────────────────────────────────────
@st.cache_data(ttl=300)
def fetch_data(ticker, period, interval):
    try:
        df = yf.download(ticker, period=period, interval=interval,
                         progress=False, auto_adjust=True)
        if df.empty: return df
        df.columns = [c[0] if isinstance(c, tuple) else c for c in df.columns]
        df.dropna(inplace=True)
        return df
    except Exception:
        return pd.DataFrame()

# ── Indicators ────────────────────────────────────────────────────────────────
def ema(s, n): return s.ewm(span=n, adjust=False).mean()

def rsi(s, n=14):
    d = s.diff()
    g = d.where(d > 0, 0).rolling(n).mean()
    l = (-d.where(d < 0, 0)).rolling(n).mean()
    return 100 - (100 / (1 + g / l))

def macd(s, f=12, sl=26, sig=9):
    ml  = s.ewm(span=f,  adjust=False).mean() - s.ewm(span=sl, adjust=False).mean()
    sl2 = ml.ewm(span=sig, adjust=False).mean()
    return pd.DataFrame({"MACD": ml, "MACD_Signal": sl2, "MACD_Hist": ml - sl2})

def bbands(s, n=20, std=2):
    m = s.rolling(n).mean(); d = s.rolling(n).std()
    return pd.DataFrame({"BB_Upper": m + d*std, "BB_Lower": m - d*std})

def atr_fn(h, l, c, n=14):
    tr = pd.concat([h-l, (h-c.shift()).abs(), (l-c.shift()).abs()], axis=1).max(axis=1)
    return tr.rolling(n).mean()

def add_indicators(df):
    if len(df) < 20: return df
    df = df.copy()
    df["EMA20"]  = ema(df["Close"], 20)
    df["EMA50"]  = ema(df["Close"], 50)
    df["EMA200"] = ema(df["Close"], 200)
    df["RSI"]    = rsi(df["Close"])
    md = macd(df["Close"])
    df["MACD"] = md["MACD"]; df["MACD_Signal"] = md["MACD_Signal"]; df["MACD_Hist"] = md["MACD_Hist"]
    bb = bbands(df["Close"])
    df["BB_Upper"] = bb["BB_Upper"]; df["BB_Lower"] = bb["BB_Lower"]
    df["ATR"] = atr_fn(df["High"], df["Low"], df["Close"])
    return df

def get_stats(df):
    if df.empty or len(df) < 2: return {}
    c = df["Close"]
    lat, prev = float(c.iloc[-1]), float(c.iloc[-2])
    chg = lat - prev
    def gv(col): return float(df[col].iloc[-1]) if col in df.columns and pd.notna(df[col].iloc[-1]) else None
    return {
        "latest": lat, "change": chg, "change_pct": (chg/prev)*100,
        "high52": float(c.tail(252).max()) if len(c)>=252 else float(c.max()),
        "low52":  float(c.tail(252).min()) if len(c)>=252 else float(c.min()),
        "rsi": gv("RSI"), "ema20": gv("EMA20"), "ema50": gv("EMA50"),
        "ema200": gv("EMA200"), "atr": gv("ATR"),
        "vol_cur": float(df["Volume"].iloc[-1]) if "Volume" in df.columns else None,
        "vol_avg": float(df["Volume"].tail(20).mean()) if "Volume" in df.columns else None,
    }

# ── Chart ─────────────────────────────────────────────────────────────────────
def build_chart(df, name, timeframe):
    fig = make_subplots(rows=3, cols=1, shared_xaxes=True,
        row_heights=[0.6, 0.2, 0.2], vertical_spacing=0.03,
        subplot_titles=(f"{name} — {timeframe}", "RSI (14)", "MACD"))
    fig.add_trace(go.Candlestick(
        x=df.index, open=df["Open"], high=df["High"], low=df["Low"], close=df["Close"],
        name="Price", increasing_line_color="#2ecc71", decreasing_line_color="#e74c3c"), row=1, col=1)
    for col, color, dash in [("EMA20","#3498db","solid"),("EMA50","#e67e22","dash"),("EMA200","#9b59b6","dot")]:
        if col in df.columns:
            fig.add_trace(go.Scatter(x=df.index, y=df[col], name=col,
                line=dict(color=color, width=1.2, dash=dash)), row=1, col=1)
    if "BB_Upper" in df.columns:
        fig.add_trace(go.Scatter(x=df.index, y=df["BB_Upper"], name="BB Upper",
            line=dict(color="rgba(150,150,150,0.4)", width=1)), row=1, col=1)
        fig.add_trace(go.Scatter(x=df.index, y=df["BB_Lower"], name="BB Lower",
            line=dict(color="rgba(150,150,150,0.4)", width=1),
            fill="tonexty", fillcolor="rgba(150,150,150,0.06)"), row=1, col=1)
    if "RSI" in df.columns:
        fig.add_trace(go.Scatter(x=df.index, y=df["RSI"], name="RSI",
            line=dict(color="#e74c3c", width=1.5)), row=2, col=1)
        fig.add_hline(y=70, line_dash="dash", line_color="red",   line_width=0.8, row=2, col=1)
        fig.add_hline(y=30, line_dash="dash", line_color="green", line_width=0.8, row=2, col=1)
    if "MACD" in df.columns:
        hc = ["#2ecc71" if v >= 0 else "#e74c3c" for v in df["MACD_Hist"].fillna(0)]
        fig.add_trace(go.Bar(x=df.index, y=df["MACD_Hist"], name="Histogram",
            marker_color=hc, opacity=0.7), row=3, col=1)
        fig.add_trace(go.Scatter(x=df.index, y=df["MACD"], name="MACD",
            line=dict(color="#3498db", width=1.2)), row=3, col=1)
        fig.add_trace(go.Scatter(x=df.index, y=df["MACD_Signal"], name="Signal",
            line=dict(color="#e67e22", width=1.2)), row=3, col=1)
    fig.update_layout(height=630, template="plotly_dark",
        paper_bgcolor="#0e1117", plot_bgcolor="#0e1117",
        font=dict(color="#fafafa", size=12),
        legend=dict(orientation="h", y=1.02, x=0),
        xaxis_rangeslider_visible=False, margin=dict(l=10, r=10, t=40, b=10))
    for r in [1, 2, 3]: fig.update_yaxes(gridcolor="#1e2130", row=r, col=1)
    return fig

# ── Groq AI ───────────────────────────────────────────────────────────────────
def ask_groq(api_key, model, stock_name, ticker, tf_label, stats, df):
    client = Groq(api_key=api_key)
    recent = df.tail(10)[["Open","High","Low","Close","Volume"]].round(2).to_string()
    def fmt(v): return f"₹{v:,.2f}" if v else "N/A"
    prompt = f"""You are an expert NSE India technical analyst.

Stock: {stock_name} ({ticker}) | Timeframe: {tf_label}
Price: ₹{stats.get('latest',0):,.2f} | Change: {stats.get('change_pct',0):+.2f}%
52W High: {fmt(stats.get('high52'))} | 52W Low: {fmt(stats.get('low52'))}
RSI: {f"{stats['rsi']:.1f}" if stats.get('rsi') else 'N/A'} | EMA20: {fmt(stats.get('ema20'))} | EMA50: {fmt(stats.get('ema50'))} | EMA200: {fmt(stats.get('ema200'))} | ATR: {fmt(stats.get('atr'))}

Recent OHLCV:
{recent}

Respond EXACTLY:
1. SIGNAL: [BUY / SELL / HOLD / WATCH]
2. TREND: [BULLISH / BEARISH / SIDEWAYS] — one sentence
3. KEY LEVELS:
   - Support: ₹___
   - Resistance: ₹___
4. SETUP: 2 sentences on what is forming
5. WHAT TO WATCH: 2 specific things
6. RISK NOTE: One sentence

Under 220 words. NSE India context."""
    r = client.chat.completions.create(model=model,
        messages=[
            {"role":"system","content":"You are a professional NSE India technical analyst. Be concise and actionable."},
            {"role":"user","content":prompt}],
        temperature=0.3, max_tokens=600)
    return r.choices[0].message.content

def ask_groq_scanner(api_key, model, scan_tf, summary):
    client = Groq(api_key=api_key)
    r = client.chat.completions.create(model=model,
        messages=[
            {"role":"system","content":"You are a professional NSE India market analyst."},
            {"role":"user","content":
                f"NSE scan — {scan_tf} timeframe:\n\n{summary}\n\n"
                f"Give:\n1. TOP 2 OPPORTUNITIES: Best setups with reasons\n"
                f"2. AVOID LIST: Risky stocks and why\n3. MARKET THEME: Sector pattern\n\nUnder 200 words."}],
        temperature=0.3, max_tokens=500)
    return r.choices[0].message.content

def ask_groq_bias(api_key, model, trades_text):
    client = Groq(api_key=api_key)
    r = client.chat.completions.create(model=model,
        messages=[
            {"role":"system","content":"You are a trading psychologist specialising in retail traders."},
            {"role":"user","content":
                f"My NSE trades:\n\n{trades_text}\n\n"
                f"Analyse and tell me:\n"
                f"1. BIGGEST WEAKNESS: My single worst pattern\n"
                f"2. EMOTIONAL PATTERNS: FOMO, revenge trading, overtrading signs\n"
                f"3. WIN/LOSS PATTERNS: When I win vs lose\n"
                f"4. ONE THING TO FIX: Single most impactful change\n\n"
                f"Be direct and honest. Under 250 words."}],
        temperature=0.4, max_tokens=600)
    return r.choices[0].message.content

def parse_signal(text):
    t = text.upper()
    for s in ["BUY","SELL","HOLD","WATCH"]:
        if f"SIGNAL: {s}" in t or f"1. SIGNAL: {s}" in t: return s
    return "—"

# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.title("📈 NSE Dashboard")
    st.caption("Powered by Groq AI — Free")
    st.divider()

    st.subheader("Groq API Key")
    api_key = st.text_input("Groq API key", type="password",
        placeholder="gsk_...",
        help="Free at console.groq.com — no credit card needed")
    if not api_key:
        st.info("Get free key at **console.groq.com**")

    model_name = st.selectbox("AI Model", list(GROQ_MODELS.keys()), index=0)
    model_id   = GROQ_MODELS[model_name]
    st.divider()

    st.subheader("Stock")
    custom_ticker = st.text_input("Custom ticker (e.g. ZOMATO.NS)", value="")
    selected_name = st.selectbox("Or pick from list", list(NSE_STOCKS.keys()))
    if custom_ticker.strip():
        ticker_sym  = custom_ticker.strip().upper()
        ticker_name = ticker_sym
    else:
        ticker_sym  = NSE_STOCKS[selected_name]
        ticker_name = selected_name

    st.subheader("Timeframe")
    timeframe = st.selectbox("Select timeframe", list(TIMEFRAME_CONFIG.keys()), index=2)
    tf_cfg    = TIMEFRAME_CONFIG[timeframe]

    st.subheader("Watchlist Scanner")
    scan_stocks = st.multiselect("Stocks to scan", list(NSE_STOCKS.keys()),
        default=["Reliance","TCS","HDFC Bank","Infosys","ICICI Bank"])
    st.divider()

    run_analysis = st.button("Run AI Analysis", type="primary",  use_container_width=True)
    run_scan     = st.button("Scan Watchlist",  type="secondary", use_container_width=True)
    st.caption("Data: Yahoo Finance · AI: Groq (free)")

# ── Main header ───────────────────────────────────────────────────────────────
st.title(f"📊 {ticker_name}")
st.caption(f"`{ticker_sym}` · {timeframe} · {datetime.now().strftime('%d %b %Y %H:%M IST')}")

with st.spinner("Fetching NSE data…"):
    df = fetch_data(ticker_sym, tf_cfg["period"], tf_cfg["interval"])

if df.empty:
    st.error(f"No data for `{ticker_sym}`. Check ticker and try again.")
    st.stop()

df = add_indicators(df)
stats = get_stats(df)

# ── Metrics row ───────────────────────────────────────────────────────────────
m1, m2, m3, m4, m5, m6 = st.columns(6)
arrow = "▲" if stats.get("change", 0) >= 0 else "▼"
m1.metric("LTP (₹)",  f"{stats.get('latest',0):,.2f}",
          f"{arrow} {stats.get('change',0):+.2f} ({stats.get('change_pct',0):+.2f}%)")
m2.metric("52W High", f"₹{stats.get('high52',0):,.2f}")
m3.metric("52W Low",  f"₹{stats.get('low52',0):,.2f}")
m4.metric("RSI",      f"{stats['rsi']:.1f}" if stats.get("rsi") else "—")
m5.metric("EMA 20",   f"₹{stats['ema20']:,.2f}" if stats.get("ema20") else "—")
m6.metric("ATR",      f"₹{stats['atr']:.2f}"   if stats.get("atr")  else "—")
st.divider()

# ── All 12 tabs ───────────────────────────────────────────────────────────────
(tab_chart, tab_data, tab_ai, tab_scan,
 tab_size, tab_journal, tab_history, tab_confluence, tab_global, tab_smart, tab_sentiment, tab_time) = st.tabs([
    "📈 Chart", "📋 Data", "🤖 AI Analysis", "🔍 Scanner",
    "📐 Position Sizer", "📓 Trade Journal", "🕐 History", "⚡ Confluence",
    "🌍 Global Signal", "🏦 Smart Money", "📡 Market Sentiment", "⏰ Time Strategies"
])

# ── 1. CHART ──────────────────────────────────────────────────────────────────
with tab_chart:
    st.plotly_chart(build_chart(df, ticker_name, timeframe), use_container_width=True)
    ca, cb, cc = st.columns(3)
    with ca:
        st.markdown("**EMA Alignment**")
        e20, e50, ltp = stats.get("ema20"), stats.get("ema50"), stats.get("latest", 0)
        if e20 and e50:
            if ltp > e20 > e50:   st.success("Bullish — Price > EMA20 > EMA50")
            elif ltp < e20 < e50: st.error("Bearish — Price < EMA20 < EMA50")
            else:                 st.warning("Mixed — no clear alignment")
    with cb:
        st.markdown("**RSI Zone**")
        rv = stats.get("rsi")
        if rv:
            if rv > 70:   st.error(f"RSI {rv:.1f} — Overbought")
            elif rv < 30: st.success(f"RSI {rv:.1f} — Oversold")
            else:         st.info(f"RSI {rv:.1f} — Neutral")
    with cc:
        st.markdown("**Volume**")
        vc, va = stats.get("vol_cur"), stats.get("vol_avg")
        if vc and va and va > 0:
            r = vc / va
            if r > 1.5:   st.success(f"{r:.1f}x avg — High conviction")
            elif r < 0.5: st.warning(f"{r:.1f}x avg — Low conviction")
            else:         st.info(f"{r:.1f}x avg — Normal volume")

# ── 2. DATA ───────────────────────────────────────────────────────────────────
with tab_data:
    st.subheader("OHLCV + Indicators (last 50 bars)")
    sc = [c for c in ["Open","High","Low","Close","Volume",
                       "EMA20","EMA50","EMA200","RSI","MACD","ATR"] if c in df.columns]
    st.dataframe(df[sc].tail(50).round(2).sort_index(ascending=False), use_container_width=True)

# ── 3. AI ANALYSIS ────────────────────────────────────────────────────────────
with tab_ai:
    st.subheader(f"AI Analysis — {ticker_name} ({timeframe})")
    st.caption(f"Model: {model_name}")
    if not api_key:
        st.warning("Enter your Groq API key in the sidebar.")
    elif run_analysis or st.button("Analyze Now", key="btn_analyze"):
        with st.spinner(f"Analyzing {ticker_name}…"):
            try:
                analysis = ask_groq(api_key, model_id, ticker_name, ticker_sym,
                                    tf_cfg["label"], stats, df)
                signal  = parse_signal(analysis)
                sc2 = {"BUY":"signal-buy","SELL":"signal-sell",
                       "HOLD":"signal-hold","WATCH":"signal-watch"}.get(signal,"signal-hold")
                c1, c2 = st.columns([1, 3])
                with c1: st.markdown(f"<span class='{sc2}'>{signal}</span>", unsafe_allow_html=True)
                with c2: st.caption(f"{datetime.now().strftime('%H:%M:%S')} · ₹{stats.get('latest',0):,.2f}")
                st.markdown(f'<div class="ai-box">{analysis.replace(chr(10),"<br>")}</div>',
                            unsafe_allow_html=True)
                st.session_state[f"ai_{ticker_sym}_{timeframe}"] = (analysis, signal)
                save_analysis(ticker_name, ticker_sym, timeframe, signal,
                              stats.get("latest", 0), analysis)
                st.success("✓ Saved to history")
            except Exception as e:
                st.error(f"Groq error: {e}")
    elif f"ai_{ticker_sym}_{timeframe}" in st.session_state:
        analysis, signal = st.session_state[f"ai_{ticker_sym}_{timeframe}"]
        sc2 = {"BUY":"signal-buy","SELL":"signal-sell",
               "HOLD":"signal-hold","WATCH":"signal-watch"}.get(signal,"signal-hold")
        st.markdown(f"<span class='{sc2}'>{signal}</span>", unsafe_allow_html=True)
        st.markdown(f'<div class="ai-box">{analysis.replace(chr(10),"<br>")}</div>',
                    unsafe_allow_html=True)
        st.caption("Cached — click Analyze Now to refresh")
    else:
        st.info("Click **Run AI Analysis** in sidebar or **Analyze Now** above.")

# ── 4. SCANNER ────────────────────────────────────────────────────────────────
with tab_scan:
    st.subheader("Watchlist Scanner")
    if not scan_stocks:
        st.info("Select stocks in the sidebar to scan.")
    else:
        scan_tf  = st.selectbox("Timeframe", ["Daily","Weekly","Monthly"], key="scan_tf_sel")
        scan_cfg = TIMEFRAME_CONFIG[scan_tf]
        if run_scan or st.button("Run Scanner", key="btn_scan"):
            results = []
            prog = st.progress(0, text="Scanning…")
            for i, sname in enumerate(scan_stocks):
                sdf = fetch_data(NSE_STOCKS[sname], scan_cfg["period"], scan_cfg["interval"])
                if not sdf.empty:
                    sdf = add_indicators(sdf); s = get_stats(sdf)
                    rv2 = s.get("rsi")
                    e20b, e50b, ltpb = s.get("ema20"), s.get("ema50"), s.get("latest", 0)
                    esig = ("Bullish" if (e20b and e50b and ltpb > e20b > e50b) else
                            "Bearish" if (e20b and e50b and ltpb < e20b < e50b) else "Mixed")
                    results.append({
                        "Stock":     sname,
                        "LTP (₹)":  f"{s.get('latest',0):,.2f}",
                        "Chg %":    f"{s.get('change_pct',0):+.2f}%",
                        "RSI":      f"{rv2:.1f}" if rv2 else "—",
                        "RSI Zone": ("Overbought" if rv2 and rv2 > 70 else
                                     "Oversold"   if rv2 and rv2 < 30 else "Neutral"),
                        "EMA Trend": esig,
                        "52W High": f"₹{s.get('high52',0):,.2f}",
                        "52W Low":  f"₹{s.get('low52',0):,.2f}",
                    })
                prog.progress((i+1)/len(scan_stocks), text=f"Scanned {i+1}/{len(scan_stocks)}")
            prog.empty()
            if results:
                sdf2 = pd.DataFrame(results)
                def cr(row):
                    st2 = [""] * len(row); cols = sdf2.columns.tolist()
                    if row["RSI Zone"] == "Oversold":
                        st2[cols.index("RSI Zone")] = "background-color:#1a3a2a;color:#2ecc71"
                    elif row["RSI Zone"] == "Overbought":
                        st2[cols.index("RSI Zone")] = "background-color:#3a1a1a;color:#e74c3c"
                    if row["EMA Trend"] == "Bullish":
                        st2[cols.index("EMA Trend")] = "background-color:#1a3a2a;color:#2ecc71"
                    elif row["EMA Trend"] == "Bearish":
                        st2[cols.index("EMA Trend")] = "background-color:#3a1a1a;color:#e74c3c"
                    return st2
                st.dataframe(sdf2.style.apply(cr, axis=1),
                             use_container_width=True, hide_index=True)
                if api_key:
                    st.divider(); st.subheader("AI Watchlist Summary")
                    with st.spinner("AI reading scan…"):
                        try:
                            ins = ask_groq_scanner(api_key, model_id, scan_tf,
                                                   sdf2.to_string(index=False))
                            st.markdown(f'<div class="ai-box">{ins.replace(chr(10),"<br>")}</div>',
                                        unsafe_allow_html=True)
                        except Exception as e:
                            st.error(f"Groq error: {e}")

# ── 5. POSITION SIZER ─────────────────────────────────────────────────────────
with tab_size:
    st.subheader("📐 Position Size Calculator")
    st.caption("ATR-based sizing — exact quantity, stop-loss and R-multiple targets")
    ps1, ps2 = st.columns(2)
    with ps1:
        capital     = st.number_input("Total capital (₹)", value=100000, step=10000)
        risk_pct    = st.slider("Risk per trade (%)", 0.5, 5.0, 1.0, 0.5)
        entry_price = st.number_input("Entry price (₹)",
                                      value=float(stats.get("latest", 100)), step=1.0)
        atr_val     = stats.get("atr")
        atr_mult    = st.slider("Stop-loss ATR multiplier", 1.0, 3.0, 1.5, 0.5)
    with ps2:
        if atr_val:
            risk_amt  = capital * (risk_pct / 100)
            stop_dist = atr_val * atr_mult
            stop_p    = entry_price - stop_dist
            qty       = max(1, int(risk_amt / stop_dist))
            pos_val   = qty * entry_price
            st.markdown("**Your trade plan**")
            r1, r2 = st.columns(2)
            r1.metric("Quantity",       f"{qty} shares")
            r2.metric("Position value", f"₹{pos_val:,.0f} ({pos_val/capital*100:.1f}%)")
            r1.metric("Stop-loss",      f"₹{stop_p:,.2f}", f"-₹{stop_dist:.2f}")
            r2.metric("Max risk",       f"₹{risk_amt:,.0f}")
            st.markdown("**R-multiple targets**")
            t1, t2, t3 = st.columns(3)
            for col, mult, label in [(t1,1,"1R"),(t2,2,"2R"),(t3,3,"3R")]:
                tgt = entry_price + stop_dist * mult
                col.metric(f"{label} target", f"₹{tgt:,.2f}",
                           f"+{(tgt-entry_price)/entry_price*100:.1f}%")
            st.info(f"ATR = ₹{atr_val:.2f} · Stop = {atr_mult}x ATR = ₹{stop_dist:.2f} from entry")
        else:
            st.warning("Select Daily or longer timeframe to get ATR-based sizing.")

# ── 6. TRADE JOURNAL ──────────────────────────────────────────────────────────
with tab_journal:
    st.subheader("📓 Trade Journal")
    j1, j2 = st.columns([1, 1])

    with j1:
        st.markdown("**Log a new trade**")
        j_stock = st.selectbox("Stock", list(NSE_STOCKS.keys()), key="j_stock")
        j_dir   = st.radio("Direction", ["BUY", "SELL"], horizontal=True)
        jc1, jc2 = st.columns(2)
        j_entry = jc1.number_input("Entry (₹)", min_value=0.01, value=100.0, step=0.5)
        j_exit  = jc2.number_input("Exit (₹)",  min_value=0.01, value=105.0, step=0.5)
        j_qty   = st.number_input("Quantity", min_value=1, value=10, step=1)
        j_notes = st.text_area("Notes (setup, reason, emotion)", height=90,
                               placeholder="e.g. EMA20 bounce, strong RSI, stopped out on news…")
        if st.button("💾 Save Trade", type="primary"):
            save_trade(j_stock, NSE_STOCKS[j_stock], j_dir,
                       j_entry, j_exit, int(j_qty), j_notes)
            st.success("Trade saved!"); st.rerun()

    with j2:
        tdf = load_trades()
        if not tdf.empty:
            total  = len(tdf)
            wins   = len(tdf[tdf["outcome"] == "WIN"])
            losses = total - wins
            total_pnl = tdf["pnl"].sum()
            avg_w = tdf[tdf["outcome"]=="WIN"]["pnl"].mean()  if wins   > 0 else 0
            avg_l = tdf[tdf["outcome"]=="LOSS"]["pnl"].mean() if losses > 0 else 0

            st.markdown("**Performance summary**")
            sa, sb = st.columns(2)
            sa.metric("Total trades", total)
            sb.metric("Win rate",     f"{wins/total*100:.0f}%")
            sa.metric("Total P&L",    f"₹{total_pnl:,.0f}")
            sb.metric("Avg W / L",    f"₹{avg_w:.0f} / ₹{avg_l:.0f}")

            # Expectancy
            if wins > 0 and losses > 0:
                expectancy = (wins/total * avg_w) + (losses/total * avg_l)
                st.metric("Expectancy per trade", f"₹{expectancy:,.0f}",
                          delta="Positive edge" if expectancy > 0 else "Negative edge")

            if api_key:
                if st.button("🧠 AI Bias Analysis", key="bias_btn"):
                    with st.spinner("Analysing your trading patterns…"):
                        try:
                            bias = ask_groq_bias(api_key, model_id,
                                                 tdf.to_string(index=False))
                            st.markdown(
                                f'<div class="ai-box">{bias.replace(chr(10),"<br>")}</div>',
                                unsafe_allow_html=True)
                        except Exception as e:
                            st.error(f"Error: {e}")

            st.divider()
            st.markdown("**Recent trades**")
            disp = tdf[["date","stock","direction","entry","exit",
                         "qty","pnl","pnl_pct","outcome"]].head(20).copy()
            disp["pnl"]     = disp["pnl"].round(2)
            disp["pnl_pct"] = disp["pnl_pct"].round(2)

            def ct(row):
                c = ("background-color:#1a3a2a;color:#2ecc71" if row["outcome"] == "WIN"
                     else "background-color:#3a1a1a;color:#e74c3c")
                return [c] * len(row)

            st.dataframe(disp.style.apply(ct, axis=1),
                         use_container_width=True, hide_index=True)

            # Delete trade
            st.divider()
            trade_ids = tdf["id"].tolist()
            del_id = st.selectbox("Delete a trade (by ID)", ["—"] + [str(i) for i in trade_ids])
            if del_id != "—" and st.button("Delete", type="secondary"):
                delete_trade(int(del_id)); st.success("Deleted"); st.rerun()
        else:
            st.info("No trades yet. Log your first trade on the left.")

# ── 7. HISTORY ────────────────────────────────────────────────────────────────
with tab_history:
    st.subheader("🕐 Analysis History")
    st.caption("Every AI analysis is auto-saved here. Persists across restarts.")

    hdf = load_history()
    if hdf.empty:
        st.info("No history yet. Run some AI analyses first.")
    else:
        hf1, hf2, hf3 = st.columns(3)
        fs = hf1.selectbox("Stock",     ["All"] + sorted(hdf["stock"].unique().tolist()))
        fg = hf2.selectbox("Signal",    ["All", "BUY", "SELL", "HOLD", "WATCH"])
        ft = hf3.selectbox("Timeframe", ["All"] + sorted(hdf["timeframe"].unique().tolist()))

        filt = hdf.copy()
        if fs != "All": filt = filt[filt["stock"]     == fs]
        if fg != "All": filt = filt[filt["signal"]    == fg]
        if ft != "All": filt = filt[filt["timeframe"] == ft]

        hc1, hc2, hc3, hc4 = st.columns(4)
        hc1.metric("Total analyses", len(filt))
        hc2.metric("BUY signals",    len(filt[filt["signal"] == "BUY"]))
        hc3.metric("SELL signals",   len(filt[filt["signal"] == "SELL"]))
        hc4.metric("HOLD signals",   len(filt[filt["signal"] == "HOLD"]))
        st.divider()

        for _, row in filt.head(30).iterrows():
            icon = {"BUY":"🟢","SELL":"🔴","HOLD":"🟡","WATCH":"🔵"}.get(row["signal"], "⚪")
            with st.expander(
                f"{icon} [{row['signal']}]  {row['stock']}  ·  "
                f"{row['timeframe']}  ·  ₹{row['price']:,.2f}  ·  {row['timestamp']}"
            ):
                st.markdown(
                    f'<div class="ai-box">{row["analysis"].replace(chr(10),"<br>")}</div>',
                    unsafe_allow_html=True)

        st.divider()
        col_exp, col_clear = st.columns([2, 1])
        with col_exp:
            if st.button("📥 Export history to CSV"):
                st.download_button(
                    "Download CSV", filt.to_csv(index=False),
                    "nse_analysis_history.csv", "text/csv")
        with col_clear:
            if st.button("🗑️ Clear all history", type="secondary"):
                conn = sqlite3.connect(DB_PATH)
                conn.execute("DELETE FROM analysis_history")
                conn.commit(); conn.close()
                st.success("History cleared"); st.rerun()

# ── 8. CONFLUENCE ─────────────────────────────────────────────────────────────
with tab_confluence:
    if CONFLUENCE_OK:
        render_confluence_tab(ticker_name, ticker_sym)
    else:
        st.warning("⚡ confluence_checker.py not found in the same folder as app.py.")
        st.info("Place confluence_checker.py next to app.py and restart Streamlit to enable this tab.")

# ── Footer ────────────────────────────────────────────────────────────────────
st.divider()
st.caption("⚠️ For educational purposes only. Not financial advice. Do your own research before trading.")

# ── Global Signal tab ──────────────────────────────────────────────────────────
with tab_global:
    if GLOBAL_OK:
        render_global_signal_tab()
    else:
        st.warning("🌍 global_signal.py not found in the same folder as app.py.")
        st.info("Place global_signal.py next to app.py and restart Streamlit.")

# ── Smart Money tab ────────────────────────────────────────────────────────────
with tab_smart:
    if SMART_OK:
        render_smart_money_tab()
    else:
        st.warning("🏦 smart_money.py not found in the same folder as app.py.")
        st.info("Place smart_money.py next to app.py and restart Streamlit.")

# ── Sentiment Tracker tab ───────────────────────────────────────────────────────
with tab_sentiment:
    if SENTIMENT_OK:
        render_sentiment_tracker_tab()
    else:
        st.warning("📡 sentiment_tracker.py not found in the same folder as app.py.")
        st.info("Place sentiment_tracker.py next to app.py and restart Streamlit.")

# ── Time Strategies tab ─────────────────────────────────────────────────────────
with tab_time:
    if TIME_STRATEGIES_OK:
        render_time_strategies_tab()
    else:
        st.warning("⏰ time_strategies.py not found in the same folder as app.py.")
        st.info("Place time_strategies.py next to app.py and restart Streamlit.")
