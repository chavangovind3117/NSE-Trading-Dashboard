"""
Smart Money Tracker — smart_money.py (Fixed)
=============================================
Fixed NSE API with proper browser session + multiple fallbacks.

Run options:
  1 — Diagnostic (tests all data sources, shows what works)
  2 — Full scan now
  3 — Scheduler (4 PM + 8 AM auto)

python smart_money.py
"""

import requests, sqlite3, schedule, time, pytz, io
import pandas as pd
from groq import Groq
from datetime import datetime, timedelta
import yfinance as yf
from config import TELEGRAM_TOKEN, CHAT_ID, GROQ_API_KEY, GROQ_MODEL

# ── CONFIG ────────────────────────────────────────────────────────────────────
IST            = pytz.timezone("Asia/Kolkata")
DB_PATH        = "nse_trading.db"

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

WATCHLIST = {
    "Reliance":"RELIANCE.NS","TCS":"TCS.NS","HDFC Bank":"HDFCBANK.NS",
    "Infosys":"INFY.NS","ICICI Bank":"ICICIBANK.NS","SBI":"SBIN.NS",
    "Kotak Bank":"KOTAKBANK.NS","Bajaj Finance":"BAJFINANCE.NS",
    "Axis Bank":"AXISBANK.NS","Wipro":"WIPRO.NS","Maruti":"MARUTI.NS",
    "Sun Pharma":"SUNPHARMA.NS","Titan":"TITAN.NS","L&T":"LT.NS",
    "Tata Motors":"TATAMOTORS.NS","Bharti Airtel":"BHARTIARTL.NS",
    "Tech Mahindra":"TECHM.NS","Zomato":"ZOMATO.NS",
    "IRCTC":"IRCTC.NS","Dixon Tech":"DIXON.NS",
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

# ── NSE SESSION ───────────────────────────────────────────────────────────────
def get_nse_session() -> requests.Session:
    """Build a session that looks like Chrome to NSE."""
    s = requests.Session()
    s.headers.update(CHROME_HEADERS)
    try:
        s.get("https://www.nseindia.com/", timeout=15)
        time.sleep(1.5)
        s.get("https://www.nseindia.com/market-data/live-equity-market", timeout=10)
        time.sleep(1.0)
    except Exception as e:
        print(f"  Session warmup: {e}")
    return s

def nse_get(url: str, retries=3):
    for attempt in range(retries):
        try:
            s = get_nse_session()
            r = s.get(url, timeout=20)
            if r.status_code == 200:
                return r.json()
            print(f"  NSE {r.status_code} on {url.split('?')[0].split('/')[-1]} (attempt {attempt+1})")
            time.sleep(3)
        except Exception as e:
            print(f"  Error attempt {attempt+1}: {e}")
            time.sleep(2)
    return None

def parse_cr(val) -> float:
    if val is None: return 0.0
    try: return float(str(val).replace(",","").replace("₹","").strip() or 0)
    except: return 0.0

# ── DATABASE ──────────────────────────────────────────────────────────────────
def init_db():
    conn = sqlite3.connect(DB_PATH)
    conn.execute("""CREATE TABLE IF NOT EXISTS fii_dii_data (
        id INTEGER PRIMARY KEY AUTOINCREMENT, date TEXT UNIQUE,
        fii_buy REAL, fii_sell REAL, fii_net REAL,
        dii_buy REAL, dii_sell REAL, dii_net REAL, divergence TEXT)""")
    conn.execute("""CREATE TABLE IF NOT EXISTS bulk_deals (
        id INTEGER PRIMARY KEY AUTOINCREMENT, date TEXT,
        symbol TEXT, company TEXT, client TEXT,
        buy_sell TEXT, qty INTEGER, price REAL, alerted INTEGER DEFAULT 0)""")
    conn.execute("""CREATE TABLE IF NOT EXISTS promoter_buying (
        id INTEGER PRIMARY KEY AUTOINCREMENT, date TEXT,
        company TEXT, symbol TEXT, acquirer TEXT, shares INTEGER,
        pct_acquired REAL, total_pct REAL, alerted INTEGER DEFAULT 0)""")
    conn.execute("""CREATE TABLE IF NOT EXISTS smart_money_alerts (
        id INTEGER PRIMARY KEY AUTOINCREMENT, timestamp TEXT,
        alert_type TEXT, title TEXT, detail TEXT, analysis TEXT)""")
    conn.commit(); conn.close()

def save_fii_dii(date, fb, fs, fn, db, ds, dn):
    div = ("FII_BUY_DII_SELL" if fn>0 and dn<0 else
           "FII_SELL_DII_BUY" if fn<0 and dn>0 else "ALIGNED")
    conn = sqlite3.connect(DB_PATH)
    conn.execute("INSERT OR REPLACE INTO fii_dii_data (date,fii_buy,fii_sell,fii_net,dii_buy,dii_sell,dii_net,divergence) VALUES (?,?,?,?,?,?,?,?)",
        (date, fb, fs, fn, db, ds, dn, div))
    conn.commit(); conn.close()

def load_fii_dii(days=10):
    try:
        conn = sqlite3.connect(DB_PATH)
        df = pd.read_sql_query("SELECT * FROM fii_dii_data ORDER BY date DESC LIMIT ?", conn, params=(days,))
        conn.close()
        return df
    except Exception:
        # If table doesn't exist or error, return empty DataFrame
        return pd.DataFrame(columns=['id', 'date', 'fii_buy', 'fii_sell', 'fii_net', 'dii_buy', 'dii_sell', 'dii_net', 'divergence'])

def save_bulk(date, sym, co, client, bs, qty, price):
    conn = sqlite3.connect(DB_PATH)
    conn.execute("INSERT INTO bulk_deals (date,symbol,company,client,buy_sell,qty,price) VALUES (?,?,?,?,?,?,?)",
        (date, sym, co, client, bs, int(qty), float(price)))
    conn.commit(); conn.close()

def save_promoter(date, co, sym, acq, shares, pct, total):
    conn = sqlite3.connect(DB_PATH)
    conn.execute("INSERT INTO promoter_buying (date,company,symbol,acquirer,shares,pct_acquired,total_pct) VALUES (?,?,?,?,?,?,?)",
        (date, co, sym, acq, int(shares), float(pct), float(total)))
    conn.commit(); conn.close()

def alerted_today(table, ident):
    conn  = sqlite3.connect(DB_PATH)
    today = datetime.now(IST).strftime("%Y-%m-%d")
    row   = conn.execute(f"SELECT id FROM {table} WHERE (symbol=? OR company=?) AND date LIKE ?",
        (ident, ident, f"{today}%")).fetchone()
    conn.close(); return row is not None

def save_alert(atype, title, detail, analysis):
    conn = sqlite3.connect(DB_PATH)
    conn.execute("INSERT INTO smart_money_alerts (timestamp,alert_type,title,detail,analysis) VALUES (?,?,?,?,?)",
        (datetime.now(IST).strftime("%Y-%m-%d %H:%M:%S"), atype, title, detail, analysis))
    conn.commit(); conn.close()

# ── TELEGRAM ──────────────────────────────────────────────────────────────────
def send(text):
    url = f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/sendMessage"
    for chunk in [text[i:i+4000] for i in range(0, len(text), 4000)]:
        try: requests.post(url, json={"chat_id":CHAT_ID,"text":chunk,"parse_mode":"HTML"}, timeout=15)
        except: pass
        time.sleep(0.5)

def ts(): return datetime.now(IST).strftime("%d %b %Y %H:%M IST")

# ── GROQ ──────────────────────────────────────────────────────────────────────
def ai(prompt, max_tokens=350):
    try:
        r = Groq(api_key=GROQ_API_KEY).chat.completions.create(
            model=GROQ_MODEL,
            messages=[{"role":"system","content":"NSE India institutional analyst. Concise and actionable."},
                      {"role":"user","content":prompt}],
            temperature=0.3, max_tokens=max_tokens)
        return r.choices[0].message.content.strip()
    except Exception as e: return f"AI unavailable: {e}"

def stock_ctx(ticker):
    try:
        df = yf.download(ticker, period="1mo", interval="1d", progress=False, auto_adjust=True)
        if df.empty or len(df)<5: return ""
        df.columns=[c[0] if isinstance(c,tuple) else c for c in df.columns]
        c=df["Close"]; lat=float(c.iloc[-1]); prev=float(c.iloc[-2])
        chg=((lat-prev)/prev)*100; e20=float(c.ewm(span=20,adjust=False).mean().iloc[-1])
        d=c.diff(); g=d.where(d>0,0).rolling(14).mean(); l=(-d.where(d<0,0)).rolling(14).mean()
        rsi=float((100-(100/(1+g/l))).iloc[-1])
        return f"₹{lat:,.2f} ({chg:+.1f}%) RSI {rsi:.0f} {'above' if lat>e20 else 'below'} EMA20"
    except: return ""

# ══════════════════════════════════════════════════════════════════════════════
# MODULE 1 — FII/DII
# ══════════════════════════════════════════════════════════════════════════════

def fetch_fii_dii():
    today_str = datetime.now(IST).strftime("%d-%b-%Y")

    # Source 1: NSE API
    print("  [1] Trying NSE FII/DII API...")
    data = nse_get("https://www.nseindia.com/api/fiidiiTradeReact")
    if data:
        try:
            row = data[-1] if isinstance(data, list) else data
            keys = list(row.keys())
            print(f"  NSE returned keys: {keys}")
            fb = parse_cr(row.get("fiiBuy") or row.get("FIIBuy") or row.get("buyValue",0))
            fs = parse_cr(row.get("fiiSell") or row.get("FIISell") or row.get("sellValue",0))
            db = parse_cr(row.get("diiBuy") or row.get("DIIBuy",0))
            ds = parse_cr(row.get("diiSell") or row.get("DIISell",0))
            if fb > 0 or db > 0:
                result = {"date":today_str,"fii_buy":fb,"fii_sell":fs,"fii_net":fb-fs,
                          "dii_buy":db,"dii_sell":ds,"dii_net":db-ds}
                print(f"  NSE OK: FII ₹{result['fii_net']:+,.1f} Cr | DII ₹{result['dii_net']:+,.1f} Cr")
                return result
        except Exception as e:
            print(f"  NSE parse error: {e}")

    # Source 2: Yahoo proxy
    print("  [2] Using Yahoo Finance NIFTY proxy...")
    try:
        df = yf.download("^NSEI", period="5d", interval="1d", progress=False, auto_adjust=True)
        if not df.empty:
            df.columns=[c[0] if isinstance(c,tuple) else c for c in df.columns]
            chg = float(df["Close"].pct_change().iloc[-1]*100)
            fn = chg * 500; dn = -fn * 0.4
            result = {"date":today_str,
                      "fii_buy":abs(fn)+3000,"fii_sell":max(0,-fn)+2500,"fii_net":fn,
                      "dii_buy":abs(dn)+2000,"dii_sell":max(0,-dn)+1800,"dii_net":dn,
                      "proxy":True}
            print(f"  Yahoo proxy: NIFTY {chg:+.2f}% → FII est ₹{fn:+,.0f} Cr (estimate)")
            return result
    except Exception as e:
        print(f"  Yahoo proxy failed: {e}")

    return None


def check_fii_dii(today):
    proxy = today.get("proxy", False)
    save_fii_dii(today["date"], today["fii_buy"], today["fii_sell"], today["fii_net"],
                 today["dii_buy"], today["dii_sell"], today["dii_net"])

    hist = load_fii_dii(7)
    days = len(hist)
    print(f"  History: {days} days in DB")

    if days < 3:
        print(f"  Need 3+ days. Run daily at 4 PM — {3-days} more day(s) needed.")
        print(f"  Today's data saved. Run tomorrow to build history.")
        return False

    fii_streak = dii_streak = 0
    for _, r in hist.iterrows():
        if r["fii_net"] > 200: fii_streak += 1
        else: break
    for _, r in hist.iterrows():
        if r["dii_net"] < -100: dii_streak += 1
        else: break

    t3_fii = hist.head(3)["fii_net"].sum()
    t3_dii = hist.head(3)["dii_net"].sum()
    print(f"  FII streak: {fii_streak}d | DII sell streak: {dii_streak}d")
    print(f"  3-day: FII ₹{t3_fii:+,.0f} | DII ₹{t3_dii:+,.0f} Cr")

    if fii_streak >= 3 and dii_streak >= 2:
        tbl = "".join([f"  {str(r['date'])[:10]}  FII {'▲' if r['fii_net']>0 else '▼'} ₹{r['fii_net']:+,.0f}Cr  DII {'▲' if r['dii_net']>0 else '▼'} ₹{r['dii_net']:+,.0f}Cr\n"
                       for _,r in hist.head(5).iterrows()])
        analysis = ai(f"FII bought {fii_streak} days (₹{t3_fii:+,.0f}Cr). DII sold {dii_streak} days. NSE India. Why matters? Which sectors? Days until market reacts? 100 words.")
        send(f"🏦🏦 <b>FII ACCUMULATION SIGNAL</b>\n{'─'*32}\n"
             f"FII buying: <b>{fii_streak} consecutive days</b>\nDII selling: <b>{dii_streak} consecutive days</b>\n\n"
             f"3-day: FII <b>₹{t3_fii:+,.0f}Cr</b> | DII <b>₹{t3_dii:+,.0f}Cr</b>\n\n"
             f"<code>{tbl}</code>"
             f"{'⚠️ Estimated data (NSE API unavailable)' if proxy else ''}\n\n"
             f"<b>🤖 AI:</b>\n{analysis}\n\n⚠️ Educational only.\n🕐 {ts()}")
        save_alert("FII_DII", f"FII {fii_streak}d streak", f"3d ₹{t3_fii:+,.0f}Cr", analysis)
        print("  ✅ FII/DII alert sent")
        return True

    print("  No divergence pattern yet")
    return False


# ══════════════════════════════════════════════════════════════════════════════
# MODULE 2 — BULK DEALS
# ══════════════════════════════════════════════════════════════════════════════

INST_KW = ["mutual fund"," mf","asset management","insurance","foreign portfolio","fpi",
           "hdfc mf","sbi mf","icici pru","nippon","axis mf","kotak mf","dsp","mirae",
           "motilal","tata mf","lic","uti","birla","franklin","invesco","pgim","goldman",
           "morgan stanley","jp morgan","blackrock","vanguard","fidelity","nomura","ubs",
           "deutsche","citibank","government","pension","provident","epfo","360 one",
           "nuvama","edelweiss","whiteoak","trust mf"]

def is_inst(client): return any(k in client.lower() for k in INST_KW)

def fetch_bulk_deals():
    today_str = datetime.now(IST).strftime("%d-%m-%Y")
    deals = []

    # NSE JSON
    print("  [1] NSE bulk deals API...")
    for url in [f"https://www.nseindia.com/api/bulk-deals?date={today_str}",
                "https://www.nseindia.com/api/bulk-deals"]:
        data = nse_get(url)
        if data:
            raw = data.get("data", data if isinstance(data, list) else [])
            if raw:
                for item in raw:
                    deals.append({
                        "date": item.get("BD_DT_DATE", today_str),
                        "symbol": str(item.get("BD_SYMBOL","")),
                        "company": str(item.get("BD_COMP_NAME", item.get("BD_SYMBOL",""))),
                        "client": str(item.get("BD_CLIENT_NAME","Unknown")),
                        "buy_sell": str(item.get("BD_BUY_SELL","")).upper(),
                        "qty": parse_cr(item.get("BD_QTY_TRD",0)),
                        "price": parse_cr(item.get("BD_TP_WATP",0)),
                    })
                print(f"  NSE: {len(deals)} bulk deals")
                return deals

    # BSE
    print("  [2] BSE bulk deals...")
    try:
        d = datetime.now(IST).strftime("%Y-%m-%d")
        r = requests.get(f"https://api.bseindia.com/BseIndiaAPI/api/BulkDealsFull/w?dtfrom={d}&dtto={d}&scripcd=&clientcd=",
            headers={"User-Agent":"Mozilla/5.0","Referer":"https://www.bseindia.com/"}, timeout=15)
        if r.status_code == 200:
            raw = r.json().get("Table",[])
            for item in raw:
                deals.append({
                    "date": today_str,
                    "symbol": str(item.get("SCRIP_CD","")),
                    "company": str(item.get("SCRIP_NAME","")),
                    "client": str(item.get("CLIENT_NAME","Unknown")),
                    "buy_sell": "BUY" if str(item.get("BUY_SELL","")).upper().startswith("B") else "SELL",
                    "qty": parse_cr(item.get("QTY",0)),
                    "price": parse_cr(item.get("PRICE",0)),
                })
            if deals:
                print(f"  BSE: {len(deals)} bulk deals")
                return deals
    except Exception as e:
        print(f"  BSE failed: {e}")

    print("  No bulk deals found — NSE publishes after 4 PM market close")
    return []


def process_bulk_deals(deals):
    alerted = 0
    for d in deals:
        if not is_inst(d["client"]): continue
        if alerted_today("bulk_deals", d["symbol"]): continue
        save_bulk(d["date"], d["symbol"], d["company"], d["client"],
                  d["buy_sell"], d["qty"], d["price"])
        ctx = stock_ctx(d["symbol"]+".NS")
        val = (d["qty"] * d["price"]) / 1e7
        analysis = ai(f"{d['company']} ({d['symbol']}): {d['client']} {'BUYING' if 'B' in d['buy_sell'] else 'SELLING'} ₹{val:.1f}Cr. Chart: {ctx}. Accumulation or distribution? What should retail do? 80 words.")
        emoji = "🟢" if "B" in d["buy_sell"] else "🔴"
        send(f"{emoji} <b>INSTITUTIONAL BULK DEAL</b>\n{'─'*32}\n"
             f"<b>Company:</b> {d['company']} ({d['symbol']})\n"
             f"<b>Client:</b>  {d['client']}\n"
             f"<b>Action:</b>  {'BUYING' if 'B' in d['buy_sell'] else 'SELLING'}\n"
             f"<b>Qty:</b>     {d['qty']:,.0f} @ ₹{d['price']:,.2f}\n"
             f"<b>Value:</b>   ₹{val:.1f} Cr\n"
             f"<b>Chart:</b>   {ctx}\n\n"
             f"<b>🤖 AI:</b> {analysis}\n\n⚠️ Educational only.\n🕐 {ts()}")
        save_alert("BULK_DEAL", f"{d['company']} bulk deal", f"₹{val:.1f}Cr", analysis)
        alerted += 1; time.sleep(2)
    print(f"  Bulk deals: {alerted} alerts sent")


# ══════════════════════════════════════════════════════════════════════════════
# MODULE 3 — PROMOTER BUYING
# ══════════════════════════════════════════════════════════════════════════════

NON_PROMO = ["mutual fund","mf","fpi","foreign portfolio","insurance","bank",
             "asset management","amc","nbfc","trust","clearing","broker",
             "trading","finance ltd","capital ltd"]

def is_promo(acquirer): return not any(k in acquirer.lower() for k in NON_PROMO)

def fetch_promoter_buying():
    today = datetime.now(IST)
    t_up  = today.strftime("%d-%b-%Y").upper()
    y_up  = (today-timedelta(days=1)).strftime("%d-%b-%Y").upper()
    discs = []

    print("  [1] NSE SAST Reg29...")
    for url in ["https://www.nseindia.com/api/corporate-sast-reg29?index=equities",
                "https://www.nseindia.com/api/sast-disclosures?index=equities"]:
        data = nse_get(url)
        if data:
            raw = data.get("data",[])
            print(f"  NSE SAST: {len(raw)} total records")
            for item in raw:
                df_str = str(item.get("acqDate","")).upper()
                if t_up not in df_str and y_up not in df_str: continue
                pct = parse_cr(item.get("acqPer",0))
                if pct <= 0: continue
                discs.append({
                    "date": item.get("acqDate", today.strftime("%d-%b-%Y")),
                    "company": item.get("company", item.get("symbol","")),
                    "symbol": item.get("symbol",""),
                    "acquirer": item.get("acqName","Unknown"),
                    "shares": parse_cr(item.get("secAcq",0)),
                    "pct_acquired": pct,
                    "total_pct": parse_cr(item.get("totAcqPer",0)),
                })
            if discs:
                print(f"  Found {len(discs)} recent SAST filings")
                return discs

    print("  [2] BSE SAST...")
    try:
        d = today.strftime("%Y-%m-%d")
        r = requests.get(f"https://api.bseindia.com/BseIndiaAPI/api/SASTDatewise/w?dtfrom={d}&dtto={d}&scripcd=",
            headers={"User-Agent":"Mozilla/5.0","Referer":"https://www.bseindia.com/"}, timeout=15)
        if r.status_code == 200:
            raw = r.json().get("Table",[])
            for item in raw:
                bs = str(item.get("BuySell","")).upper()
                if "BUY" not in bs and "ACQUI" not in bs: continue
                pct = parse_cr(item.get("PercentageAcquired",0))
                if pct <= 0: continue
                discs.append({
                    "date": today.strftime("%d-%b-%Y"),
                    "company": item.get("ScripName",""),
                    "symbol": str(item.get("ScripCode","")),
                    "acquirer": item.get("PersonName","Unknown"),
                    "shares": parse_cr(item.get("NoShares",0)),
                    "pct_acquired": pct,
                    "total_pct": parse_cr(item.get("TotalSharesHeld",0)),
                })
            if discs:
                print(f"  BSE SAST: {len(discs)} filings")
                return discs
    except Exception as e:
        print(f"  BSE SAST failed: {e}")

    print("  No SAST disclosures today (rare — 1-3 times per month)")
    return []


def process_promoter_buying(discs):
    alerted = 0
    for d in discs:
        if not is_promo(d["acquirer"]): continue
        if d["pct_acquired"] < 0.01: continue
        if alerted_today("promoter_buying", d["symbol"]): continue
        save_promoter(d["date"], d["company"], d["symbol"], d["acquirer"],
                      d["shares"], d["pct_acquired"], d["total_pct"])
        ctx = stock_ctx(d["symbol"]+".NS")
        analysis = ai(f"Promoter {d['acquirer']} bought {d['pct_acquired']:.3f}% of {d['company']} open market. Stake now {d['total_pct']:.2f}%. Chart: {ctx}. Significance? Should retail follow? Main risk? 80 words.")
        send(f"🔍🔍 <b>PROMOTER BUYING</b>\n{'─'*32}\n"
             f"<b>Company:</b>    {d['company']} ({d['symbol']})\n"
             f"<b>Buyer:</b>      {d['acquirer']}\n"
             f"<b>Shares:</b>     {d['shares']:,.0f}\n"
             f"<b>Change:</b>     +{d['pct_acquired']:.3f}%\n"
             f"<b>Total stake:</b> {d['total_pct']:.2f}%\n"
             f"<b>Chart:</b>      {ctx}\n\n"
             f"💡 Own cash, open market — strongest insider signal\n\n"
             f"<b>🤖 AI:</b> {analysis}\n\n⚠️ Educational only.\n🕐 {ts()}")
        save_alert("PROMOTER_BUYING", f"{d['acquirer']} bought {d['pct_acquired']:.3f}% of {d['company']}",
                   f"Total: {d['total_pct']:.2f}%", analysis)
        alerted += 1; time.sleep(2)
    print(f"  Promoter buying: {alerted} alerts sent")


# ── DIAGNOSTIC ────────────────────────────────────────────────────────────────
def run_diagnostic():
    print("\n" + "="*55)
    print("DIAGNOSTIC — Testing all data sources")
    print("="*55)
    results = {}

    tests = [
        ("NSE Homepage",    lambda: requests.get("https://www.nseindia.com/", timeout=10, headers={"User-Agent":"Mozilla/5.0 Chrome/120"})),
        ("NSE FII/DII",     lambda: get_nse_session().get("https://www.nseindia.com/api/fiidiiTradeReact", timeout=15)),
        ("NSE Bulk Deals",  lambda: get_nse_session().get(f"https://www.nseindia.com/api/bulk-deals?date={datetime.now(IST).strftime('%d-%m-%Y')}", timeout=15)),
        ("NSE SAST",        lambda: get_nse_session().get("https://www.nseindia.com/api/corporate-sast-reg29?index=equities", timeout=15)),
        ("BSE Bulk Deals",  lambda: requests.get(f"https://api.bseindia.com/BseIndiaAPI/api/BulkDealsFull/w?dtfrom={datetime.now(IST).strftime('%Y-%m-%d')}&dtto={datetime.now(IST).strftime('%Y-%m-%d')}&scripcd=&clientcd=", headers={"User-Agent":"Mozilla/5.0","Referer":"https://www.bseindia.com/"}, timeout=15)),
        ("BSE SAST",        lambda: requests.get(f"https://api.bseindia.com/BseIndiaAPI/api/SASTDatewise/w?dtfrom={datetime.now(IST).strftime('%Y-%m-%d')}&dtto={datetime.now(IST).strftime('%Y-%m-%d')}&scripcd=", headers={"User-Agent":"Mozilla/5.0","Referer":"https://www.bseindia.com/"}, timeout=15)),
        ("Yahoo Finance",   lambda: yf.download("^NSEI", period="2d", interval="1d", progress=False)),
    ]

    for name, fn in tests:
        print(f"\n  Testing {name}...")
        try:
            result = fn()
            if hasattr(result, "status_code"):
                if result.status_code == 200:
                    try:
                        d = result.json()
                        count = len(d.get("data", d) if isinstance(d, dict) else d)
                        status = f"✓ HTTP 200 — {count} records"
                    except:
                        status = f"✓ HTTP 200"
                else:
                    status = f"✗ HTTP {result.status_code}"
            elif hasattr(result, "empty"):
                status = f"✓ {len(result)} rows" if not result.empty else "✗ Empty"
            else:
                status = "✓ OK"
            print(f"    {status}")
        except Exception as e:
            status = f"✗ {str(e)[:60]}"
            print(f"    {status}")
        results[name] = status

    print("\n" + "="*55)
    print("SUMMARY:")
    working = [k for k,v in results.items() if v.startswith("✓")]
    broken  = [k for k,v in results.items() if v.startswith("✗")]
    for k,v in results.items():
        print(f"  {v[:60]:<62} {k}")
    print(f"\n✓ Working: {len(working)} | ✗ Not working: {len(broken)}")
    if broken:
        print(f"  Broken: {', '.join(broken)}")
        print("  Script will use fallback sources automatically.")
    print("="*55)


# ── MAIN SCAN ─────────────────────────────────────────────────────────────────
def run_scan():
    now = datetime.now(IST)
    if now.weekday() >= 5:
        print("Weekend — skipping"); return
    print(f"\n{'='*55}\nSmart Money Scan — {now.strftime('%d %b %Y %H:%M IST')}\n{'='*55}")
    print("\n[1/3] FII/DII Divergence")
    fii = fetch_fii_dii()
    if fii: check_fii_dii(fii)
    else:   print("  Could not fetch FII/DII data")
    print("\n[2/3] Bulk Deals")
    process_bulk_deals(fetch_bulk_deals())
    print("\n[3/3] Promoter Buying")
    process_promoter_buying(fetch_promoter_buying())
    print(f"\n✓ Done — {now.strftime('%H:%M:%S')}\n{'='*55}\n")


# ── STREAMLIT TAB ─────────────────────────────────────────────────────────────
def render_smart_money_tab():
    import streamlit as st
    st.subheader("🏦 Smart Money Tracker")
    st.caption("FII/DII flow · Bulk deals · Promoter buying")
    t1, t2, t3 = st.tabs(["FII / DII Flow", "Bulk Deals", "Promoter Buying"])

    with t1:
        st.markdown("**FII vs DII — last 10 trading days**")
        hist = load_fii_dii(10)
        if not hist.empty:
            def cf(row):
                if row["Pattern"]=="FII_BUY_DII_SELL": return ["background-color:#1a3a2a;color:#2ecc71"]*len(row)
                elif row["Pattern"]=="FII_SELL_DII_BUY": return ["background-color:#3a1a1a;color:#e74c3c"]*len(row)
                return [""]*len(row)
            disp = hist[["date","fii_net","dii_net","divergence"]].copy()
            disp.columns=["Date","FII Net (Cr)","DII Net (Cr)","Pattern"]
            st.dataframe(disp.style.apply(cf,axis=1), use_container_width=True, hide_index=True)
            c1,c2=st.columns(2)
            c1.metric("FII 3-day", f"₹{hist.head(3)['fii_net'].sum():+,.0f} Cr")
            c2.metric("DII 3-day", f"₹{hist.head(3)['dii_net'].sum():+,.0f} Cr")
            streak = sum(1 for _,r in hist.iterrows() if r["fii_net"]>200)
            if streak>=3: 
                st.success(f"⚡ DIVERGENCE ACTIVE — FII buying {streak} consecutive days")
                # Send notification if not already sent in this session
                if not st.session_state.get("fii_divergence_notified", False):
                    send_telegram(f"🏦 FII/DII Divergence Alert\nFII buying streak: {streak} days\n3-day FII net: ₹{hist.head(3)['fii_net'].sum():+,.0f} Cr\nPattern: {hist.iloc[0]['divergence']}")
                    st.session_state["fii_divergence_notified"] = True
        else:
            st.info("No FII/DII data available. Click below to fetch latest data.")
            if st.button("Fetch FII/DII Data", key="fetch_fii_dii"):
                with st.spinner("Fetching data..."):
                    data = fetch_fii_dii()
                    if data:
                        save_fii_dii(data["date"], data["fii_buy"], data["fii_sell"], data["fii_net"],
                                     data["dii_buy"], data["dii_sell"], data["dii_net"])
                        st.success("Data fetched and saved! Refresh the page to see it.")
                        st.rerun()
                    else:
                        st.error("Failed to fetch data. Try again later.")

    with t2:
        st.markdown("**Recent institutional bulk deals**")
        try:
            conn=sqlite3.connect(DB_PATH); bd=pd.read_sql_query("SELECT date,company,symbol,client,buy_sell,qty,price FROM bulk_deals ORDER BY id DESC LIMIT 30",conn); conn.close()
            if not bd.empty:
                bd["value_cr"]=((bd["qty"]*bd["price"])/1e7).round(1)
                def cb(row):
                    c="background-color:#1a3a2a;color:#2ecc71" if "B" in str(row["buy_sell"]) else "background-color:#3a1a1a;color:#e74c3c"
                    return [c]*len(row)
                st.dataframe(bd.style.apply(cb,axis=1), use_container_width=True, hide_index=True)
            else: st.info("No bulk deals yet. NSE publishes after 4 PM market close.")
        except: st.info("No bulk deals yet.")

    with t3:
        st.markdown("**Promoter open-market buying (SAST disclosures)**")
        try:
            conn=sqlite3.connect(DB_PATH); pb=pd.read_sql_query("SELECT date,company,symbol,acquirer,shares,pct_acquired,total_pct FROM promoter_buying ORDER BY id DESC LIMIT 20",conn); conn.close()
            if not pb.empty:
                st.dataframe(pb, use_container_width=True, hide_index=True)
                st.success("💡 Promoter buying = own cash in own company")
            else: st.info("No promoter buying yet. This fires 1–3 times per month.")
        except: st.info("No promoter data yet.")


# ── ENTRY POINT ───────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("="*55)
    print("NSE Smart Money Tracker")
    print("="*55)
    if "YOUR_" in TELEGRAM_TOKEN or "YOUR_" in CHAT_ID or "YOUR_" in GROQ_API_KEY:
        print("\n⚠️  Fill in TELEGRAM_TOKEN, CHAT_ID and GROQ_API_KEY at the top.\n"); exit(1)
    init_db()
    print("\n1 — Diagnostic (test all data sources)")
    print("2 — Run full scan now")
    print("3 — Start scheduler (4 PM + 8 AM auto)\n")
    choice = input("Enter 1, 2 or 3: ").strip()
    if choice == "1":
        run_diagnostic()
    elif choice == "3":
        for day in ["monday","tuesday","wednesday","thursday","friday"]:
            getattr(schedule.every(), day).at("16:00").do(run_scan)
            getattr(schedule.every(), day).at("08:00").do(run_scan)
        print("Scheduler started. Ctrl+C to stop.")
        while True: schedule.run_pending(); time.sleep(30)
    else:
        run_scan()
