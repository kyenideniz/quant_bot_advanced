from flask import Flask, jsonify
import yfinance as yf
import pandas as pd
import numpy as np
import scipy.cluster.hierarchy as sch
import firebase_admin
from firebase_admin import credentials, firestore
import os
import json
import warnings
import requests
from datetime import datetime

# --- CONFIGURATION ---
warnings.simplefilter(action='ignore', category=FutureWarning)
pd.options.mode.chained_assignment = None

app = Flask(__name__)
app.config['JSONIFY_PRETTYPRINT_REGULAR'] = True

# --- FIREBASE ---
if not firebase_admin._apps:
    # 1. Try to load from local file (Best for local testing)
    if os.path.exists('firebase_key.json'):
        print("🔥 Connecting to Firebase via Local Key file...")
        cred = credentials.Certificate('firebase_key.json')
        firebase_admin.initialize_app(cred)
        
    # 2. Try to load from Environment Variable (Best for Vercel/Cloud)
    elif os.environ.get('FIREBASE_CREDENTIALS'):
        print("☁️ Connecting to Firebase via Environment Variable...")
        cred_dict = json.loads(os.environ.get('FIREBASE_CREDENTIALS'))
        cred = credentials.Certificate(cred_dict)
        firebase_admin.initialize_app(cred)
    else:
        print("⚠️ No Firebase keys found. Database will be disabled.")

try:
    db = firestore.client()
except:
    db = None

# --- CONSTANTS ---
INITIAL_CAPITAL = 100000.0   
RISK_PER_TRADE = 0.02
COMMISSION_RATE = 0.001 
COLLECTION_NAME = "trading_bot"
DOC_NAME = "portfolio_state_advanced"

# --- HELPER FUNCTIONS ---
# --- HELPER FUNCTIONS (Cloud Only) ---

def get_state():
    # If DB connection failed, we can't do anything. Return default state but don't save locally.
    if db is None: 
        print("⚠️ DB Connection missing. Using temporary RAM state.")
        return {
            "cash": INITIAL_CAPITAL,
            "positions": {},
            "config": {},
            "logs": [],
            "last_rebalance": "2020-01-01" 
        }

    doc_ref = db.collection(COLLECTION_NAME).document(DOC_NAME)
    try:
        doc = doc_ref.get()
        if doc.exists:
            state = doc.to_dict()
            # Ensure config dict exists to prevent key errors
            if "config" not in state: state["config"] = {}
            return state
        else:
            # First time run: Create the document in Cloud DB
            initial_state = {
                "cash": INITIAL_CAPITAL,
                "positions": {},
                "config": {},
                "logs": [],
                "last_rebalance": "2020-01-01" 
            }
            doc_ref.set(initial_state)
            return initial_state
    except Exception as e:
        print(f"⚠️ Error reading from Firebase: {e}")
        return None

def save_state(state):
    if db is None: return

    # Limit logs to keep DB clean
    if len(state['logs']) > 50: state['logs'] = state['logs'][-50:]

    try:
        db.collection(COLLECTION_NAME).document(DOC_NAME).set(state)
        print("✅ State saved to Firebase.")
    except Exception as e:
        print(f"❌ Failed to save to Firebase: {e}")

def log_event(state, msg):
    print(msg) 
    state['logs'].append(f"{datetime.now().strftime('%Y-%m-%d %H:%M:%S')} | {msg}")

def retry_download(tickers, period):
    try:
        # Group_by ticker ensures consistent format
        df = yf.download(tickers, period=period, interval="1d", progress=False, group_by='ticker', auto_adjust=True)
        if df.empty: return None
        return df
    except: return None

def is_trading_hour():
    # Production: Uncomment lines below
    #  return True
    import pytz
    nyc = pytz.timezone('America/New_York')
    now = datetime.now(nyc)
    if now.weekday() >= 5: return False 
    market_open = now.replace(hour=9, minute=30, second=0, microsecond=0)
    market_close = now.replace(hour=16, minute=0, second=0, microsecond=0)
    return market_open <= now <= market_close

# --- 🧠 DYNAMIC UNIVERSE GENERATOR ---

def get_fresh_universe():
    """
    Scrapes the internet for the current top assets.
    """
    print("🌍 SCRAPING GLOBAL ASSETS...")
    tickers = []
    
    # 1. Scrape S&P 500 from Wikipedia (Robust Version)
    try:
        url = 'https://en.wikipedia.org/wiki/List_of_S%26P_500_companies'
        # Fake a browser to avoid 403 blocks
        headers = {'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7)'}
        r = requests.get(url, headers=headers)
        
        # Parse all tables
        tables = pd.read_html(r.text)
        
        sp500_df = None
        
        # 🔍 SMART SEARCH: Find the table that actually has the "Symbol" column
        for t in tables:
            if 'Symbol' in t.columns:
                sp500_df = t
                break
        
        if sp500_df is None:
            raise ValueError("Found tables, but none had the 'Symbol' column.")
        
        # Take the first 90 symbols
        top_90_stocks = sp500_df['Symbol'].head(90).tolist()
        
        # Clean ticker symbols (e.g. BRK.B -> BRK-B)
        top_90_stocks = [x.replace('.', '-') for x in top_90_stocks]
        tickers.extend(top_90_stocks)
        
        print(f"✅ Wiki Success: Found {len(top_90_stocks)} stocks.")
        
    except Exception as e:
        print(f"⚠️ Wiki Scrape Failed: {e}. Using Emergency Backup.")
        tickers.extend(['NVDA', 'MSFT', 'AAPL', 'AMZN', 'GOOGL', 'META', 'TSLA', 'JPM', 'LLY', 'AVGO'])

    # 2. Add Fixed Macro Assets (Crypto + Commodities)
    macro_assets = [
        'BTC-USD', 'ETH-USD', 'SOL-USD', # Crypto
        'GLD', 'SLV', 'USO', 'GDX',      # Commodities
        'TLT', 'UUP'                     # Bonds/Dollar
    ]
    tickers.extend(macro_assets)
    
    # Remove duplicates and return
    return list(set(tickers))

# --- 🧠 HRP & OPTIMIZATION MATH ---

def run_hrp(prices):
    returns = prices.pct_change().dropna()
    cov, corr = returns.cov(), returns.corr()
    dist = np.sqrt((1 - corr) / 2)
    
    # Handle clustering edge cases
    if len(corr) < 2: return {prices.columns[0]: 1.0}
    
    link = sch.linkage(dist, 'single')
    
    def get_quasi_diag(link):
        link = link.astype(int)
        sort_ix = pd.Series([link[-1, 0], link[-1, 1]])
        num_items = link[-1, 3]
        while sort_ix.max() >= num_items:
            sort_ix.index = range(0, sort_ix.shape[0] * 2, 2)
            df0 = sort_ix[sort_ix >= num_items]
            i = df0.index; j = df0.values - num_items
            sort_ix[i] = link[j, 0]
            df0 = pd.Series(link[j, 1], index=i + 1)
            sort_ix = pd.concat([sort_ix, df0])
            sort_ix = sort_ix.sort_index()
            sort_ix.index = range(sort_ix.shape[0])
        return sort_ix.tolist()

    sort_ix = get_quasi_diag(link)
    sort_ix = corr.index[sort_ix].tolist()
    
    def get_rec_bisection(cov, sort_ix):
        w = pd.Series(1, index=sort_ix)
        c_items = [sort_ix]
        while len(c_items) > 0:
            c_items = [i[j:k] for i in c_items for j, k in ((0, len(i) // 2), (len(i) // 2, len(i))) if len(i) > 1]
            for i in range(0, len(c_items), 2):
                c0, c1 = c_items[i], c_items[i + 1]
                var0 = np.dot(np.dot(np.ones((1,len(c0))), cov.loc[c0,c0]), np.ones((len(c0),1)))[0,0] / len(c0)**2
                var1 = np.dot(np.dot(np.ones((1,len(c1))), cov.loc[c1,c1]), np.ones((len(c1),1)))[0,0] / len(c1)**2
                alpha = 1 - var0 / (var0 + var1)
                w[c0] *= alpha; w[c1] *= 1 - alpha
        return w

    return get_rec_bisection(cov, sort_ix).to_dict()

def optimize_params(df):
    # 1. Setup Data
    # Ensure we are working with simple 1D arrays
    close_vals = df['Close'].values
    high_vals = df['High'].values
    low_vals = df['Low'].values
    
    # Calculate returns for the whole period (fill NaN with 0 to avoid crashes)
    returns = df['Close'].pct_change().fillna(0).values

    best_score = -np.inf
    best_params = (55, 20) # Default fallback

    # 2. Define Ranges (Matching your new script)
    entry_range = range(20, 70, 5)
    exit_range = range(10, 40, 5)

    # 3. Brute Force Grid Search
    for entry in entry_range:
        for exit in exit_range:
            if exit >= entry: continue

            # Pre-calculate Rolling indicators
            # (We shift by 1 because we make decisions based on YESTERDAY's high/low)
            r_high = pd.Series(high_vals).rolling(entry).max().shift(1).fillna(0).values
            r_low = pd.Series(low_vals).rolling(exit).min().shift(1).fillna(0).values

            # Simulation Loop
            pos = np.zeros(len(close_vals))
            in_pos = 0 # 0 = Neutral, 1 = Long
            
            # We iterate through the days to simulate holding
            # Start after the lookback period
            for i in range(entry + 1, len(close_vals)):
                price = close_vals[i]
                breakout = r_high[i]
                breakdown = r_low[i]
                
                if in_pos == 0:
                    if price >= breakout:
                        in_pos = 1
                elif in_pos == 1:
                    if price <= breakdown:
                        in_pos = 0
                    else:
                        in_pos = 1 # Stay long
                
                pos[i] = in_pos

            # Calculate Performance
            # We shift position array to align: "Position yesterday determines return today"
            # pos[:-1] are positions from day 0 to N-1
            # returns[1:] are returns from day 1 to N
            strat_rets = returns[1:] * pos[:-1]
            
            # Compounding Return (The "Reality Check" metric)
            total_ret = np.nancumprod(1 + strat_rets)[-1]

            if total_ret > best_score:
                best_score = total_ret
                best_params = (entry, exit)

    return best_params

# --- 🚀 MONTHLY MANAGER (Rebalance) ---

def run_monthly_rebalance(state):
    log_event(state, "⏳ MONTHLY REBALANCE STARTED...")
    
    # 1. Get Dynamic Universe (~100 Assets)
    universe = get_fresh_universe()
    log_event(state, f"🔍 Scanning {len(universe)} Assets...")
    
    # 2. Bulk Download (6 Months)
    data = retry_download(universe, "1y")
    if data is None: return state

    # 3. Momentum Score & Rank
    scores = {}
    for ticker in universe:
        try:
            if isinstance(data.columns, pd.MultiIndex):
                if ticker not in data.columns.levels[0]: continue
                df = data[ticker]
            else:
                if ticker != universe[0]: continue
                df = data
            
            close = df['Close'].dropna()
            if len(close) < 100: continue
            
            # Momentum / Volatility
            mom = (close.iloc[-1] / close.iloc[0]) - 1
            vol = close.pct_change().std() * np.sqrt(126)
            if vol > 0: scores[ticker] = mom / vol
        except: continue

    # 4. Pick Top 12 Candidates
    candidates = sorted(scores, key=scores.get, reverse=True)[:12]
    log_event(state, f"💎 Top 12 Candidates: {candidates}")
    
    # 5. Run HRP Clustering
    if isinstance(data.columns, pd.MultiIndex):
        price_matrix = data.xs('Close', level=1, axis=1)[candidates].dropna()
    else:
        price_matrix = data['Close']
    
    raw_weights = run_hrp(price_matrix)
    
    # 6. Filter & Optimize
    new_config = {}
    
    # Drop anything < 5% allocation (Focus capital)
    final_portfolio = {k: v for k, v in raw_weights.items() if v >= 0.05}
    
    # Re-normalize weights to 100%
    total_w = sum(final_portfolio.values())
    if total_w > 0:
        final_portfolio = {k: v/total_w for k, v in final_portfolio.items()}
    
    # Find Entry/Exit for the finalists
    for ticker, weight in final_portfolio.items():
        t_df = data[ticker] if isinstance(data.columns, pd.MultiIndex) else data
        entry, exit = optimize_params(t_df)
        
        new_config[ticker] = {
            'Target': round(weight, 3),
            'Entry': entry,
            'Exit': exit
        }
        
    state['config'] = new_config
    state['last_rebalance'] = datetime.now().strftime("%Y-%m-%d")
    log_event(state, f"✅ NEW PORTFOLIO: {list(new_config.keys())}")
    return state

# --- ⚙️ DAILY WORKER (Trade) ---

def run_main_logic():
    if not is_trading_hour(): return "Market Closed"

    state = get_state()
    if not state: return "DB Error"

    # 2. Rebalance Check (New Month?)
    last_reb_str = state.get('last_rebalance', '2020-01-01')
    last_reb = datetime.strptime(last_reb_str, "%Y-%m-%d")
    if datetime.now().month != last_reb.month:
        state = run_monthly_rebalance(state)

    # 3. Equity Calc
    total_equity = state['cash']
    active_tickers = list(state.get('config', {}).keys())
    
    if not active_tickers:
        state = run_monthly_rebalance(state)
        active_tickers = list(state.get('config', {}).keys())

    for ticker in active_tickers:
        pos = state['positions'].get(ticker, {'status': 'NEUTRAL', 'shares': 0, 'entry_price': 0})
        state['positions'][ticker] = pos
        if pos['shares'] > 0:
            total_equity += (pos['shares'] * pos['entry_price'])

    # 4. Trading Loop
    for ticker in active_tickers:
        df = retry_download(ticker, "200d")
        if df is None: continue
        if isinstance(df.columns, pd.MultiIndex): df.columns = df.columns.get_level_values(1)
        
        price = df['Close'].iloc[-1]
        params = state['config'][ticker]
        
        pos = state['positions'][ticker]
        signal = "HOLD"
        
        # Real-time equity update
        if pos['shares'] > 0:
            total_equity -= (pos['shares'] * pos['entry_price'])
            total_equity += (pos['shares'] * price)

        hist_high = df['High'].iloc[:-1].rolling(params['Entry']).max().iloc[-1]
        hist_low = df['Low'].iloc[:-1].rolling(params['Exit']).min().iloc[-1]
        
        if pos['status'] == "NEUTRAL" and price > hist_high: signal = "BUY"
        elif pos['status'] == "LONG" and price < hist_low: signal = "SELL"

        target_val = total_equity * params['Target']
        
        if signal == "BUY" and state['cash'] > 0:
            shares = target_val / price
            cost = shares * price * (1.001)
            if cost < state['cash'] and shares > 0:
                state['cash'] -= cost
                state['positions'][ticker] = {'status': 'LONG', 'shares': shares, 'entry_price': price}
                log_event(state, f"BUY {ticker} (Target: {params['Target']*100:.1f}%)")

        elif signal == "SELL":
            rev = pos['shares'] * price * (0.999)
            state['cash'] += rev
            state['positions'][ticker] = {'status': 'NEUTRAL', 'shares': 0, 'entry_price': 0}
            log_event(state, f"SELL {ticker}")

    save_state(state)
    return "Logic Executed"

# Indicators
def get_atr(df, window=14):
    high, low, close = df['High'], df['Low'], df['Close'].shift(1)
    tr = pd.concat([high-low, (high-close).abs(), (low-close).abs()], axis=1).max(axis=1)
    return tr.rolling(window).mean().iloc[-1]

@app.route('/')
def home():
    try: return jsonify(get_state())
    except: return jsonify({"status":"error"})

@app.route('/run')
def execute():
    try: return jsonify({"status": "success", "msg": run_main_logic()})
    except Exception as e: return jsonify({"status": "error", "msg": str(e)})

# --- FIXED CODE ---
if __name__ == '__main__':
    # Use port 5001 to avoid conflict with macOS AirPlay
    app.run(debug=True, port=5001)
