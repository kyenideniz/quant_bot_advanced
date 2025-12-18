from flask import Flask, jsonify, render_template,  request
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
import pytz

# --- CONFIGURATION ---
warnings.simplefilter(action='ignore', category=FutureWarning)
pd.options.mode.chained_assignment = None

app = Flask(__name__)
app.config['JSONIFY_PRETTYPRINT_REGULAR'] = True  # For older Flask versions
try:
    app.json.compact = False                      # For Flask 2.3+ (Newer versions)
except AttributeError:
    pass

# 2. Disable default sorting (Respects our numbering 1, 2, 3...)
app.config['JSON_SORT_KEYS'] = False
app.config['JSONIFY_SORT_KEYS'] = False

# --- FIREBASE SETUP ---
if not firebase_admin._apps:
    if os.path.exists('firebase_key.json'):
        cred = credentials.Certificate('firebase_key.json')
        firebase_admin.initialize_app(cred)
    elif os.environ.get('FIREBASE_CREDENTIALS'):
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
COLLECTION_NAME = "trading_bot"
DOC_NAME = "portfolio_state_advanced"
MACRO_ASSETS = [
    # Major Crypto (Layer 1s & DeFi)
    'BTC-USD', 'ETH-USD', 'SOL-USD', 'BNB-USD', 'XRP-USD', 
    'ADA-USD', 'DOGE-USD', 'AVAX-USD', 'LINK-USD', 'DOT-USD',
    
    # Commodities/Macro
    'GLD', 'SLV', 'USO', 'GDX', 'TLT', 'UUP'
]

# --- 1. CORE MATH & REGIME UTILS ---

def get_atr(df, window=14):
    high, low, close = df['High'], df['Low'], df['Close'].shift(1)
    tr = pd.concat([high-low, (high-close).abs(), (low-close).abs()], axis=1).max(axis=1)
    return tr.rolling(window).mean().iloc[-1]

def get_regime(df, period=16, threshold=0.25):
    """
    Determines the market state using DYNAMIC parameters.
    """
    try:
        # 1. Efficiency Ratio (ER)
        change = df['Close'].diff(period).abs()
        volatility_sum = df['Close'].diff().abs().rolling(window=period).sum()
        
        if volatility_sum.iloc[-1] == 0: return "THE CHANNEL"
        
        er = change / volatility_sum
        current_er = er.iloc[-1]

        # 2. Volatility Ratio (VR)
        high, low = df['High'], df['Low']
        tr = high - low
        atr_short = tr.rolling(5).mean()
        atr_long = tr.rolling(50).mean()
        
        if pd.isna(atr_long.iloc[-1]) or atr_long.iloc[-1] == 0:
            current_vr = 1.0
        else:
            vr = atr_short / atr_long
            current_vr = vr.iloc[-1]

        # 3. Classify
        IS_TRENDING = current_er > threshold
        IS_VIOLENT = current_vr > 1.2

        if IS_TRENDING and not IS_VIOLENT: return "THE GRIND"
        elif IS_TRENDING and IS_VIOLENT: return "THE EXPLOSION"
        elif not IS_TRENDING and not IS_VIOLENT: return "THE CHANNEL"
        else: return "DANGER ZONE"
            
    except Exception as e:
        print(f"⚠️ Regime Calc Error: {e}")
        return "THE CHANNEL"

def optimize_regime_params(df):
    """
    THE MIDDLE GROUND OPTIMIZER (Two-Stage Search)
    """
    close = df['Close']
    
    # STAGE 1: BROAD SEARCH
    broad_periods = [10, 20, 30, 40]
    broad_thresh = [0.20, 0.30, 0.40]
    
    best_broad_score = -np.inf
    best_broad_p = 20
    best_broad_t = 0.30
    
    for p in broad_periods:
        change = close.diff(p).abs()
        vol = close.diff().abs().rolling(p).sum()
        er_series = change / vol
        
        for t in broad_thresh:
            signal = (er_series > t).astype(int).shift(1)
            total_ret = np.sum(close.pct_change().fillna(0) * signal)
            
            if total_ret > best_broad_score:
                best_broad_score = total_ret
                best_broad_p = p
                best_broad_t = t
                
    # STAGE 2: FINE TUNING
    fine_periods = [best_broad_p - 5, best_broad_p, best_broad_p + 5]
    fine_thresh = [best_broad_t - 0.05, best_broad_t, best_broad_t + 0.05]
    
    final_score = -np.inf
    final_p = best_broad_p
    final_t = best_broad_t
    
    for p in fine_periods:
        if p < 5: continue
        change = close.diff(p).abs()
        vol = close.diff().abs().rolling(p).sum()
        er_series = change / vol
        for t in fine_thresh:
            signal = (er_series > t).astype(int).shift(1)
            total_ret = np.sum(close.pct_change().fillna(0) * signal)
            if total_ret > final_score:
                final_score = total_ret
                final_p = p
                final_t = t
                
    return int(final_p), round(final_t, 2)

def get_rsi(df, window=14):
    """Calculates the 14-day RSI."""
    delta = df['Close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=window).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=window).mean()
    rs = gain / loss
    rsi = 100 - (100 / (1 + rs))
    return rsi.iloc[-1]

# --- 2. HELPERS (DB, VIEW, DATA) ---

def get_state():
    if db is None: 
        return {"cash": INITIAL_CAPITAL, "positions": {}, "config": {}, "logs": [], "last_rebalance": "2020-01-01"}
    doc_ref = db.collection(COLLECTION_NAME).document(DOC_NAME)
    try:
        doc = doc_ref.get()
        if doc.exists: return doc.to_dict()
        else:
            initial = {"cash": INITIAL_CAPITAL, "positions": {}, "config": {}, "logs": [], "last_rebalance": "2020-01-01"}
            doc_ref.set(initial)
            return initial
    except: return None

def save_state(state):
    if db is None: return
    if len(state['logs']) > 50: state['logs'] = state['logs'][-50:]
    try: db.collection(COLLECTION_NAME).document(DOC_NAME).set(state)
    except Exception as e: print(f"❌ DB Save Error: {e}")

def log_event(state, msg):
    print(msg)
    state['logs'].append(f"{datetime.now().strftime('%Y-%m-%d %H:%M:%S')} | {msg}")

def retry_download(tickers, period):
    try:
        df = yf.download(tickers, period=period, interval="1d", progress=False, group_by='ticker', auto_adjust=True)
        if df.empty: return None
        return df
    except: return None

def is_trading_hour():
    # Production: Uncomment below
    return True 
    nyc = pytz.timezone('America/New_York')
    now = datetime.now(nyc)
    if now.weekday() >= 5: return False 
    market_open = now.replace(hour=9, minute=30, second=0, microsecond=0)
    market_close = now.replace(hour=16, minute=0, second=0, microsecond=0)
    return market_open <= now <= market_close

def is_crypto(ticker):
    return ticker.endswith('-USD')

# --- VIEW LOGIC (Restored for Dashboard) ---
def get_portfolio_data(state):
    # 1. SETUP DATA FETCHING
    # Get everything we currently own
    held_tickers = [t for t, p in state['positions'].items() if p.get('shares', 0) > 0]
    # Get everything the bot WANTS to trade (The Watchlist)
    watched_tickers = list(state.get('config', {}).keys())
    
    # Combine them (removing duplicates)
    # We also add MACRO_ASSETS just in case, though usually they are in the watchlist/held list
    all_tickers = list(set(held_tickers + watched_tickers + MACRO_ASSETS))
    
    if not all_tickers:
        return {
            "TotalEquity": state['cash'], "MarketValue": 0.0,
            "TotalGainLoss": 0.0, "OverallReturnPct": 0.0,
            "PositionDetails": state['positions']
        }

    # 2. FETCH LIVE PRICES
    # We fetch data for EVERYTHING in our universe
    data = retry_download(all_tickers, "100d")
    
    # 3. CALCULATE METRICS
    total_market_value = 0.0
    total_cost_basis = 0.0
    enhanced_positions = {}

    # We iterate over 'all_tickers' now, not just 'state['positions']'
    for ticker in all_tickers:
        # Get existing position data OR create a default 'Neutral' blob
        pos = state['positions'].get(ticker, {
            'status': 'NEUTRAL', 
            'shares': 0, 
            'entry_price': 0, 
            'highest_price': 0
        })

        # -- Get Price Data --
        try:
            df = data[ticker] if isinstance(data.columns, pd.MultiIndex) else data
            df = df.dropna()
            current_price = df['Close'].iloc[-1]
        except:
            current_price = pos.get('entry_price', 0)
            df = pd.DataFrame() # Empty if failed

        shares = pos.get('shares', 0)
        entry_price = pos.get('entry_price', 0)
        cost_basis = shares * entry_price

        details = {} 

        if shares > 0 and not df.empty:
            # --- ACTIVE POSITION LOGIC (Same as before) ---
            current_value = shares * current_price
            total_market_value += current_value
            total_cost_basis += cost_basis
            
            conf = state['config'].get(ticker, {})
            r_period = conf.get('RegimePeriod', 20)
            r_thresh = conf.get('RegimeThresh', 0.25)
            
            atr = get_atr(df)
            regime = get_regime(df, period=r_period, threshold=r_thresh)
            highest = pos.get('highest_price', current_price)
            
            # Stop Loss Calc
            current_return = (current_price - entry_price) / entry_price
            stop_price = 0.0
            
            if regime == "THE GRIND":
                m = 1.5 if current_return > 0.10 else 3.5
                stop_price = highest - (m * atr)
            elif regime == "THE EXPLOSION":
                if current_return > 0.15:   m = 1.5
                elif current_return > 0.10: m = 2.5
                else:                       m = 5.0
                stop_price = highest - (m * atr)
            elif regime == "THE CHANNEL":
                stop_price = entry_price - (1.5 * atr)
            else:
                stop_price = highest - (2.0 * atr)

            if stop_price < entry_price and current_return > 0.07:
                stop_price = entry_price + (0.1 * atr)
            
            # Dropped Asset Logic for View
            tgt = conf.get('Target', 0.0)
            if tgt == 0.0 and current_return > 0.015:
                 floor = entry_price * 1.002
                 if stop_price < floor: stop_price = floor

            dist_pct = ((current_price - stop_price) / current_price) * 100
            
            details['01. Return %'] = round(((current_value - cost_basis) / cost_basis) * 100, 2)
            details['02. Gain/Loss'] = round(current_value - cost_basis, 2)
            details['03. Current Price'] = round(current_price, 2)
            details['04. Stop Price'] = round(stop_price, 2)
            details['05. Entry Price'] = round(entry_price, 2)
            details['06. Risk Cushion'] = f"{round(dist_pct, 1)}%"
            details['07. Current Atr'] = round(atr, 2)
            details['08. Shares'] = round(shares, 4)
            details['09. Highest Price'] = round(highest, 2)
            details['10. Status'] = 'LONG'
            details['11. Market Regime'] = regime
            details['13. Cost Basis'] = round(cost_basis, 2)

        else:
            # --- WATCHLIST LOGIC (For new assets) ---
            details['01. Status'] = 'NEUTRAL'
            details['02. Shares'] = 0

        enhanced_positions[ticker] = details
        
    enhanced_positions = dict(sorted(enhanced_positions.items(), key=lambda item: item[1].get('01. Return %', -1000), reverse=True))

    total_equity = state['cash'] + total_market_value
    total_pl = total_market_value - total_cost_basis
    overall_ret = ((total_equity / INITIAL_CAPITAL) - 1) * 100

    return {
        "TotalEquity": round(total_equity, 3),
        "MarketValue": round(total_market_value, 3),
        "TotalGainLoss": round(total_pl, 3),
        "OverallReturnPct": round(overall_ret, 2),
        "PositionDetails": enhanced_positions
    }

# --- 3. OPTIMIZATION & LOGIC MODULES ---

def run_hrp(prices):
    returns = prices.pct_change().dropna()
    cov, corr = returns.cov(), returns.corr()
    dist = np.sqrt((1 - corr) / 2)
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
    best_params = (50, 20)
    return best_params

def get_fresh_universe():
    print("🌍 SCRAPING ASSETS...")
    tickers = []
    try:
        url = 'https://en.wikipedia.org/wiki/List_of_S%26P_500_companies'
        headers = {'User-Agent': 'Mozilla/5.0'}
        tables = pd.read_html(requests.get(url, headers=headers).text)
        sp500_df = next(t for t in tables if 'Symbol' in t.columns)
        top_stocks = sp500_df['Symbol'].head(90).tolist()
        tickers.extend([x.replace('.', '-') for x in top_stocks])
    except:
        tickers.extend(['NVDA', 'MSFT', 'AAPL', 'AMZN', 'GOOGL', 'META', 'TSLA'])
    tickers.extend(MACRO_ASSETS)
    return list(set(tickers))

def run_monthly_rebalance(state):
    log_event(state, "⏳ REBALANCE STARTED (Dual-Speed Lookback)...")
    
    universe = get_fresh_universe()
    
    # 1. Split Universe
    crypto_tickers = [t for t in universe if t.endswith('-USD')]
    macro_tickers = [t for t in universe if not t.endswith('-USD')]
    
    # 2. Fetch Data (Dual Lookback)
    data_crypto = retry_download(crypto_tickers, "90d") 
    data_macro = retry_download(macro_tickers, "1y")    
    
    if data_crypto is None and data_macro is None: return state

    # 3. Build Unified Data Dictionary
    combined_data = {}
    def pack_data(source_df, ticker_list):
        if source_df is None or source_df.empty: return
        if isinstance(source_df.columns, pd.MultiIndex):
            for t in ticker_list:
                try: combined_data[t] = source_df[t]
                except KeyError: pass
        else:
            if len(ticker_list) == 1: combined_data[ticker_list[0]] = source_df

    pack_data(data_crypto, crypto_tickers)
    pack_data(data_macro, macro_tickers)

    # 4. ADVANCED SCORING (Sortino for Crypto, Adjusted Sharpe for Macro)
    scores = {}
    
    for ticker, df in combined_data.items():
        try:
            close = df['Close'].dropna()
            if len(close) < 60: continue 

            # A. Calculate Raw Momentum (Return)
            # Crypto: 90 days (from your dual-speed logic)
            # Macro: 1 year (from your dual-speed logic)
            total_return = (close.iloc[-1] / close.iloc[0]) - 1
            
            # B. Get Daily Returns
            daily_rets = close.pct_change().dropna()

            # C. Define "Risk" differently for Crypto vs Macro
            if ticker.endswith('-USD'):
                # === CRYPTO SCORING: SORTINO RATIO ===
                # We only look at NEGATIVE volatility (Downside Deviation).
                # Upside volatility (pumps) is NOT penalized.
                negative_rets = daily_rets[daily_rets < 0]
                
                # Safety: If no negative days, use a tiny number to avoid div/0
                downside_std = negative_rets.std() * np.sqrt(365) if not negative_rets.empty else 0.01
                if downside_std == 0: downside_std = 0.01
                
                # Bonus: We boost the score slightly (1.2x) to account for the
                # higher risk premium required to hold crypto.
                score = (total_return / downside_std) * 1.2
                
            else:
                # === MACRO SCORING: SHARPE RATIO ===
                # For Gold/Stocks, we penalize ALL volatility because
                # stability is the main goal here.
                total_std = daily_rets.std() * np.sqrt(252)
                if total_std == 0: total_std = 0.01
                
                score = (total_return / total_std)

            # Sanity check: If score is valid, save it
            if not np.isnan(score) and not np.isinf(score):
                scores[ticker] = score

        except Exception as e:
            print(f"Skipping {ticker}: {e}")
            continue

    # 5. NATURAL SELECTION (No Quotas)
    candidates = sorted(scores, key=scores.get, reverse=True)[:20]
    
    if not candidates: return state 
    log_event(state, f"🏆 TOP 20 WINNERS: {candidates}")
    

    # 6. Prepare HRP Matrix
    all_closes = []
    for t in candidates:
        if t in combined_data:
            series = combined_data[t]['Close']
            series.name = t
            all_closes.append(series)
    
    if not all_closes: return state
    price_matrix = pd.concat(all_closes, axis=1)
    price_matrix = price_matrix.tail(90).dropna()

    # 7. Run HRP & Allocation
    raw_weights = run_hrp(price_matrix)
    final_portfolio = {k: v for k, v in raw_weights.items() if v >= 0.025}
    
    total_w = sum(final_portfolio.values())
    if total_w > 0: final_portfolio = {k: v/total_w for k, v in final_portfolio.items()}
    
    # 8. Generate New Config (The Winners)
    new_config = {}
    for ticker, weight in final_portfolio.items():
        t_df = combined_data[ticker]
        entry, exit = optimize_params(t_df)
        reg_period, reg_thresh = optimize_regime_params(t_df)
        
        default_entry = 20 if ticker.endswith('-USD') else 55
        
        new_config[ticker] = {
            'Target': round(weight, 3),
            'Entry': default_entry,
            'Exit': exit,
            'RegimePeriod': reg_period,
            'RegimeThresh': reg_thresh
        }
    
    # --- 9. SAFETY NET: PRESERVE HELD ASSETS ---
    # If we own it, we MUST keep watching it, even if it's not in the top 20.
    held_tickers = [t for t, p in state['positions'].items() if p.get('shares', 0) > 0]
    
    for t in held_tickers:
        if t not in new_config:
            # It dropped out of the top list, but we are holding it.
            # We add it back with Target = 0.0 (Stop Buying, Just Monitor).
            
            # Fetch data if we don't have it already
            if t in combined_data:
                t_df = combined_data[t]
            else:
                t_df = retry_download(t, "100d")
            
            # Fallback if download fails: keep old config
            if t_df is None or t_df.empty:
                old_conf = state['config'].get(t, {})
                new_config[t] = old_conf
                new_config[t]['Target'] = 0.0 # Force target to 0
                continue

            # Optimize params so we have fresh Stop Loss logic
            entry, exit = optimize_params(t_df)
            reg_period, reg_thresh = optimize_regime_params(t_df)
            default_entry = 20 if t.endswith('-USD') else 55
            
            new_config[t] = {
                'Target': 0.0, # 0% Target means "Sell only" mode
                'Entry': default_entry,
                'Exit': exit,
                'RegimePeriod': reg_period,
                'RegimeThresh': reg_thresh
            }
            log_event(state, f"🛡️ RETAINING {t} (Held Position - Monitoring for Exit)")

    state['config'] = new_config
    state['last_rebalance'] = datetime.now().strftime("%Y-%m-%d")
    log_event(state, f"✅ REBALANCE COMPLETE. Watching {len(new_config)} assets.")
    return state

# --- MAIN LOGIC ---
def run_main_logic(force_run=False):
    # 1. SETUP TIME & SCOPE
    nyc = pytz.timezone('America/New_York')
    now = datetime.now(nyc)
    
    # We define "Market Close" as the 3 PM (15:00) hour in New York.
    # Stocks/Gold are only processed during this hour to avoid intraday noise.
    # Crypto is processed 24/7 (every time this function runs).
    is_market_close_window = (now.hour == 15) or force_run
    
    state = get_state()
    if not state: return "DB Error"

    # 2. WEEKLY REBALANCE TRIGGER
    # Checks if 7 days have passed since the last rebalance
    last_reb_str = state.get('last_rebalance', '2020-01-01')
    try:
        last_reb = datetime.strptime(last_reb_str, "%Y-%m-%d")
    except ValueError:
        # Handle cases where timestamp might be included
        last_reb = datetime.strptime(last_reb_str.split(" ")[0], "%Y-%m-%d")

    days_since_rebalance = (datetime.now() - last_reb).days
    
    if days_since_rebalance >= 7:
        log_event(state, f"🔄 TRIGGERING WEEKLY REBALANCE (Last: {days_since_rebalance} days ago)")
        # This calls your new 'Dual-Speed' rebalance function
        state = run_monthly_rebalance(state) 
    
    # 3. DEFINE ACTION LIST (The "Hourly" Filter)
    active_tickers = list(state.get('config', {}).keys())
    held_tickers = list(state.get('positions', {}).keys())
    all_monitored_tickers = list(set(active_tickers + held_tickers))
    
    tickers_to_process = []
    
    for t in all_monitored_tickers:
        if is_crypto(t):
            # Always process Crypto (24/7)
            tickers_to_process.append(t)
        elif is_market_close_window:
            # Only process Stocks/Gold/Bond at 3 PM ET
            tickers_to_process.append(t)
            
    if not tickers_to_process:
        # If it's 10 AM and we only hold Gold, we do nothing.
        return "Skipping Macro Assets (Not Market Close)"

    # 4. CALCULATE EQUITY FOR SIZING
    total_equity = state['cash']
    for ticker in held_tickers:
        pos = state['positions'][ticker]
        if pos.get('shares', 0) > 0:
            total_equity += (pos['shares'] * pos['entry_price'])

    # 5. MAIN TRADING LOOP
    for ticker in tickers_to_process:
        # Download data (Crypto gets 100d, Macro gets 100d for indicators)
        df = retry_download(ticker, "100d")
        if df is None or len(df) < 60: continue
        if isinstance(df.columns, pd.MultiIndex): df.columns = df.columns.get_level_values(1)
        
        price = df['Close'].iloc[-1]
        current_atr = get_atr(df, window=14)
        current_rsi = get_rsi(df)
        
        pos = state['positions'].get(ticker, {'status': 'NEUTRAL', 'shares': 0, 'entry_price': 0, 'highest_price': 0})
        existing_conf = state['config'].get(ticker, {})

        # --- DYNAMIC PARAMETERS ---
        # Crypto uses fast (20d) entry by default. Macro uses slow (55d).
        default_entry = 20 if is_crypto(ticker) else 55
        
        params = {
            'Entry': existing_conf.get('Entry', default_entry),
            'Exit': existing_conf.get('Exit', 20),
            'Target': existing_conf.get('Target', 0.0),
            'RegimePeriod': existing_conf.get('RegimePeriod', 16),
            'RegimeThresh': existing_conf.get('RegimeThresh', 0.25)
        }

        # Update Highest Price (for Trailing Stops)
        if pos['status'] == 'LONG':
            if 'highest_price' not in pos or pos['highest_price'] == 0: 
                pos['highest_price'] = pos['entry_price']
            if price > pos['highest_price']:
                pos['highest_price'] = price
                state['positions'][ticker] = pos

        # Market Regime & Breakout Levels
        market_regime = get_regime(df, period=params['RegimePeriod'], threshold=params['RegimeThresh'])
        
        # Calculate TWO breakout levels
        hist_high_slow = df['High'].iloc[:-1].rolling(55).max().iloc[-1] # Safe (Turtle 2)
        hist_high_fast = df['High'].iloc[:-1].rolling(20).max().iloc[-1] # Aggressive (Turtle 1)
        
        signal = "HOLD"
        exit_reason = ""

        # === BUY / RE-ENTRY SIGNALS ===
        if pos['status'] == "NEUTRAL" and ticker in active_tickers:
            
            # A. BREAKOUT LOGIC
            # If Crypto, we ALWAYS take the 20-day breakout.
            if is_crypto(ticker) and price > hist_high_fast:
                signal = "BUY"
                reason = "Crypto Fast Breakout (20d)"
            
            # If Macro, we take 20-day ONLY if regime is EXPLOSION.
            elif not is_crypto(ticker) and market_regime == "THE EXPLOSION" and price > hist_high_fast:
                 signal = "BUY"
                 reason = "Explosive Macro Breakout (20d)"
            
            # Fallback: Standard 55-day Breakout for Macro
            elif price > hist_high_slow:
                signal = "BUY"
                reason = "Standard Breakout (55d)"
            
            # B. DIP RE-ENTRY LOGIC
            # Fixed variable name bug here (market_regime)
            elif market_regime in ["THE EXPLOSION", "THE GRIND"]:
                # Wider dip tolerance for crypto (up to 15% drop from high)
                lower_limit = 0.85 if is_crypto(ticker) else 0.92
                pullback_zone = (price < hist_high_fast * 0.98) and (price > hist_high_fast * lower_limit)
                
                # RSI Filter: >30 ensures we don't catch a falling knife
                if pullback_zone and current_rsi > 30:
                    signal = "BUY"
                    reason = f"Confirmed Dip (RSI: {current_rsi:.1f})"

        # === SELL / PROFIT-TAKING SIGNALS ===
        elif pos['status'] == "LONG":
            highest = pos.get('highest_price', price)
            entry = pos.get('entry_price', price)
            current_return = (price - entry) / entry
            
            stop_price = 0.0

            # 1. THE EXPLOSION
            if market_regime == "THE EXPLOSION":
                if current_return > 0.15:   m = 1.5
                elif current_return > 0.10: m = 2.5
                else:                       m = 5.0
                stop_price = highest - (m * current_atr)
                exit_reason = f"Explosion Trail ({m}x ATR)"

            # 2. THE GRIND
            elif market_regime == "THE GRIND":
                m = 1.5 if current_return > 0.10 else 3.5
                stop_price = highest - (m * current_atr)
                exit_reason = f"Grind Trail ({m}x ATR)"

            # 3. THE CHANNEL
            elif market_regime == "THE CHANNEL":
                stop_price = entry - (1.5 * current_atr)
                exit_reason = "Channel Stop"

            # 4. DANGER ZONE
            else: 
                stop_price = highest - (2.0 * current_atr)
                exit_reason = "Danger Zone Tight Stop"

            # 5. BREAKEVEN GUARD (>7% Profit) - Standard for Active Assets
            if stop_price < entry and current_return > 0.07:
                stop_price = entry + (0.1 * current_atr)
                exit_reason = "Breakeven Guard (>7%)"

            # --- 6. DROPPED ASSET PROTOCOL (SMART GUARD) ---
            # Only applies if Target is 0 (Asset was rejected by rebalance)
            if params.get('Target', 0.0) == 0.0:
                # Only enforce floor if we have > 3% profit cushion
                if current_return > 0.03: 
                    breakeven_floor = entry * 1.002
                    # Ensure stop never drops below breakeven
                    if stop_price < breakeven_floor:
                        stop_price = breakeven_floor
                        exit_reason = "Dropped Asset Guard (Floor)"

            if price < stop_price:
                signal = "SELL"

        # === EXECUTION ===
        if signal == "BUY" and state['cash'] > 0:
            target_val = total_equity * params.get('Target', 0.0)
            
            # Minimum trade size check ($500) to avoid dust
            if target_val > 500:
                shares = target_val / price
                cost = shares * price * 1.001 # 0.1% slippage/fee buffer
                
                if cost < state['cash']:
                    state['cash'] -= cost
                    state['positions'][ticker] = {
                        'status': 'LONG', 
                        'shares': shares, 
                        'entry_price': price,
                        'highest_price': price, 
                        'atr_at_entry': current_atr
                    }
                    log_event(state, f"🚀 BUY {ticker} | Price: ${price:.2f} | Reason: {reason}")

        elif signal == "SELL" and pos['shares'] > 0:
            shares = pos['shares']
            buy_price = pos['entry_price']
            revenue = shares * price * 0.999
            
            dollar_profit = revenue - (shares * buy_price)
            pct_profit = (dollar_profit / (shares * buy_price)) * 100
            
            state['cash'] += revenue
            state['positions'][ticker] = {'status': 'NEUTRAL', 'shares': 0, 'entry_price': 0, 'highest_price': 0}
            
            log_msg = (f"💰 CLOSED {ticker} | "
                       f"Profit: ${dollar_profit:+.2f} ({pct_profit:+.2f}%) | "
                       f"Reason: {exit_reason}")
            log_event(state, log_msg)
            
    save_state(state)
    return f"Logic Executed. Processed: {tickers_to_process}"

# --- 4. FLASK ROUTES ---
# --- GLOBAL CACHE STORAGE ---
# This lives in the server's RAM. It resets if the server restarts.
PORTFOLIO_CACHE = {
    "data": None,
    "timestamp": None
}

@app.route('/')
def home():
    global PORTFOLIO_CACHE
    
    # 1. Check if we have valid cached data (less than 5 minutes old)
    now = datetime.now()
    is_cache_valid = False
    
    if PORTFOLIO_CACHE["data"] and PORTFOLIO_CACHE["timestamp"]:
        age = (now - PORTFOLIO_CACHE["timestamp"]).total_seconds()
        if age < 300:  # 300 seconds = 5 minutes
            is_cache_valid = True
            print(f"⚡ USING CACHE (Age: {age:.0f}s)")

    # 2. If Cache is VALID, use it.
    if is_cache_valid:
        final_data = PORTFOLIO_CACHE["data"]
        
    # 3. If Cache is EXPIRED, fetch fresh data (The "Slow" Part)
    else:
        print("⏳ CACHE EXPIRED. Fetching fresh data...")
        base_state = get_state()
        if not base_state: 
            return jsonify({"status": "error", "msg": "DB Error"})
        
        # Get live market data
        perf_data = get_portfolio_data(base_state)
        
        # Prepare the output dictionary
        output = base_state.copy()
        output['positions'] = perf_data['PositionDetails']

        # Calc ROI
        invested_capital = 100000 - output['cash']
        return_on_bought = 0.0
        if invested_capital > 0:
            return_on_bought = perf_data['TotalGainLoss'] * 100 / invested_capital

        final_data = {
            "PORTFOLIO_SUMMARY": {
                "1. Overall Return (%)": perf_data['OverallReturnPct'],
                "2. Return on Bought Assets (%)" : round(return_on_bought, 2),
                "3. Total Equity": perf_data['TotalEquity'],
                "4. Total P&L": perf_data['TotalGainLoss'],
                "5. Market Value": perf_data['MarketValue'],
                "6. Cash": round(output['cash'], 2)
            },
            **output
        }
        
        # SAVE to Cache
        PORTFOLIO_CACHE["data"] = final_data
        PORTFOLIO_CACHE["timestamp"] = now

    # 4. Render the page
    return render_template('dashboard.html', data=final_data)
    
@app.route('/run')
def execute():
    # Check if the URL has ?force=true
    force = request.args.get('force') == 'true'
    try: 
        # Pass the force flag to your logic
        return jsonify({"status": "success", "msg": run_main_logic(force_run=force)})
    except Exception as e: 
        return jsonify({"status": "error", "msg": str(e)})
    
@app.route('/force_rebalance')
def force_rebalance():
    try:
        state = get_state()
        if not state: return jsonify({"status": "error", "msg": "DB Load Failed"})
        
        # Manually run the rebalance logic
        state = run_monthly_rebalance(state)
        
        # Save immediately
        save_state(state)
        
        return jsonify({
            "status": "success", 
            "msg": f"Forced Rebalance Complete. Portfolio now watching {len(state['config'])} assets."
        })
    except Exception as e:
        return jsonify({"status": "error", "msg": str(e)})
        
if __name__ == '__main__':
    app.run(debug=True, port=5001)
