from flask import Flask, jsonify, render_template
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
    'BTC-USD', 'ETH-USD', 'SOL-USD',  # Crypto
    'GLD', 'SLV', 'USO', 'GDX',       # Commodities
    'TLT', 'UUP'                      # Bonds/Dollar
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

# --- VIEW LOGIC (Restored for Dashboard) ---
def get_portfolio_data(state):
    # 1. Setup Data Fetching
    held_tickers = [t for t, p in state['positions'].items() if p.get('shares', 0) > 0]
    tickers_to_fetch = list(set(held_tickers + MACRO_ASSETS))
    
    if not tickers_to_fetch:
        return {
            "TotalEquity": state['cash'], "MarketValue": 0.0,
            "TotalGainLoss": 0.0, "OverallReturnPct": 0.0,
            "PositionDetails": state['positions']
        }

    # 2. Fetch Live Prices & History
    data = retry_download(tickers_to_fetch, "100d")
    
    # 3. Calculate Metrics & Format Positions
    total_market_value = 0.0
    total_cost_basis = 0.0
    enhanced_positions = {}

    for ticker, pos in state['positions'].items():
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
            current_value = shares * current_price
            total_market_value += current_value
            total_cost_basis += cost_basis
            
            # --- CALCULATE METRICS ---
            conf = state['config'].get(ticker, {})
            r_period = conf.get('RegimePeriod', 20)
            r_thresh = conf.get('RegimeThresh', 0.25)
            
            atr = get_atr(df)
            regime = get_regime(df, period=r_period, threshold=r_thresh)
            highest = pos.get('highest_price', current_price)
            
            # Calculate Stop Level
            stop_price = 0.0
            if regime == "THE GRIND":      stop_price = highest - (3.5 * atr)
            elif regime == "THE EXPLOSION": stop_price = highest - (5.0 * atr)
            elif regime == "THE CHANNEL":   stop_price = entry_price - (1.5 * atr)
            elif regime == "DANGER ZONE":   stop_price = highest - (2.0 * atr)
            
            # Breakeven Visualization (Optional: Does not affect logic, just view)
            # Inside get_portfolio_data(), replace the stop_price calculation:

            # Calculate Stop Level for Dashboard Display
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

            dist_pct = ((current_price - stop_price) / current_price) * 100
            
            # --- POPULATE DASHBOARD KEYS ---
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
            
            # --- NEW: ENTRY PRICE ---
            details['13. Cost Basis'] = round(cost_basis, 2)

        else:
            details['01. Status'] = 'NEUTRAL'
            details['02. Shares'] = 0
            
        enhanced_positions[ticker] = details

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
    log_event(state, "⏳ MONTHLY REBALANCE & REGIME TUNING STARTED...")
    
    universe = get_fresh_universe()
    data = retry_download(universe, "1y")
    if data is None: return state

    # Momentum Rank
    scores = {}
    for ticker in universe:
        try:
            df = data[ticker] if isinstance(data.columns, pd.MultiIndex) else data
            close = df['Close'].dropna()
            if len(close) < 100: continue
            mom = (close.iloc[-1] / close.iloc[0]) - 1
            vol = close.pct_change().std() * np.sqrt(126)
            if vol > 0: scores[ticker] = mom / vol
        except: continue

    candidates = sorted(scores, key=scores.get, reverse=True)[:12]
    
    # HRP Weights
    price_matrix = data.xs('Close', level=1, axis=1)[candidates].dropna() if isinstance(data.columns, pd.MultiIndex) else data['Close']
    raw_weights = run_hrp(price_matrix)
    final_portfolio = {k: v for k, v in raw_weights.items() if v >= 0.05}
    total_w = sum(final_portfolio.values())
    if total_w > 0: final_portfolio = {k: v/total_w for k, v in final_portfolio.items()}
    
    new_config = {}
    for ticker, weight in final_portfolio.items():
        t_df = data[ticker] if isinstance(data.columns, pd.MultiIndex) else data
        entry, exit = optimize_params(t_df)
        
        # Optimize Regime Params (Middle Ground Search)
        reg_period, reg_thresh = optimize_regime_params(t_df)
        
        new_config[ticker] = {
            'Target': round(weight, 3),
            'Entry': entry,
            'Exit': exit,
            'RegimePeriod': reg_period,
            'RegimeThresh': reg_thresh
        }
    
    state['config'] = new_config
    state['last_rebalance'] = datetime.now().strftime("%Y-%m-%d")
    log_event(state, f"✅ NEW PORTFOLIO: {list(new_config.keys())}")
    return state

def run_main_logic():
    if not is_trading_hour(): return "Market Closed"
    state = get_state()
    if not state: return "DB Error"

    active_tickers = list(state.get('config', {}).keys())
    held_tickers = list(state.get('positions', {}).keys())
    all_monitored_tickers = list(set(active_tickers + held_tickers))

    # --- Monthly Rebalance Trigger ---
    last_reb = datetime.strptime(state.get('last_rebalance', '2020-01-01'), "%Y-%m-%d")
    if datetime.now().month != last_reb.month:
        state = run_monthly_rebalance(state)
        active_tickers = list(state.get('config', {}).keys())
        all_monitored_tickers = list(set(active_tickers + held_tickers))

    # Calculate Total Equity for Sizing
    total_equity = state['cash']
    for ticker in held_tickers:
        pos = state['positions'][ticker]
        if pos.get('shares', 0) > 0:
            # Note: Using entry_price for a rough equity estimate; 
            # Production could fetch live price here for better precision.
            total_equity += (pos['shares'] * pos['entry_price'])

    for ticker in all_monitored_tickers:
        df = retry_download(ticker, "100d")
        if df is None or len(df) < 60: continue
        if isinstance(df.columns, pd.MultiIndex): df.columns = df.columns.get_level_values(1)
        
        price = df['Close'].iloc[-1]
        current_atr = get_atr(df, window=14)
        pos = state['positions'].get(ticker, {'status': 'NEUTRAL', 'shares': 0, 'entry_price': 0, 'highest_price': 0})
        
        existing_conf = state['config'].get(ticker, {})
        params = {
            'Entry': existing_conf.get('Entry', 55),
            'Exit': existing_conf.get('Exit', 20),
            'Target': existing_conf.get('Target', 0.0),
            'RegimePeriod': existing_conf.get('RegimePeriod', 16),
            'RegimeThresh': existing_conf.get('RegimeThresh', 0.25)
        }

        # Update Highest Price for Trailing Stops
        if pos['status'] == 'LONG':
            if 'highest_price' not in pos or pos['highest_price'] == 0: 
                pos['highest_price'] = pos['entry_price']
            if price > pos['highest_price']:
                pos['highest_price'] = price
                state['positions'][ticker] = pos

        market_regime = get_regime(df, period=params['RegimePeriod'], threshold=params['RegimeThresh'])
        hist_high = df['High'].iloc[:-1].rolling(params['Entry']).max().iloc[-1]
        
        signal = "HOLD"
        exit_reason = ""

        # === 1. BUY / RE-ENTRY SIGNALS ===
        # Fetch RSI for confirmation
        current_rsi = get_rsi(df)

        # BUY SIGNALS (Updated with RSI Filter)
        if pos['status'] == "NEUTRAL" and ticker in active_tickers:
            # A: Standard Breakout (No RSI needed, price is at a new 55-day high)
            if price > hist_high: 
                signal = "BUY"
                reason = "Breakout"
            
            # B: Dip Re-entry (Requires RSI Confirmation)
            elif market_regime in ["THE EXPLOSION", "THE GRIND"]:
                pullback_zone = (price < hist_high * 0.98) and (price > hist_high * 0.92)
                
                # RSI Filter: Ensure we aren't buying while momentum is still crashing
                # We want the 'Dip' but we want to see RSI showing signs of stabilization (e.g., > 30)
                if pullback_zone and current_rsi > 35:
                    signal = "BUY"
                    reason = f"Confirmed Dip (RSI: {current_rsi:.1f})"
                    
        # === 2. SELL / PROFIT-TAKING SIGNALS ===
        elif pos['status'] == "LONG":
            highest = pos.get('highest_price', price)
            entry = pos.get('entry_price', price)
            current_return = (price - entry) / entry
            
            # --- DYNAMIC MULTIPLIER (Lock in Silver Profit) ---
            if market_regime == "THE EXPLOSION":
                if current_return > 0.15:   m = 1.5 # Secure 15%+ gains
                elif current_return > 0.10: m = 2.5 # Secure 10% gains
                else:                       m = 5.0 # Default loose
                stop_price = highest - (m * current_atr)
                exit_reason = f"Explosion Trail ({m}x ATR)"

            elif market_regime == "THE GRIND":
                m = 1.5 if current_return > 0.10 else 3.5
                stop_price = highest - (m * current_atr)
                exit_reason = f"Grind Trail ({m}x ATR)"

            elif market_regime == "THE CHANNEL":
                stop_price = entry - (1.5 * current_atr)
                exit_reason = "Channel Stop"

            else: # DANGER ZONE / FALLBACK
                stop_price = highest - (2.0 * current_atr)
                exit_reason = "Danger Zone Tight Stop"

            # Breakeven Ratchet (>7% Profit Rule)
            if stop_price < entry and current_return > 0.07:
                stop_price = entry + (0.1 * current_atr)
                exit_reason = "Breakeven Guard (>7%)"

            if price < stop_price:
                signal = "SELL"

        # === 3. EXECUTION ===
        if signal == "BUY" and state['cash'] > 0:
            target_val = total_equity * params.get('Target', 0.0)
            shares = target_val / price
            cost = shares * price * 1.001
            
            if cost < state['cash'] and shares > 0:
                state['cash'] -= cost
                state['positions'][ticker] = {
                    'status': 'LONG', 
                    'shares': shares, 
                    'entry_price': price,
                    'highest_price': price, 
                    'atr_at_entry': current_atr
                }
                # Updated BUY log to show entry price
                log_event(state, f"🚀 BUY {ticker} | Price: ${price:.2f} | Shares: {shares:.2f}")

        elif signal == "SELL" and pos['shares'] > 0:
            # Calculate financials before closing
            shares = pos['shares']
            buy_price = pos['entry_price']
            sell_price = price
            
            revenue = shares * sell_price * 0.999
            cost_basis = shares * buy_price
            
            # Profit Calculations
            dollar_profit = revenue - cost_basis
            pct_profit = (dollar_profit / cost_basis) * 100
            
            state['cash'] += revenue
            state['positions'][ticker] = {'status': 'NEUTRAL', 'shares': 0, 'entry_price': 0, 'highest_price': 0}
            
            # Updated SELL log with detailed P&L
            log_msg = (f"💰 CLOSED {ticker} | "
                       f"Buy: ${buy_price:.2f} | "
                       f"Sell: ${sell_price:.2f} | "
                       f"Profit: ${dollar_profit:+.2f} ({pct_profit:+.2f}%) | "
                       f"Reason: {exit_reason}")
            log_event(state, log_msg)
            
    save_state(state)
    return "Logic Executed"

# --- 4. FLASK ROUTES ---

@app.route('/')
def home():
    base_state = get_state()
    if not base_state: 
        return jsonify({"status": "error", "msg": "DB Error"})
    
    # 1. Get Portfolio Data (Same as before)
    perf_data = get_portfolio_data(base_state)
    
    # 2. Prepare Base Output
    output = base_state.copy()
    output['positions'] = perf_data['PositionDetails']

    # Handle division by zero if cash is exactly 100k
    invested_capital = 100000 - output['cash']
    return_on_bought = 0.0
    if invested_capital > 0:
        return_on_bought = perf_data['TotalGainLoss'] * 100 / invested_capital

    # 3. Create the Summary Dictionary
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
    
    # 4. Render the HTML Dashboard
    # Instead of returning jsonify(final_data), we pass final_data to the template
    return render_template('dashboard.html', data=final_data)

@app.route('/run')
def execute():
    try: return jsonify({"status": "success", "msg": run_main_logic()})
    except Exception as e: return jsonify({"status": "error", "msg": str(e)})

if __name__ == '__main__':
    app.run(debug=True, port=5001)
