# neat_trader_v2.py
import yfinance as yf
import pandas as pd
import numpy as np
import ta
from sklearn.preprocessing import MinMaxScaler
import neat
import os
import pickle 
import random 

from trader import Trader 
from functionalities import plot_backtest_results, plot_generational_performance

# --- Configuration ---
TICKER = "BTC-USD" 
DATA_PERIOD = "7d" 
DATA_INTERVAL = "1m" 
TRAIN_DAYS = 5
INITIAL_STARTING_CAPITAL = 200.0 
INITIAL_STARTING_HOLDINGS = 0.0 
N_LAGS = 5 
CONFIG_FILE_PATH = "./config-feedforward"
N_GENERATIONS = 50 # Run for at least 50-100 generations
MAX_EXPECTED_CREDIT = INITIAL_STARTING_CAPITAL * 10 
MAX_EXPECTED_HOLDINGS_VALUE = INITIAL_STARTING_CAPITAL * 10
PLOT_BEST_OF_GENERATION_EVERY = 10 
TRADING_FEE_PERCENT = 0.001 
EVAL_WINDOW_SIZE_MINUTES = 24 * 60 * 2 # Evaluate on a 2-day random window from the 5-day training set
                                     # Longer window might give more sell opportunities

_COLUMN_NAMES_RESOLVED_FOR_CURRENT_FETCH = False
COL_OPEN, COL_HIGH, COL_LOW, COL_CLOSE, COL_VOLUME = 'Open', 'High', 'Low', 'Close', 'Volume'

# --- Custom NEAT Reporter ---
class GenerationReporter(neat.reporting.BaseReporter): # (Keep as in previous response)
    def __init__(self, plot_interval, train_data_scaled, train_data_raw, neat_config, initial_capital, trading_fee):
        super().__init__() 
        print("<<<<< GenerationReporter __init__ CALLED >>>>>") 
        self.plot_interval = plot_interval; self.generation_count = 0
        self.train_data_scaled_for_reporter = train_data_scaled
        self.train_data_raw_for_reporter = train_data_raw
        self.neat_config = neat_config; self.initial_capital = initial_capital
        self.trading_fee = trading_fee
        self.best_fitness_overall = -float('inf'); self.best_genome_overall = None
        self.generations_list = []
        self.metrics_history = {
            "Best Fitness": [], "Portfolio Value ($)": [], "Realized PnL ($)": [], "Liquid Credit ($)": [],
            "Total Trades": [], "Buys": [], "Sells": [], "Win Rate (Sells %)": [],
            "Avg Profit/Loss per Sell ($)": [], "Total Fees Paid ($)": [] 
        }

    def start_generation(self, generation):
        print(f"<<<<< GenerationReporter start_generation CALLED for gen {generation} >>>>>")
        self.generation_count = generation

    def end_generation(self, config, population_genomes_dict, species_set):
        self._actual_end_of_generation_logic(config, population_genomes_dict, species_set)

    def end_of_generation(self, config, population_genomes_dict, species_set):
        print(f"<<<<< GenerationReporter end_of_generation CALLED for gen {self.generation_count} >>>>>")
        self._actual_end_of_generation_logic(config, population_genomes_dict, species_set)
        
    def _actual_end_of_generation_logic(self, config, population_genomes_dict, species_set_object):
        print(f"  --- Reporter Logic for Gen {self.generation_count} Starting ---")
        best_genome_this_gen, best_fitness_this_gen = None, -float('inf')
        all_current_genomes = list(population_genomes_dict.values())
        if not all_current_genomes: print(f"  REPORTER: No genomes for gen {self.generation_count}.")
        for g in all_current_genomes: 
            if g.fitness is not None and g.fitness > best_fitness_this_gen:
                best_fitness_this_gen, best_genome_this_gen = g.fitness, g
        
        if best_genome_this_gen and (self.best_genome_overall is None or best_fitness_this_gen > self.best_fitness_overall):
            self.best_fitness_overall, self.best_genome_overall = best_fitness_this_gen, best_genome_this_gen

        self.generations_list.append(self.generation_count)

        if best_genome_this_gen:
            self.metrics_history["Best Fitness"].append(best_fitness_this_gen)
            print(f"    REPORTER: Best of Gen {self.generation_count}: Genome ID {best_genome_this_gen.key}, Fitness: {best_fitness_this_gen:.2f}")
            
            net = neat.nn.FeedForwardNetwork.create(best_genome_this_gen, self.neat_config)
            rep_trader = Trader(self.initial_capital, INITIAL_STARTING_HOLDINGS, trading_fee_percent=self.trading_fee) 
            prof_sells, total_sells, buys, sells_count = 0,0,0,0; total_profit_from_sells = 0.0

            sim_data_scaled = self.train_data_scaled_for_reporter
            sim_data_raw = self.train_data_raw_for_reporter

            for i in range(len(sim_data_scaled)):
                if not rep_trader.is_alive: break
                feat, price, ts = sim_data_scaled[i], sim_data_raw.iloc[i][COL_CLOSE], sim_data_raw.index[i]
                state = rep_trader.get_state_for_nn(price, MAX_EXPECTED_CREDIT, MAX_EXPECTED_HOLDINGS_VALUE)
                nn_in = np.concatenate((feat, state)); action, amount = net.activate(nn_in)
                if action > 0.6: 
                    if rep_trader.buy(amount * rep_trader.credit, price, ts): buys +=1
                elif action < 0.4: 
                    old_gains = rep_trader.realized_gains_this_evaluation
                    if rep_trader.sell(amount * rep_trader.holdings_shares, price, ts):
                        sells_count +=1; total_sells += 1
                        profit_this_trade = rep_trader.realized_gains_this_evaluation - old_gains
                        total_profit_from_sells += profit_this_trade
                        if profit_this_trade > 0: prof_sells +=1
                rep_trader.update_history(ts, price)
            
            final_pf_rep = rep_trader.get_portfolio_value(sim_data_raw.iloc[-1][COL_CLOSE])
            win_rate_rep = (prof_sells / total_sells * 100) if total_sells > 0 else 0.0
            avg_pl_sell = (total_profit_from_sells / total_sells) if total_sells > 0 else 0.0
            
            self.metrics_history["Portfolio Value ($)"].append(final_pf_rep)
            self.metrics_history["Realized PnL ($)"].append(rep_trader.realized_gains_this_evaluation)
            self.metrics_history["Liquid Credit ($)"].append(rep_trader.credit)
            self.metrics_history["Total Trades"].append(len(rep_trader.trade_log))
            self.metrics_history["Buys"].append(buys); self.metrics_history["Sells"].append(sells_count)
            self.metrics_history["Win Rate (Sells %)"].append(win_rate_rep)
            self.metrics_history["Avg Profit/Loss per Sell ($)"].append(avg_pl_sell)
            self.metrics_history["Total Fees Paid ($)"].append(rep_trader.total_fees_paid)

            print(f"      REPORTER Gen {self.generation_count} Best Sim (Full Train Data Results):")
            print(f"          Portfolio: ${final_pf_rep:10.2f} | Credit: ${rep_trader.credit:10.2f} | Realized PnL: ${rep_trader.realized_gains_this_evaluation:8.2f} | Fees Paid: ${rep_trader.total_fees_paid:7.2f}")
            print(f"          Total Trades: {len(rep_trader.trade_log):4d} (Buys: {buys:4d}, Sells: {sells_count:4d})")
            print(f"          Win Rate (Sells): {win_rate_rep:6.2f}% | Avg P/L per Sell: ${avg_pl_sell:8.2f}")
            print(f"          Max Portfolio during sim: ${rep_trader.max_portfolio_value_achieved:10.2f}")

            if self.plot_interval > 0 and (self.generation_count + 1) % self.plot_interval == 0:
                print(f"      REPORTER: Plotting for Gen {self.generation_count} Best on Train Data...")
                run_simulation_and_plot(best_genome_this_gen, self.neat_config,
                                        self.train_data_scaled_for_reporter, self.train_data_raw_for_reporter,
                                        title_prefix=f"Gen {self.generation_count} Best (Train)")
        else: 
            for key in self.metrics_history.keys(): self.metrics_history[key].append(np.nan)
            print(f"  REPORTER: No genome with reportable fitness in gen {self.generation_count}.")
        print(f"  --- Reporter Logic for Gen {self.generation_count} Finished ---\n")

    def post_evaluate(self, config, population_object, species_set_object, best_genome_from_neat): 
        if best_genome_from_neat and best_genome_from_neat.fitness is not None:
            if self.best_genome_overall is None or best_genome_from_neat.fitness > self.best_fitness_overall:
                self.best_fitness_overall = best_genome_from_neat.fitness
                self.best_genome_overall = best_genome_from_neat
    def found_solution(self, config, generation, best): 
        print(f"<<<<< GenerationReporter found_solution CALLED at gen {generation}. Best fitness: {best.fitness if best else 'N/A'} >>>>>")
        if best and best.fitness is not None and (self.best_genome_overall is None or best.fitness > self.best_fitness_overall):
            self.best_fitness_overall, self.best_genome_overall = best.fitness, best
    def info(self, msg): pass 
    def plot_generational_metrics(self): 
        print("<<<<< GenerationReporter plot_generational_metrics CALLED >>>>>")
        if self.generations_list and any(len(v) > 0 for v in self.metrics_history.values()):
            valid_metrics_history = {k: v for k, v in self.metrics_history.items() if not all(np.isnan(val) if isinstance(val, float) else False for val in v)}
            if valid_metrics_history:
                 plot_generational_performance(self.generations_list, valid_metrics_history, title="Best Genome Metrics Per Generation (on Training Data)")
            else: print("No valid generational metrics data to plot (all NaNs).")
        else: print("No generational data accumulated to plot.")

# --- Helper Functions (KEEP AS IS) ---
def resolve_column_names(df_columns, ticker_symbol_str): # (Keep this function as previously corrected)
    global COL_OPEN, COL_HIGH, COL_LOW, COL_CLOSE, COL_VOLUME, _COLUMN_NAMES_RESOLVED_FOR_CURRENT_FETCH
    if _COLUMN_NAMES_RESOLVED_FOR_CURRENT_FETCH: return
    _COL_OPEN_temp, _COL_HIGH_temp, _COL_LOW_temp, _COL_CLOSE_temp, _COL_VOLUME_temp = 'Open', 'High', 'Low', 'Close', 'Volume'
    matched_pattern = False
    if isinstance(df_columns, pd.MultiIndex):
        patterns = [
            { 'name': f"('{ticker_symbol_str}', METRIC_NAME)", 'cols': { 'Open': (ticker_symbol_str, 'Open'), 'High': (ticker_symbol_str, 'High'), 'Low': (ticker_symbol_str, 'Low'), 'Close': (ticker_symbol_str, 'Close'), 'Volume': (ticker_symbol_str, 'Volume')}},
            { 'name': f"(METRIC_NAME, '{ticker_symbol_str}')", 'cols': { 'Open': ('Open', ticker_symbol_str), 'High': ('High', ticker_symbol_str), 'Low': ('Low', ticker_symbol_str), 'Close': ('Close', ticker_symbol_str), 'Volume': ('Volume', ticker_symbol_str)}}
        ]
        for p_info in patterns:
            if all(col_tuple in df_columns for col_tuple in p_info['cols'].values()):
                print(f"INFO: Detected yfinance column structure: {p_info['name']}")
                _COL_OPEN_temp, _COL_HIGH_temp, _COL_LOW_temp, _COL_CLOSE_temp, _COL_VOLUME_temp = (p_info['cols']['Open'], p_info['cols']['High'], p_info['cols']['Low'], p_info['cols']['Close'], p_info['cols']['Volume'])
                matched_pattern = True; break
        if not matched_pattern:
            pot_price_metric = {'Open': ('Price', 'Open'), 'High': ('Price', 'High'), 'Low': ('Price', 'Low'), 'Close': ('Price', 'Close')}
            actual_vol_col = None
            if ('Price', 'Volume') in df_columns: actual_vol_col = ('Price', 'Volume')
            elif ('Volume', '') in df_columns: actual_vol_col = ('Volume', '')
            elif ('Volume', ticker_symbol_str) in df_columns: actual_vol_col = ('Volume', ticker_symbol_str)
            if all(c in df_columns for c in pot_price_metric.values()) and actual_vol_col:
                print("INFO: Detected yfinance column structure: ('Price', METRIC_NAME) with Volume")
                _COL_OPEN_temp, _COL_HIGH_temp, _COL_LOW_temp, _COL_CLOSE_temp, _COL_VOLUME_temp = pot_price_metric['Open'], pot_price_metric['High'], pot_price_metric['Low'], pot_price_metric['Close'], actual_vol_col
                matched_pattern = True
        if not matched_pattern: print(f"WARNING: Unhandled yfinance MultiIndex. Defaulting to simple names: {df_columns.tolist()}")
    elif not all(col_str in df_columns for col_str in [_COL_OPEN_temp, _COL_HIGH_temp, _COL_LOW_temp, _COL_CLOSE_temp, _COL_VOLUME_temp]):
        print(f"WARNING: Standard column names not found in flat DataFrame. Defaulting: {df_columns.tolist()}")
    else: print("INFO: Detected yfinance column structure: Simple names"); matched_pattern = True 
    COL_OPEN, COL_HIGH, COL_LOW, COL_CLOSE, COL_VOLUME = _COL_OPEN_temp, _COL_HIGH_temp, _COL_LOW_temp, _COL_CLOSE_temp, _COL_VOLUME_temp
    _COLUMN_NAMES_RESOLVED_FOR_CURRENT_FETCH = True

def fetch_data(ticker, period, interval): # (Keep as previously corrected)
    global _COLUMN_NAMES_RESOLVED_FOR_CURRENT_FETCH; _COLUMN_NAMES_RESOLVED_FOR_CURRENT_FETCH = False 
    print(f"Fetching data for {ticker} ({period}, {interval})...")
    if isinstance(ticker, str): data = yf.download(ticker, period=period, interval=interval, progress=False, auto_adjust=True)
    else: data = yf.download(ticker, period=period, interval=interval, progress=False, auto_adjust=True, group_by='ticker' if isinstance(ticker, list) and len(ticker) > 1 else None)
    if data.empty: raise ValueError(f"No data: {ticker}")
    actual_ticker_for_resolve = ticker[0] if isinstance(ticker, list) else ticker
    print(f"Original yfinance data (first 5 rows):\n{data.head()}\nOriginal yfinance columns: {data.columns}")
    resolve_column_names(data.columns, actual_ticker_for_resolve)
    required_cols = [COL_OPEN, COL_HIGH, COL_LOW, COL_CLOSE, COL_VOLUME]
    missing = [c for c in required_cols if c not in data.columns]
    if missing: raise KeyError(f"Cols missing for dropna: {missing}. DF cols: {data.columns.tolist()}")
    data.dropna(subset=required_cols, inplace=True); print(f"Data fetched/NA dropped: {data.shape[0]} rows")
    return data

def calculate_technical_indicators(df): # (Keep as previously corrected)
    print("Calculating TAs..."); df_for_ta = df.copy()
    df_ta = ta.add_all_ta_features(df_for_ta, COL_OPEN, COL_HIGH, COL_LOW, COL_CLOSE, COL_VOLUME, True)
    prefs = ['trend_macd', 'trend_macd_signal', 'trend_macd_diff', 'momentum_rsi', 'momentum_stoch_rsi', 
             'volatility_atr', 'trend_ema_fast', 'trend_ema_slow', 'volume_obv', 'others_cr']
    actual_names, simple_names = [], []
    for p in prefs:
        for col_name in df_ta.columns:
            str_col, is_tuple = str(col_name), isinstance(col_name, tuple)
            if (str_col.startswith(p) or (is_tuple and col_name[0] == p)) and col_name not in actual_names:
                actual_names.append(col_name); simple_names.append(p); break
    out_df = pd.DataFrame(index=df_ta.index); out_df['Close'] = df_ta[COL_CLOSE].values
    for act, simp in zip(actual_names, simple_names):
        unique_simp = simp; ctr = 1
        while unique_simp in out_df.columns: unique_simp = f"{simp}_{ctr}"; ctr+=1
        out_df[unique_simp] = df_ta[act].values
    print(f"Features for model: {out_df.columns.tolist()}"); return out_df

def add_lagged_features(df, n_lags=1): # (Keep as previously corrected)
    print(f"Adding {n_lags} lags..."); lagged_df = df.copy()
    for lag in range(1, n_lags + 1):
        shifted = df.shift(lag)
        for col in df.columns: lagged_df[f'{col}_lag{lag}'] = shifted[col]
    lagged_df.dropna(inplace=True); print(f"Shape after lags/dropna: {lagged_df.shape}"); return lagged_df

def split_data_by_days(df, train_days_count): # (Keep as previously corrected)
    print(f"Splitting: {train_days_count} train days..."); df = df.sort_index()
    unique_days = sorted(df.index.normalize().unique())
    if len(unique_days) < train_days_count + 1: 
        print(f"ERROR split_data: Not enough days ({len(unique_days)}) for {train_days_count} train + valid.")
        return None, None 
    split_date_boundary = unique_days[train_days_count - 1]
    train_df = df[df.index.normalize() <= split_date_boundary]
    validation_df = df[df.index.normalize() > split_date_boundary]
    print(f"Train: {train_df.shape}, {train_df.index.min()} to {train_df.index.max()}")
    print(f"Valid: {validation_df.shape}, {validation_df.index.min()} to {validation_df.index.max()}")
    if train_df.empty or validation_df.empty: 
        print("ERROR split_data: Empty train/valid df.")
        return None, None 
    return train_df, validation_df

def normalize_data(train_df, val_df): # (Keep as previously corrected)
    print("Normalizing..."); scaler = MinMaxScaler()
    train_scaled = pd.DataFrame(scaler.fit_transform(train_df), columns=train_df.columns, index=train_df.index)
    val_scaled = pd.DataFrame(scaler.transform(val_df), columns=val_df.columns, index=val_df.index)
    return train_scaled, val_scaled, scaler

train_data_scaled_np_global = None
train_data_raw_global = None 
train_data_indices_global = None 

# --- FITNESS FUNCTION (REVISED AGAIN - Stricter, Clearer Signals) ---
def eval_genomes(genomes, config):
    global train_data_scaled_np_global, train_data_raw_global
    if train_data_scaled_np_global is None: raise RuntimeError("Train data not set.")

    full_train_len = len(train_data_scaled_np_global)
    window_size = EVAL_WINDOW_SIZE_MINUTES 
    if window_size >= full_train_len or window_size <= 0 : # Ensure window is valid and < full_train_len
        window_size = max(1, full_train_len // 2) # Default to half if problematic, ensure it's at least 1
        # print(f"Adjusted EVAL_WINDOW_SIZE to {window_size} due to training data length.")


    for genome_id, genome in genomes:
        net = neat.nn.FeedForwardNetwork.create(genome, config)
        trader = Trader(INITIAL_STARTING_CAPITAL, INITIAL_STARTING_HOLDINGS, trading_fee_percent=TRADING_FEE_PERCENT) 
        genome.fitness = 0.0 
        
        if full_train_len > window_size : 
            start_index = random.randint(0, full_train_len - window_size -1) 
            end_index = start_index + window_size
        else: 
            start_index = 0; end_index = full_train_len
        
        current_eval_scaled_data = train_data_scaled_np_global[start_index : end_index]
        current_eval_raw_data_df = train_data_raw_global.iloc[start_index : end_index]

        if len(current_eval_scaled_data) == 0: 
            genome.fitness = -1e12; continue 

        num_buys, num_sells, num_profitable_sells, num_loss_sells = 0,0,0,0
        
        for i in range(len(current_eval_scaled_data)): 
            if not trader.is_alive: break
            features = current_eval_scaled_data[i]
            price = current_eval_raw_data_df.iloc[i][COL_CLOSE] 
            ts = current_eval_raw_data_df.index[i]
            state = trader.get_state_for_nn(price, MAX_EXPECTED_CREDIT, MAX_EXPECTED_HOLDINGS_VALUE)
            nn_in = np.concatenate((features, state)); action, amount = net.activate(nn_in)
            
            # Action thresholds: 0.6 for buy, 0.4 for sell
            if action > 0.6: 
                if trader.buy(amount * trader.credit, price, ts): num_buys+=1
            elif action < 0.4: 
                old_gains = trader.realized_gains_this_evaluation
                if trader.sell(amount * trader.holdings_shares, price, ts):
                    num_sells+=1
                    profit_this_trade = trader.realized_gains_this_evaluation - old_gains
                    if profit_this_trade > 0: num_profitable_sells+=1
                    elif profit_this_trade < 0: num_loss_sells +=1
            trader.update_history(ts, price)
        
        # --- END-OF-EPISODE FITNESS CALCULATION (Primary part) ---
        final_portfolio = trader.get_portfolio_value(current_eval_raw_data_df.iloc[-1][COL_CLOSE])
        final_credit = trader.credit
        total_trades = num_buys + num_sells
        net_profit = final_portfolio - INITIAL_STARTING_CAPITAL # Profit AFTER fees (fees reduce final_portfolio)
        
        # 1. BASE FITNESS: NET PROFIT (heavily scaled)
        #    If net_profit is negative, this term becomes a large penalty.
        #    If net_profit is positive, it's a large reward.
        #    The multiplier makes this the dominant factor.
        fitness_score = net_profit * 10.0  # Tunable: e.g., $10 profit = 100 fitness points. $10 loss = -100 fitness.

        # 2. CREDIT PRESERVATION / GROWTH (Secondary, but important constraint)
        #    a. Severe penalty if credit is depleted (e.g., less than 20% of initial left)
        if final_credit < (INITIAL_STARTING_CAPITAL * 0.20):
            fitness_score -= INITIAL_STARTING_CAPITAL * 5.0 # Massive penalty, e.g., -1000 for $200 initial
        #    b. Bonus if credit has grown AND overall profitable
        elif net_profit > 0 and final_credit > INITIAL_STARTING_CAPITAL:
            credit_growth_factor = (final_credit - INITIAL_STARTING_CAPITAL) / INITIAL_STARTING_CAPITAL
            fitness_score += credit_growth_factor * INITIAL_STARTING_CAPITAL * 1.5 # Tunable

        # 3. REALIZED PROFIT FROM SELLS (Encourages closing trades)
        #    This is PnL *from completed sell transactions only*.
        #    If realized_gains is positive, it's a good bonus. If negative, it's a penalty.
        fitness_score += trader.realized_gains_this_evaluation * 3.0 # Tunable

        # 4. PENALTIES FOR POOR TRADING HABITS
        #    a. Churning: Too many trades for too little net profit
        max_trades_allowed_for_window = len(current_eval_scaled_data) / 8 # Stricter: max 1 trade per 8 bars
        if total_trades > max_trades_allowed_for_window and net_profit < (INITIAL_STARTING_CAPITAL * 0.01): # <1% profit
            fitness_score -= (total_trades - max_trades_allowed_for_window) * (INITIAL_STARTING_CAPITAL * 0.03) # Penalty scaled by initial cap

        #    b. Inactivity: If very few trades AND no net profit (or loss)
        if total_trades < 2 and net_profit <= 0:
            fitness_score -= INITIAL_STARTING_CAPITAL * 2.5 # Huge penalty for doing nothing and losing/stagnating

        #    c. High Fees: If total fees are a large part of initial capital and agent isn't making much more in profit
        if trader.total_fees_paid > (INITIAL_STARTING_CAPITAL * 0.05): # Fees > 5% of initial capital
            if net_profit < (trader.total_fees_paid * 1.5): # If net profit doesn't significantly outweigh fees
                fitness_score -= trader.total_fees_paid * 2.0 # Penalize the fees paid significantly

        #    d. Holding a losing position at the end / Not selling when should have
        #       This is implicitly handled by net_profit being lower.
        #       If it ends holding shares and num_sells is 0 but num_buys > 0, and net_profit is low/negative:
        if num_buys > 0 and num_sells == 0 and net_profit < (INITIAL_STARTING_CAPITAL * 0.01): # Bought, never sold, little/no profit
             fitness_score -= INITIAL_STARTING_CAPITAL * 1.0 # Penalize just holding

        genome.fitness = fitness_score
        if np.isnan(genome.fitness) or np.isinf(genome.fitness): genome.fitness = -1e12


def run_simulation_and_plot(genome, config, data_scaled_np, data_raw, title_prefix): # Keep as previously corrected
    print(f"\n--- {title_prefix} Evaluation ---")
    net = neat.nn.FeedForwardNetwork.create(genome, config)
    tr = Trader(INITIAL_STARTING_CAPITAL, INITIAL_STARTING_HOLDINGS, trading_fee_percent=TRADING_FEE_PERCENT)
    buys,sells = 0,0; i_sim = 0
    for i_sim_loop in range(len(data_scaled_np)): 
        i_sim = i_sim_loop 
        if not tr.is_alive: break
        feat, price, ts = data_scaled_np[i_sim], data_raw.iloc[i_sim][COL_CLOSE], data_raw.index[i_sim]
        state = tr.get_state_for_nn(price, MAX_EXPECTED_CREDIT, MAX_EXPECTED_HOLDINGS_VALUE)
        nn_in = np.concatenate((feat, state)); action, amount = net.activate(nn_in)
        if action > 0.6: 
            if tr.buy(amount * tr.credit, price, ts): buys+=1
        elif action < 0.4: 
            if tr.sell(amount * tr.holdings_shares, price, ts): sells+=1
        tr.update_history(ts, price)
    last_sim_step_idx = 0
    if len(data_raw) > 0 : 
        last_sim_step_idx = min(i_sim, len(data_raw) - 1) if 'i_sim' in locals() and len(data_scaled_np) > 0 else len(data_raw) -1
        final_val = tr.get_portfolio_value(data_raw.iloc[last_sim_step_idx][COL_CLOSE]) 
    else: final_val = tr.credit 
    print(f"{title_prefix} - Initial: ${INITIAL_STARTING_CAPITAL:.2f}, Final Portfolio: ${final_val:.2f}, Final Credit: ${tr.credit:.2f}")
    print(f"Trades Logged: {len(tr.trade_log)} (Simulation Buys: {buys}, Sells: {sells}), Realized PnL: ${tr.realized_gains_this_evaluation:.2f}, Fees Paid: ${tr.total_fees_paid:.2f}")
    print(f"Profit/Loss (Portfolio): {((final_val - INITIAL_STARTING_CAPITAL) / INITIAL_STARTING_CAPITAL) * 100:.2f}%")
    if len(data_raw) > 0:
        plot_backtest_results(data_raw, tr.trade_log, tr.history, f"{title_prefix} Results for {TICKER}", COL_CLOSE)
    else: print(f"Plot Info: No data available to plot for '{title_prefix}'.")

def run_neat_trader(config_file): # Keep as previously corrected
    global train_data_scaled_np_global, train_data_raw_global 
    raw_df = fetch_data(TICKER, DATA_PERIOD, DATA_INTERVAL)
    feats_df = calculate_technical_indicators(raw_df)
    feats_lags_df = add_lagged_features(feats_df, N_LAGS)
    
    train_feats, val_feats = split_data_by_days(feats_lags_df, TRAIN_DAYS)
    if train_feats is None or val_feats is None: 
        print("ERROR: Data splitting failed. Exiting."); return
        
    train_data_raw_global = raw_df.loc[train_feats.index].copy() 
    val_data_raw = raw_df.loc[val_feats.index].copy()
    
    train_scaled, val_scaled, _ = normalize_data(train_feats, val_feats)
    train_data_scaled_np_global = train_scaled.to_numpy() 
    val_scaled_np = val_scaled.to_numpy()

    num_trader_state_features = 3 
    total_nn_inputs = train_data_scaled_np_global.shape[1] + num_trader_state_features
    print(f"Number of input features for NEAT: {total_nn_inputs}. UPDATE CONFIG FILE!")

    cfg = neat.Config(neat.DefaultGenome, neat.DefaultReproduction, neat.DefaultSpeciesSet, neat.DefaultStagnation, config_file)
    if cfg.genome_config.num_inputs != total_nn_inputs:
        print(f"CONFIG ERROR: num_inputs mismatch! Script: {total_nn_inputs}, Config: {cfg.genome_config.num_inputs}"); return
    
    pop = neat.Population(cfg)
    print("DEBUG: Adding GenerationReporter...") 
    gen_reporter = GenerationReporter(
        plot_interval=PLOT_BEST_OF_GENERATION_EVERY,
        train_data_scaled=train_data_scaled_np_global, 
        train_data_raw=train_data_raw_global,
        neat_config=cfg,
        initial_capital=INITIAL_STARTING_CAPITAL,
        trading_fee=TRADING_FEE_PERCENT 
    )
    pop.add_reporter(gen_reporter) 
    print("DEBUG: Adding StdOutReporter...") 
    pop.add_reporter(neat.StdOutReporter(True)) 
    print("DEBUG: Adding StatisticsReporter...") 
    stats = neat.StatisticsReporter(); pop.add_reporter(stats) 
    
    print("\nStarting NEAT evolution..."); 
    pop.run(eval_genomes, N_GENERATIONS) 
    
    winner_overall = gen_reporter.best_genome_overall 
    if winner_overall is None: 
        print("DEBUG: gen_reporter.best_genome_overall was None. Falling back to population scan.")
        best_fitness = -float('inf'); temp_winner = None
        all_genomes_final_pop = list(pop.population.values()) 
        for g_val_iter in all_genomes_final_pop:
            if g_val_iter.fitness is not None and g_val_iter.fitness > best_fitness:
                best_fitness = g_val_iter.fitness; temp_winner = g_val_iter
        winner_overall = temp_winner
        if winner_overall is None: print("CRITICAL ERROR: No winner genome could be determined from fallback."); return
        else: print(f"DEBUG: Fallback winner found with fitness {getattr(winner_overall, 'fitness', 'N/A')}")

    print(f"\nOverall Best Genome (Fitness: {getattr(winner_overall, 'fitness', 'N/A')}): Genome ID {winner_overall.key if winner_overall else 'N/A'}")
    if winner_overall:
        with open(f"winner_genome_{TICKER}.pkl", "wb") as f: pickle.dump(winner_overall, f)
        print(f"Saved overall best genome to winner_genome_{TICKER}.pkl")

    gen_reporter.plot_generational_metrics() 

    if winner_overall:
        print("\n--- Final Evaluation of Overall Best Genome ---")
        run_simulation_and_plot(winner_overall, cfg, train_data_scaled_np_global, train_data_raw_global, "Overall Best Genome on Training Data")
        run_simulation_and_plot(winner_overall, cfg, val_scaled_np, val_data_raw, "Overall Best Genome on Validation Data")
    else:
        print("No best genome found to evaluate.")

if __name__ == "__main__":
    if not os.path.exists(CONFIG_FILE_PATH): print(f"Config file not found: {CONFIG_FILE_PATH}")
    else: run_neat_trader(CONFIG_FILE_PATH)
