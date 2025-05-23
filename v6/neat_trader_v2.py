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
import copy

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
N_GENERATIONS = 50
MAX_EXPECTED_CREDIT = INITIAL_STARTING_CAPITAL * 10 
MAX_EXPECTED_HOLDINGS_VALUE = INITIAL_STARTING_CAPITAL * 10
PLOT_BEST_OF_GENERATION_EVERY = 10 
TRADING_FEE_PERCENT = 0.001 
EVAL_WINDOW_SIZE_MINUTES = 18 * 60 

_COLUMN_NAMES_RESOLVED_FOR_CURRENT_FETCH = False
COL_OPEN, COL_HIGH, COL_LOW, COL_CLOSE, COL_VOLUME = 'Open', 'High', 'Low', 'Close', 'Volume'
current_eval_window_start_index = 0 
max_portfolio_ever_achieved_in_training = INITIAL_STARTING_CAPITAL 
best_record_breaker_details = {
    "genome_obj": None, "window_fitness": -float('inf'), "portfolio_achieved_on_full_train": -float('inf')
}

# --- Custom NEAT Reporter ---
class GenerationReporter(neat.reporting.BaseReporter):
    def __init__(self, plot_interval, train_data_scaled, train_data_raw, neat_config, initial_capital, trading_fee):
        super().__init__() 
        self.plot_interval = plot_interval; self.generation_count = 0
        self.train_data_scaled_for_reporter = train_data_scaled
        self.train_data_raw_for_reporter = train_data_raw 
        self.neat_config = neat_config; self.initial_capital = initial_capital
        self.trading_fee = trading_fee
        self.neat_best_fitness_so_far = -float('inf') 
        self.neat_overall_best_genome_obj = None 
        self.generations_list = []
        # CORRECTED and EXPANDED Keys
        self.metrics_history = {
            "Best Fitness (Window)": [], 
            "Max Portfolio Ever ($) (Training Record)": [], # KeyError Fix: Added this key
            "Best Gen Genome's Portfolio ($) (Full Train Sim)": [],
            "Best Gen Genome's Net Profit ($) (Full Train Sim)": [],
            "Best Gen Genome's Credit ($) (Full Train Sim)": [],      
            "Best Gen Genome's Total Trades (Full Train Sim)": [],   
            "Best Gen Genome's Buys (Full Train Sim)": [],          
            "Best Gen Genome's Sells (Full Train Sim)": [],         
            "Best Gen Genome's Fees Paid ($) (Full Train Sim)": [] 
        }

    def start_generation(self, generation): # (Keep as previously corrected for window date printing)
        global current_eval_window_start_index 
        self.generation_count = generation
        full_train_len = len(self.train_data_scaled_for_reporter) 
        window_size_for_eval = EVAL_WINDOW_SIZE_MINUTES
        if window_size_for_eval >= full_train_len or window_size_for_eval <=10 :
             window_size_for_eval = max(10, full_train_len // 3) 
        if full_train_len > window_size_for_eval:
            if N_GENERATIONS > 1: 
                total_advanceable_range = full_train_len - window_size_for_eval
                advance_per_generation = max(1, total_advanceable_range // (N_GENERATIONS -1)) if N_GENERATIONS > 1 and total_advanceable_range > 0 else 0
                current_eval_window_start_index = generation * advance_per_generation
            else: current_eval_window_start_index = 0
            current_eval_window_start_index = min(current_eval_window_start_index, total_advanceable_range -1 if total_advanceable_range >0 else 0)
            current_eval_window_start_index = max(0, current_eval_window_start_index) 
        else: current_eval_window_start_index = 0 
        eval_win_start_idx = current_eval_window_start_index
        eval_win_end_idx = min(eval_win_start_idx + window_size_for_eval, full_train_len)
        if len(self.train_data_raw_for_reporter) > eval_win_start_idx and len(self.train_data_raw_for_reporter) >= eval_win_end_idx and eval_win_end_idx > eval_win_start_idx:
            start_date_str = self.train_data_raw_for_reporter.index[eval_win_start_idx].strftime('%Y-%m-%d %H:%M')
            end_date_str = self.train_data_raw_for_reporter.index[eval_win_end_idx-1].strftime('%Y-%m-%d %H:%M')
            print(f"  Gen {generation}: eval_genomes window: idx {eval_win_start_idx}-{eval_win_end_idx-1} ({start_date_str} to {end_date_str}), size={eval_win_end_idx - eval_win_start_idx}")

    def end_generation(self, config, population_genomes_dict, species_set):
        self._actual_end_of_generation_logic(config, population_genomes_dict, species_set)
        
    def _actual_end_of_generation_logic(self, config, population_genomes_dict, species_set_object):
        global max_portfolio_ever_achieved_in_training, best_record_breaker_details 
        best_genome_this_gen_by_window_fitness, current_gen_max_window_fitness = None, -float('inf')
        all_current_genomes = list(population_genomes_dict.values())
        for g in all_current_genomes: 
            if g.fitness is not None and g.fitness > current_gen_max_window_fitness:
                current_gen_max_window_fitness, best_genome_this_gen_by_window_fitness = g.fitness, g
        
        if best_genome_this_gen_by_window_fitness and (current_gen_max_window_fitness > self.neat_best_fitness_so_far): 
            self.neat_best_fitness_so_far = current_gen_max_window_fitness
            self.neat_overall_best_genome_obj = best_genome_this_gen_by_window_fitness 
            print(f"  REPORTER: ** New best NEAT genome (by window fitness)! ** Gen: {self.generation_count}, ID: {best_genome_this_gen_by_window_fitness.key}, Fitness: {current_gen_max_window_fitness:.2f}")

        self.generations_list.append(self.generation_count)
        self.metrics_history["Max Portfolio Ever ($) (Training Record)"].append(max_portfolio_ever_achieved_in_training) # Log current global max

        if best_genome_this_gen_by_window_fitness:
            self.metrics_history["Best Fitness (Window)"].append(current_gen_max_window_fitness)
            net = neat.nn.FeedForwardNetwork.create(best_genome_this_gen_by_window_fitness, self.neat_config)
            rep_trader = Trader(self.initial_capital, INITIAL_STARTING_HOLDINGS, trading_fee_percent=self.trading_fee) 
            buys_rep_sim, sells_rep_sim = 0,0 
            for i in range(len(self.train_data_scaled_for_reporter)): 
                if not rep_trader.is_alive: break
                feat, price, ts = self.train_data_scaled_for_reporter[i], self.train_data_raw_for_reporter.iloc[i][COL_CLOSE], self.train_data_raw_for_reporter.index[i]
                state = rep_trader.get_state_for_nn(price, MAX_EXPECTED_CREDIT, MAX_EXPECTED_HOLDINGS_VALUE)
                nn_in = np.concatenate((feat, state)); action, amount = net.activate(nn_in)
                if action > 0.6: 
                    if rep_trader.buy(amount * rep_trader.credit, price, ts): buys_rep_sim +=1
                elif action < 0.4: 
                    if rep_trader.sell(amount * rep_trader.holdings_shares, price, ts): sells_rep_sim +=1
                rep_trader.update_history(ts, price) 
            final_pf_rep = rep_trader.get_portfolio_value(self.train_data_raw_for_reporter.iloc[-1][COL_CLOSE])
            net_profit_rep = final_pf_rep - self.initial_capital
            
            if rep_trader.max_portfolio_value_achieved > max_portfolio_ever_achieved_in_training:
                max_portfolio_ever_achieved_in_training = rep_trader.max_portfolio_value_achieved
                best_record_breaker_details["genome_obj"] = copy.deepcopy(best_genome_this_gen_by_window_fitness) 
                best_record_breaker_details["window_fitness"] = current_gen_max_window_fitness 
                best_record_breaker_details["portfolio_achieved_on_full_train"] = rep_trader.max_portfolio_value_achieved
                print(f"    REPORTER: !!! NEW GLOBAL MAX PORTFOLIO (on Full Train Sim): ${max_portfolio_ever_achieved_in_training:.2f} "
                      f"by Gen {self.generation_count}'s Best (ID {best_genome_this_gen_by_window_fitness.key}, WindowFit {current_gen_max_window_fitness:.2f}) !!!")

            self.metrics_history["Best Gen Genome's Portfolio ($) (Full Train Sim)"].append(final_pf_rep)
            self.metrics_history["Best Gen Genome's Net Profit ($) (Full Train Sim)"].append(net_profit_rep)
            self.metrics_history["Best Gen Genome's Credit ($) (Full Train Sim)"].append(rep_trader.credit)
            self.metrics_history["Best Gen Genome's Total Trades (Full Train Sim)"].append(len(rep_trader.trade_log))
            self.metrics_history["Best Gen Genome's Buys (Full Train Sim)"].append(buys_rep_sim)
            self.metrics_history["Best Gen Genome's Sells (Full Train Sim)"].append(sells_rep_sim)
            self.metrics_history["Best Gen Genome's Fees Paid ($) (Full Train Sim)"].append(rep_trader.total_fees_paid)
            
            print(f"    REPORTER Gen {self.generation_count} Best (ID {best_genome_this_gen_by_window_fitness.key}, WindowFit: {current_gen_max_window_fitness:.2f}) "
                  f"SimOnFullTrain: NetProfit: ${net_profit_rep:.2f}, Pf: ${final_pf_rep:.2f}, Cr: ${rep_trader.credit:.2f}, "
                  f"Trades: {len(rep_trader.trade_log)} (B:{buys_rep_sim}/S:{sells_rep_sim}), Fees: ${rep_trader.total_fees_paid:.2f}")
            # print(f"      Current Global Max Portfolio Record: ${max_portfolio_ever_achieved_in_training:.2f}") # Redundant if new record printed

            if self.plot_interval > 0 and (self.generation_count + 1) % self.plot_interval == 0:
                print(f"      REPORTER: Plotting for Gen {self.generation_count} Best on Train Data...")
                run_simulation_and_plot(best_genome_this_gen_by_window_fitness, self.neat_config,
                                        self.train_data_scaled_for_reporter, self.train_data_raw_for_reporter,
                                        title_prefix=f"Gen {self.generation_count} Best (Full Train Sim)")
        else: 
            for key in self.metrics_history.keys(): 
                if key == "Max Portfolio Ever ($) (Training Record)":
                    self.metrics_history[key].append(max_portfolio_ever_achieved_in_training) 
                else:
                    self.metrics_history[key].append(np.nan)
            print(f"  REPORTER: No genome with reportable fitness in gen {self.generation_count}.")

    def post_evaluate(self, config, population_object, species_set_object, best_genome_from_neat): 
        if best_genome_from_neat and best_genome_from_neat.fitness is not None:
            if self.neat_overall_best_genome_obj is None or best_genome_from_neat.fitness > self.neat_best_fitness_so_far:
                self.neat_best_fitness_so_far = best_genome_from_neat.fitness
                self.neat_overall_best_genome_obj = best_genome_from_neat 
    def found_solution(self, config, generation, best_found_by_neat): 
        print(f"<<<<< GenerationReporter found_solution CALLED at gen {generation}. Best fitness by NEAT: {best_found_by_neat.fitness if best_found_by_neat else 'N/A'} >>>>>")
        if best_found_by_neat and best_found_by_neat.fitness is not None and \
           (self.neat_overall_best_genome_obj is None or best_found_by_neat.fitness > self.neat_best_fitness_so_far):
            self.neat_best_fitness_so_far = best_found_by_neat.fitness
            self.neat_overall_best_genome_obj = best_found_by_neat
    def info(self, msg): pass 
    def plot_generational_metrics(self): 
        print("<<<<< GenerationReporter plot_generational_metrics CALLED >>>>>")
        metrics_to_plot = {k: v for k, v in self.metrics_history.items() if "Full Train Sim" in k or "Best Fitness" in k or "Max Portfolio Ever" in k}
        if self.generations_list and any(len(v) > 0 for v in metrics_to_plot.values()):
            valid_metrics_history = {k: v for k, v in metrics_to_plot.items() if not all(np.isnan(val) if isinstance(val, float) else False for val in v)}
            if valid_metrics_history:
                 plot_generational_performance(self.generations_list, valid_metrics_history, title="Key Metrics Per Generation")
            else: print("No valid generational metrics data to plot (all NaNs).")
        else: print("No generational data accumulated to plot.")


# --- Helper Functions (KEEP AS IS) ---
# ... (resolve_column_names, fetch_data, calculate_technical_indicators, add_lagged_features, split_data_by_days, normalize_data) ...
def resolve_column_names(df_columns, ticker_symbol_str): 
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

def fetch_data(ticker, period, interval): 
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

def calculate_technical_indicators(df): 
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

def add_lagged_features(df, n_lags=1): 
    print(f"Adding {n_lags} lags..."); lagged_df = df.copy()
    for lag in range(1, n_lags + 1):
        shifted = df.shift(lag)
        for col in df.columns: lagged_df[f'{col}_lag{lag}'] = shifted[col]
    lagged_df.dropna(inplace=True); print(f"Shape after lags/dropna: {lagged_df.shape}"); return lagged_df

def split_data_by_days(df, train_days_count): 
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

def normalize_data(train_df, val_df): 
    print("Normalizing..."); scaler = MinMaxScaler()
    train_scaled = pd.DataFrame(scaler.fit_transform(train_df), columns=train_df.columns, index=train_df.index)
    val_scaled = pd.DataFrame(scaler.transform(val_df), columns=val_df.columns, index=val_df.index)
    return train_scaled, val_scaled, scaler

train_data_scaled_np_global = None
train_data_raw_global = None 
train_data_indices_global = None 

# --- FITNESS FUNCTION (REVISED "High-Stakes Profit Pursuit") ---
def eval_genomes(genomes, config):
    global train_data_scaled_np_global, train_data_raw_global, current_eval_window_start_index, \
           max_portfolio_ever_achieved_in_training # Access global record

    if train_data_scaled_np_global is None: raise RuntimeError("Train data not set.")

    full_train_len = len(train_data_scaled_np_global)
    window_size_for_eval = EVAL_WINDOW_SIZE_MINUTES 
    if window_size_for_eval >= full_train_len or window_size_for_eval <= 10 : 
        window_size_for_eval = max(10, full_train_len // 2) 
    
    start_idx_for_this_gen_eval = current_eval_window_start_index
    end_idx_for_this_gen_eval = min(start_idx_for_this_gen_eval + window_size_for_eval, full_train_len)
    if (end_idx_for_this_gen_eval - start_idx_for_this_gen_eval) < min(10, window_size_for_eval / 2) :
        if full_train_len > window_size_for_eval :
             start_idx_for_this_gen_eval = max(0, full_train_len - window_size_for_eval -1)
        else: start_idx_for_this_gen_eval = 0
        end_idx_for_this_gen_eval = min(start_idx_for_this_gen_eval + window_size_for_eval, full_train_len)

    current_eval_scaled_data = train_data_scaled_np_global[start_idx_for_this_gen_eval : end_idx_for_this_gen_eval]
    current_eval_raw_data_df = train_data_raw_global.iloc[start_idx_for_this_gen_eval : end_idx_for_this_gen_eval]

    if len(current_eval_scaled_data) < 10: 
        for _, genome in genomes: genome.fitness = -1e12 
        return

    # Define Profit Targets and Scaling Factors INSIDE eval_genomes
    # These targets are for the *current evaluation window*
    PROFIT_TARGET_BREAKEVEN = 0.002       # 0.2% profit (barely beat fees)
    PROFIT_TARGET_MODEST = 0.01           # 1.0% profit
    PROFIT_TARGET_GOOD = 0.025          # 2.5% profit
    PROFIT_TARGET_EXCEPTIONAL = 0.05    # 5.0% profit
    
    # Fitness scaling factor based on initial capital
    FITNESS_PROFIT_SCALER = INITIAL_STARTING_CAPITAL * 10.0 # Tunable, scales profit to fitness score
    FITNESS_LOSS_SCALER = INITIAL_STARTING_CAPITAL * 15.0   # Tunable, scales loss to fitness penalty (make it higher than profit)
    RECORD_BREAK_BASE_BONUS = INITIAL_STARTING_CAPITAL * 50.0 # Large flat bonus for breaking record
    RECORD_BREAK_IMPROVEMENT_SCALER = INITIAL_STARTING_CAPITAL * 200.0 # Scales how much record was broken

    for genome_id, genome in genomes:
        net = neat.nn.FeedForwardNetwork.create(genome, config)
        trader = Trader(INITIAL_STARTING_CAPITAL, INITIAL_STARTING_HOLDINGS, trading_fee_percent=TRADING_FEE_PERCENT) 
        genome.fitness = 0.0 
        num_buys, num_sells, num_profitable_sells = 0,0,0
        window_peak_portfolio = INITIAL_STARTING_CAPITAL 
        
        for i in range(len(current_eval_scaled_data)): 
            if not trader.is_alive: break
            features = current_eval_scaled_data[i]
            price = current_eval_raw_data_df.iloc[i][COL_CLOSE] 
            ts = current_eval_raw_data_df.index[i]
            state = trader.get_state_for_nn(price, MAX_EXPECTED_CREDIT, MAX_EXPECTED_HOLDINGS_VALUE)
            nn_in = np.concatenate((features, state)); action, amount = net.activate(nn_in)
            
            if action > 0.55: 
                if trader.buy(amount * trader.credit, price, ts): num_buys+=1
            elif action < 0.45: 
                old_gains = trader.realized_gains_this_evaluation
                if trader.sell(amount * trader.holdings_shares, price, ts):
                    num_sells+=1; profit_this_trade = trader.realized_gains_this_evaluation - old_gains
                    # Consider a sell "profitable" only if it makes a meaningful amount after fees
                    if profit_this_trade > (TRADING_FEE_PERCENT * INITIAL_STARTING_CAPITAL * 0.02): # e.g. > 0.02% of InitialCap profit
                        num_profitable_sells+=1 
            trader.update_history(ts, price)
            if trader.get_portfolio_value(price) > window_peak_portfolio:
                window_peak_portfolio = trader.get_portfolio_value(price)
        
        final_portfolio = trader.get_portfolio_value(current_eval_raw_data_df.iloc[-1][COL_CLOSE])
        final_credit = trader.credit
        total_trades = num_buys + num_sells
        net_profit_abs = final_portfolio - INITIAL_STARTING_CAPITAL
        profit_ratio = net_profit_abs / INITIAL_STARTING_CAPITAL if INITIAL_STARTING_CAPITAL > 0 else 0.0
        
        fitness_score = 0.0

        # 1. HARD GATES: CATASTROPHIC LOSS OR ILLIQUIDITY
        if final_portfolio < INITIAL_STARTING_CAPITAL * 0.65: # Lost > 35%
            fitness_score = -FITNESS_PROFIT_SCALER * 1000.0 
            genome.fitness = fitness_score; continue 
        if final_credit < INITIAL_STARTING_CAPITAL * 0.05: # Credit < 5%
            fitness_score = -FITNESS_PROFIT_SCALER * 500.0
            genome.fitness = fitness_score; continue 

        # 2. PRIMARY GOAL: SIGNIFICANT NET PROFIT IN THIS WINDOW
        #    Exponentially reward achieving higher profit tiers.
        if profit_ratio >= PROFIT_TARGET_EXCEPTIONAL: # > 5%
            fitness_score = (profit_ratio ** 2.0) * FITNESS_PROFIT_SCALER * 5.0 # Highest reward
        elif profit_ratio >= PROFIT_TARGET_GOOD:     # > 3%
            fitness_score = (profit_ratio ** 1.8) * FITNESS_PROFIT_SCALER * 3.0
        elif profit_ratio >= PROFIT_TARGET_MODEST:   # > 1.5%
            fitness_score = (profit_ratio ** 1.5) * FITNESS_PROFIT_SCALER * 1.5
        elif profit_ratio > PROFIT_TARGET_BREAKEVEN: # > 0.2% (just above fees)
            fitness_score = profit_ratio * FITNESS_PROFIT_SCALER * 0.5 
        elif profit_ratio > -0.01: # Tiny loss, essentially breakeven or negligible profit
            fitness_score = profit_ratio * FITNESS_PROFIT_SCALER * 0.1 # Small negative or tiny positive
        else: # More significant loss (but not catastrophic)
            # Make loss penalty very steep.
            fitness_score = - (abs(profit_ratio) ** 1.5) * FITNESS_LOSS_SCALER * 2.0

        # 3. "BEAT GLOBAL RECORD" ASPIRATIONAL BONUS
        #    This is an *additional* bonus if the window's peak was a new global high
        #    AND the window ended with at least modest profit.
        if window_peak_portfolio > max_portfolio_ever_achieved_in_training and \
           profit_ratio >= PROFIT_TARGET_MODEST : 
            improvement_over_record_abs = window_peak_portfolio - max_portfolio_ever_achieved_in_training
            # Bonus is proportional to how much the record was broken
            fitness_score += (improvement_over_record_abs / INITIAL_STARTING_CAPITAL) * RECORD_BREAK_IMPROVEMENT_SCALER
            fitness_score += RECORD_BREAK_FLAT_BONUS # And a flat component for the achievement
        
        # 4. NUDGES & PENALTIES (Secondary, applied if not already super high/low from above)
        #    a. Realized PnL from profitable sells (if overall doing okay)
        if fitness_score > - (INITIAL_STARTING_CAPITAL * 0.5) and \
           trader.realized_gains_this_evaluation > (INITIAL_STARTING_CAPITAL * 0.01): # Realized PnL > 1% of initial cap
             fitness_score += (trader.realized_gains_this_evaluation / INITIAL_STARTING_CAPITAL) * FITNESS_PROFIT_SCALER * 0.05 # Small multiplier

        #    b. Penalty for low activity IF NOT achieving at least a "Good Win"
        if total_trades < 3 and profit_ratio < PROFIT_TARGET_GOOD:
            fitness_score -= FITNESS_PROFIT_SCALER * 0.2 # Penalty for being inactive and not very profitable

        #    c. Penalty for very high churn IF NOT achieving at least a "Good Win"
        max_trades_for_churn = max(5, len(current_eval_scaled_data) // 8)
        if total_trades > max_trades_for_churn and profit_ratio < PROFIT_TARGET_GOOD:
            fitness_score -= (total_trades - max_trades_for_churn) * (FITNESS_PROFIT_SCALER * 0.0005) * total_trades


        genome.fitness = fitness_score
        if np.isnan(genome.fitness) or np.isinf(genome.fitness): genome.fitness = -1e12

# --- run_simulation_and_plot (Keep as previously corrected) ---
def run_simulation_and_plot(genome, config, data_scaled_np, data_raw, title_prefix):
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
        if action > 0.55: 
            if tr.buy(amount * tr.credit, price, ts): buys+=1
        elif action < 0.45: 
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

# --- run_neat_trader (Keep as previously corrected) ---
def run_neat_trader(config_file): 
    global train_data_scaled_np_global, train_data_raw_global, current_eval_window_start_index, \
           max_portfolio_ever_achieved_in_training, best_record_breaker_details

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
    
    current_eval_window_start_index = 0 
    max_portfolio_ever_achieved_in_training = INITIAL_STARTING_CAPITAL 
    best_record_breaker_details = { "genome_obj": None, "window_fitness": -float('inf'), "portfolio_achieved_on_full_train": -float('inf')}

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
    
    winner_to_evaluate = None; winner_fitness_note = -float('inf'); final_report_source = "None"
    if best_record_breaker_details["genome_obj"] and \
       best_record_breaker_details["portfolio_achieved_on_full_train"] > INITIAL_STARTING_CAPITAL * 1.005 : 
        print(f"INFO: Prioritizing 'Global Record Breaker' Genome ID {best_record_breaker_details['genome_obj'].key} "
              f"(Achieved Portfolio on Full Train: ${best_record_breaker_details['portfolio_achieved_on_full_train']:.2f}, "
              f"Original WindowFit: {best_record_breaker_details['window_fitness']:.2f})")
        winner_to_evaluate = best_record_breaker_details["genome_obj"]
        winner_fitness_note = best_record_breaker_details["window_fitness"] 
        final_report_source = "Global Record Breaker (Portfolio Peak on Full Train Sim)"
    elif gen_reporter.neat_overall_best_genome_obj: 
        print(f"INFO: Selecting winner from Reporter's best overall (by window fitness): ID {gen_reporter.neat_overall_best_genome_obj.key}, WindowFit: {gen_reporter.neat_best_fitness_so_far:.2f}")
        winner_to_evaluate = gen_reporter.neat_overall_best_genome_obj
        winner_fitness_note = gen_reporter.neat_best_fitness_so_far
        final_report_source = "Reporter's Best (by highest window fitness)"
    else: 
        print("DEBUG: No winner from reporter or record breaker. Falling back to NEAT's population scan at end.")
        best_fitness_fallback = -float('inf'); temp_winner_fallback = None
        all_genomes_final_pop = list(pop.population.values()) 
        for g_val_iter in all_genomes_final_pop:
            if g_val_iter.fitness is not None and g_val_iter.fitness > best_fitness_fallback:
                best_fitness_fallback = g_val_iter.fitness; temp_winner_fallback = g_val_iter
        winner_to_evaluate = temp_winner_fallback
        winner_fitness_note = best_fitness_fallback if temp_winner_fallback and temp_winner_fallback.fitness is not None else -float('inf')
        final_report_source = "NEAT Population Fallback"
        if winner_to_evaluate is None: print("CRITICAL ERROR: No winner genome could be determined."); return
        else: print(f"DEBUG: Fallback winner found with fitness {getattr(winner_to_evaluate, 'fitness', 'N/A')}")

    fitness_display_string = f"{winner_fitness_note:.2f}" if winner_fitness_note is not None and winner_fitness_note != -float('inf') else "N/A"
    print(f"\nOverall Best Genome Selected ({final_report_source}): ID {winner_to_evaluate.key if winner_to_evaluate else 'N/A'}, Fitness Context: {fitness_display_string}")
    
    if winner_to_evaluate:
        with open(f"winner_genome_{TICKER}.pkl", "wb") as f: pickle.dump(winner_to_evaluate, f)
        print(f"Saved overall best genome to winner_genome_{TICKER}.pkl")

    gen_reporter.plot_generational_metrics() 

    if winner_to_evaluate:
        print("\n--- Final Evaluation of Selected Overall Best Genome ---")
        run_simulation_and_plot(winner_to_evaluate, cfg, train_data_scaled_np_global, train_data_raw_global, "Selected Best Genome on Training Data")
        run_simulation_and_plot(winner_to_evaluate, cfg, val_scaled_np, val_data_raw, "Selected Best Genome on Validation Data")
    else:
        print("No best genome found to evaluate.")

if __name__ == "__main__":
    if not os.path.exists(CONFIG_FILE_PATH): print(f"Config file not found: {CONFIG_FILE_PATH}")
    else: run_neat_trader(CONFIG_FILE_PATH)
