# neat_attention_trader.py
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
import math # For pow
import torch 

from trader import Trader 
from functionalities import plot_backtest_results, plot_generational_performance 
from attention import initialize_shared_attention, get_attention_output 

# --- Configuration ---
TICKER = "BTC-USD"
DATA_PERIOD = "7d" 
DATA_INTERVAL = "1m"
TRAIN_DAYS = 5     
INITIAL_STARTING_CAPITAL = 200.0
INITIAL_STARTING_HOLDINGS = 0.0
N_LAGS = 2 
CONFIG_FILE_PATH = "./config-feedforward-attention" 
N_GENERATIONS = 150 
MAX_EXPECTED_CREDIT = INITIAL_STARTING_CAPITAL * 5 
MAX_EXPECTED_HOLDINGS_VALUE = INITIAL_STARTING_CAPITAL * 5 
PLOT_BEST_OF_GENERATION_EVERY = 10 
PLOT_WINDOW_PERFORMANCE = True
TRADING_FEE_PERCENT = 0.001 
EVAL_WINDOW_SIZE_MINUTES = 3 * 60 

# --- Attention Mechanism Configuration ---
ATTENTION_SEQUENCE_LENGTH = 15  
ATTENTION_OUTPUT_DIM = 24       
ATTENTION_HEADS = 3             
if ATTENTION_OUTPUT_DIM % ATTENTION_HEADS != 0:
    raise ValueError("ATTENTION_OUTPUT_DIM must be divisible by ATTENTION_HEADS for multi-head attention internal splitting.")

# --- Fitness Scaler & Threshold (GLOBAL) ---
FITNESS_SCALER = INITIAL_STARTING_CAPITAL * 10.0 
FITNESS_THRESHOLD_CONFIG = 500000.0 # Using your high threshold from config

_COLUMN_NAMES_RESOLVED_FOR_CURRENT_FETCH = False
COL_OPEN, COL_HIGH, COL_LOW, COL_CLOSE, COL_VOLUME = 'Open', 'High', 'Low', 'Close', 'Volume'
current_eval_window_start_index = 0
max_portfolio_ever_achieved_in_training = INITIAL_STARTING_CAPITAL 
best_record_breaker_details = {
    "genome_obj": None, "window_fitness": -float('inf'), "portfolio_achieved_on_full_train": -float('inf')
}
current_eval_window_raw_data_for_plotting = None

# --- GenerationReporter Class (Unchanged from previous correct version) ---
# ... (Insert the full GenerationReporter class here) ...
class GenerationReporter(neat.reporting.BaseReporter):
    def __init__(self, plot_interval, train_data_scaled_for_reporter_features, train_data_raw_for_reporter_prices, neat_config, initial_capital, trading_fee):
        super().__init__()
        self.plot_interval = plot_interval; self.generation_count = 0
        self.train_data_scaled_features_global_ref = train_data_scaled_for_reporter_features
        self.train_data_raw_prices_global_ref = train_data_raw_for_reporter_prices
        self.neat_config = neat_config; self.initial_capital = initial_capital
        self.trading_fee = trading_fee
        self.neat_best_fitness_so_far = -float('inf')
        self.neat_overall_best_genome_obj = None
        self.generations_list = []
        self.metrics_history = {
            "Best Fitness (Window)": [],
            "Max Portfolio Ever ($) (Training Record)": [],
            "Best Gen Genome's Portfolio ($) (Full Train Sim)": [],
            "Best Gen Genome's Net Profit ($) (Full Train Sim)": [],
            "Best Gen Genome's Credit ($) (Full Train Sim)": [],
            "Best Gen Genome's Total Trades (Full Train Sim)": [],
            "Best Gen Genome's Buys (Full Train Sim)": [],
            "Best Gen Genome's Sells (Full Train Sim)": [],
            "Best Gen Genome's Fees Paid ($) (Full Train Sim)": [],
            "Projected Weekly Return (%) (Full Train Sim)": [],
            "Projected Monthly Return (%) (Full Train Sim)": []
        }

    def start_generation(self, generation):
        global current_eval_window_start_index, current_eval_window_raw_data_for_plotting, \
               train_data_scaled_np_global 
        
        self.generation_count = generation
        full_train_len = len(train_data_scaled_np_global) 
        window_size_for_eval = EVAL_WINDOW_SIZE_MINUTES

        min_eval_window_size_for_reporter = ATTENTION_SEQUENCE_LENGTH + 10 
        if window_size_for_eval >= full_train_len or window_size_for_eval <= min_eval_window_size_for_reporter : 
            window_size_for_eval = max(min_eval_window_size_for_reporter + 5, full_train_len // 3)

        if full_train_len > window_size_for_eval:
            advance_step = max(1, int(window_size_for_eval * 0.75)) 
            total_advanceable_range = max(0, full_train_len - window_size_for_eval)
            if total_advanceable_range > 0 :
                current_eval_window_start_index = (generation * advance_step) % total_advanceable_range
            else:
                current_eval_window_start_index = 0
        else:
            current_eval_window_start_index = 0

        current_eval_window_start_index = max(0, current_eval_window_start_index)
        current_eval_window_start_index = max(ATTENTION_SEQUENCE_LENGTH -1 , current_eval_window_start_index) 
        current_eval_window_start_index = min(current_eval_window_start_index, max(0, full_train_len - window_size_for_eval))

        eval_win_start_idx = current_eval_window_start_index
        eval_win_end_idx = min(eval_win_start_idx + window_size_for_eval, full_train_len)
        
        if eval_win_end_idx <= eval_win_start_idx: 
             current_eval_window_raw_data_for_plotting = pd.DataFrame() 
        else:
            current_eval_window_raw_data_for_plotting = self.train_data_raw_prices_global_ref.iloc[eval_win_start_idx:eval_win_end_idx]


        if not current_eval_window_raw_data_for_plotting.empty and \
           len(self.train_data_raw_prices_global_ref) > eval_win_start_idx and \
           len(self.train_data_raw_prices_global_ref) >= eval_win_end_idx and \
           eval_win_end_idx > eval_win_start_idx:
            start_date_str = self.train_data_raw_prices_global_ref.index[eval_win_start_idx].strftime('%Y-%m-%d %H:%M')
            end_date_str = self.train_data_raw_prices_global_ref.index[eval_win_end_idx-1].strftime('%Y-%m-%d %H:%M')
            print(f"  Gen {generation}: eval_genomes window: Global Idx {eval_win_start_idx}-{eval_win_end_idx-1} ({start_date_str} to {end_date_str}), size={eval_win_end_idx - eval_win_start_idx}")
        else:
            print(f"  Gen {generation}: eval_genomes window: Global Idx {eval_win_start_idx}-{eval_win_end_idx-1}, size={eval_win_end_idx - eval_win_start_idx}. (Note: May be invalid if size is too small)")


    def _actual_end_of_generation_logic(self, config, population_genomes_dict, species_set_object):
        global max_portfolio_ever_achieved_in_training, best_record_breaker_details, \
               current_eval_window_raw_data_for_plotting, train_data_scaled_np_global, \
               num_input_features_from_data_global 

        best_genome_this_gen_by_window_fitness, current_gen_max_window_fitness = None, -float('inf')
        all_current_genomes = list(population_genomes_dict.values())
        for g_obj in all_current_genomes: 
            if g_obj.fitness is not None and g_obj.fitness > current_gen_max_window_fitness:
                current_gen_max_window_fitness, best_genome_this_gen_by_window_fitness = g_obj.fitness, g_obj

        if best_genome_this_gen_by_window_fitness and (current_gen_max_window_fitness > self.neat_best_fitness_so_far):
            self.neat_best_fitness_so_far = current_gen_max_window_fitness
            self.neat_overall_best_genome_obj = best_genome_this_gen_by_window_fitness
            print(f"  REPORTER: ** New best NEAT genome (by window fitness)! ** Gen: {self.generation_count}, ID: {best_genome_this_gen_by_window_fitness.key}, Fitness: {current_gen_max_window_fitness:.2f}")

        self.generations_list.append(self.generation_count)
        self.metrics_history["Max Portfolio Ever ($) (Training Record)"].append(max_portfolio_ever_achieved_in_training)
        
        projected_weekly_return_pct_train_val = np.nan
        projected_monthly_return_pct_train_val = np.nan

        if best_genome_this_gen_by_window_fitness:
            self.metrics_history["Best Fitness (Window)"].append(current_gen_max_window_fitness)
            
            net_full_sim = neat.nn.FeedForwardNetwork.create(best_genome_this_gen_by_window_fitness, self.neat_config)
            rep_trader_full_sim = Trader(self.initial_capital, INITIAL_STARTING_HOLDINGS, trading_fee_percent=self.trading_fee)
            buys_rep_sim, sells_rep_sim = 0,0
            
            sim_start_index_reporter = ATTENTION_SEQUENCE_LENGTH - 1
            final_pf_rep = self.initial_capital 
            net_profit_rep = 0.0

            if sim_start_index_reporter >= len(self.train_data_scaled_features_global_ref):
                print(f"    REPORTER: Full train sim skipped for Gen {self.generation_count}, data too short for attention.")
            else:
                for i in range(sim_start_index_reporter, len(self.train_data_scaled_features_global_ref)):
                    if not rep_trader_full_sim.is_alive: break
                    
                    current_step_features_np = self.train_data_scaled_features_global_ref[i]
                    start_seq_idx = i - ATTENTION_SEQUENCE_LENGTH + 1
                    sequence_for_attention_np = self.train_data_scaled_features_global_ref[start_seq_idx : i+1]
                    
                    attention_context_np = get_attention_output(
                        sequence_for_attention_np,
                        current_seq_len=sequence_for_attention_np.shape[0],
                        target_seq_len=ATTENTION_SEQUENCE_LENGTH,
                        feature_dim=num_input_features_from_data_global
                    )

                    price = self.train_data_raw_prices_global_ref.iloc[i][COL_CLOSE]
                    ts = self.train_data_raw_prices_global_ref.index[i]
                    state = rep_trader_full_sim.get_state_for_nn(price, MAX_EXPECTED_CREDIT, MAX_EXPECTED_HOLDINGS_VALUE)
                    
                    nn_in = np.concatenate((current_step_features_np, attention_context_np, state))
                    action, amount = net_full_sim.activate(nn_in)

                    if action > 0.55: 
                        if rep_trader_full_sim.buy(amount * rep_trader_full_sim.credit, price, ts): buys_rep_sim +=1
                    elif action < 0.45: 
                        if rep_trader_full_sim.sell(amount * rep_trader_full_sim.holdings_shares, price, ts): sells_rep_sim +=1
                    rep_trader_full_sim.update_history(ts, price)
                
                if not self.train_data_raw_prices_global_ref.empty: 
                    final_pf_rep = rep_trader_full_sim.get_portfolio_value(self.train_data_raw_prices_global_ref.iloc[-1][COL_CLOSE])
                net_profit_rep = final_pf_rep - self.initial_capital

            if rep_trader_full_sim.max_portfolio_value_achieved > max_portfolio_ever_achieved_in_training:
                max_portfolio_ever_achieved_in_training = rep_trader_full_sim.max_portfolio_value_achieved
                best_record_breaker_details["genome_obj"] = copy.deepcopy(best_genome_this_gen_by_window_fitness)
                best_record_breaker_details["window_fitness"] = current_gen_max_window_fitness
                best_record_breaker_details["portfolio_achieved_on_full_train"] = rep_trader_full_sim.max_portfolio_value_achieved
                print(f"    REPORTER: !!! NEW GLOBAL MAX PORTFOLIO (on Full Train Sim): ${max_portfolio_ever_achieved_in_training:.2f} "
                      f"by Gen {self.generation_count}'s Best (ID {best_genome_this_gen_by_window_fitness.key}, WindowFit {current_gen_max_window_fitness:.2f}) !!!")

            self.metrics_history["Best Gen Genome's Portfolio ($) (Full Train Sim)"].append(final_pf_rep)
            self.metrics_history["Best Gen Genome's Net Profit ($) (Full Train Sim)"].append(net_profit_rep)
            self.metrics_history["Best Gen Genome's Credit ($) (Full Train Sim)"].append(rep_trader_full_sim.credit)
            self.metrics_history["Best Gen Genome's Total Trades (Full Train Sim)"].append(len(rep_trader_full_sim.trade_log))
            self.metrics_history["Best Gen Genome's Buys (Full Train Sim)"].append(buys_rep_sim)
            self.metrics_history["Best Gen Genome's Sells (Full Train Sim)"].append(sells_rep_sim)
            self.metrics_history["Best Gen Genome's Fees Paid ($) (Full Train Sim)"].append(rep_trader_full_sim.total_fees_paid)

            print(f"    REPORTER Gen {self.generation_count} Best (ID {best_genome_this_gen_by_window_fitness.key}, WindowFit: {current_gen_max_window_fitness:.2f}) "
                  f"SimOnFullTrain: NetProfit: ${net_profit_rep:.2f}, Pf: ${final_pf_rep:.2f}, Cr: ${rep_trader_full_sim.credit:.2f}, "
                  f"Trades: {len(rep_trader_full_sim.trade_log)} (B:{buys_rep_sim}/S:{sells_rep_sim}), Fees: ${rep_trader_full_sim.total_fees_paid:.2f}")
            
            if not self.train_data_raw_prices_global_ref.empty and self.initial_capital > 1e-6 and \
               sim_start_index_reporter < len(self.train_data_scaled_features_global_ref) : 
                
                train_start_time = self.train_data_raw_prices_global_ref.index[sim_start_index_reporter]
                train_end_time = self.train_data_raw_prices_global_ref.index[-1] 
                train_duration_seconds = (train_end_time - train_start_time).total_seconds()

                if train_duration_seconds > 3600: 
                    train_profit_ratio_for_projection = net_profit_rep / self.initial_capital
                    seconds_in_week = 7 * 24 * 3600
                    seconds_in_month = 30 * 24 * 3600 
                    periods_in_week = seconds_in_week / train_duration_seconds
                    periods_in_month = seconds_in_month / train_duration_seconds
                    
                    projected_weekly_return_pct_train_val = (math.pow(1 + train_profit_ratio_for_projection, periods_in_week) - 1) * 100
                    projected_monthly_return_pct_train_val = (math.pow(1 + train_profit_ratio_for_projection, periods_in_month) - 1) * 100
                    
                    print(f"      REPORTER Projected (Full Train Sim, Gen {self.generation_count} Best, COMPOUNDED over ~{train_duration_seconds/3600:.1f}hrs): "
                          f"Weekly: {projected_weekly_return_pct_train_val:.2f}%, Monthly: {projected_monthly_return_pct_train_val:.2f}%")
                    if abs(projected_weekly_return_pct_train_val) > 1000 or abs(projected_monthly_return_pct_train_val) > 5000:
                        print("      WARNING (Full Train Proj): Projections very high/low. Interpret with caution (in-sample performance).")
                else:
                     print(f"      REPORTER: Full training data sim period ({pd.to_timedelta(train_duration_seconds, unit='s')}) too short for weekly/monthly projection.")

            if self.plot_interval > 0 and (self.generation_count + 1) % self.plot_interval == 0:
                print(f"      REPORTER: Plotting for Gen {self.generation_count}...")
                run_simulation_and_plot(best_genome_this_gen_by_window_fitness, self.neat_config,
                                        self.train_data_scaled_features_global_ref, 
                                        self.train_data_raw_prices_global_ref,      
                                        title_prefix=f"Gen {self.generation_count} Best (Full Train Sim)")
                
                if PLOT_WINDOW_PERFORMANCE and current_eval_window_raw_data_for_plotting is not None and not current_eval_window_raw_data_for_plotting.empty:
                    eval_win_global_start_idx = current_eval_window_start_index
                    eval_win_len = len(current_eval_window_raw_data_for_plotting)
                    current_window_scaled_data_for_plot = self.train_data_scaled_features_global_ref[eval_win_global_start_idx : eval_win_global_start_idx + eval_win_len]
                    
                    if len(current_window_scaled_data_for_plot) > 0 and \
                       len(current_window_scaled_data_for_plot) == len(current_eval_window_raw_data_for_plotting) and \
                       len(current_window_scaled_data_for_plot) >= ATTENTION_SEQUENCE_LENGTH:
                        run_simulation_and_plot(best_genome_this_gen_by_window_fitness, self.neat_config,
                                                current_window_scaled_data_for_plot, 
                                                current_eval_window_raw_data_for_plotting, 
                                                title_prefix=f"Gen {self.generation_count} Best (Actual Eval Window)")
                    else:
                        print(f"      REPORTER: Skipped plotting eval window for Gen {self.generation_count} due to data mismatch/empty/too short. Scaled len: {len(current_window_scaled_data_for_plot)}, Raw len: {len(current_eval_window_raw_data_for_plotting)}")
        else: 
            for key_metric in self.metrics_history.keys():
                if key_metric == "Max Portfolio Ever ($) (Training Record)":
                    self.metrics_history[key_metric].append(max_portfolio_ever_achieved_in_training)
                else:
                    self.metrics_history[key_metric].append(np.nan) 
            print(f"  REPORTER: No genome with reportable fitness in gen {self.generation_count}.")
        
        self.metrics_history["Projected Weekly Return (%) (Full Train Sim)"].append(projected_weekly_return_pct_train_val)
        self.metrics_history["Projected Monthly Return (%) (Full Train Sim)"].append(projected_monthly_return_pct_train_val)

    def end_generation(self, config, population_genomes_dict, species_set):
        self._actual_end_of_generation_logic(config, population_genomes_dict, species_set)

    def post_evaluate(self, config, population_object, species_set_object, best_genome_from_neat):
        if best_genome_from_neat and best_genome_from_neat.fitness is not None:
            if self.neat_overall_best_genome_obj is None or best_genome_from_neat.fitness > self.neat_best_fitness_so_far:
                self.neat_best_fitness_so_far = best_genome_from_neat.fitness
                self.neat_overall_best_genome_obj = best_genome_from_neat
    def found_solution(self, config, generation, best_found_by_neat): 
        print(f"REPORTER: Solution found in generation {generation} by genome {best_found_by_neat.key} with fitness {best_found_by_neat.fitness:.2f}!")
        if best_found_by_neat and best_found_by_neat.fitness is not None and \
           (self.neat_overall_best_genome_obj is None or best_found_by_neat.fitness > self.neat_best_fitness_so_far):
            self.neat_best_fitness_so_far = best_found_by_neat.fitness
            self.neat_overall_best_genome_obj = best_found_by_neat
    def info(self, msg): pass
    
    def plot_generational_metrics(self):
        if not self.generations_list:
            print("No generational data accumulated to plot (generations_list is empty).")
            return

        metrics_to_plot = {k: v for k, v in self.metrics_history.items() if "Full Train Sim" in k or "Best Fitness" in k or "Max Portfolio Ever" in k or "Projected" in k} 
        if any(len(v) > 0 for v in metrics_to_plot.values()): 
            valid_metrics_history = {}
            for k, v_list in metrics_to_plot.items():
                if len(v_list) < len(self.generations_list):
                    v_list_padded = v_list + [np.nan] * (len(self.generations_list) - len(v_list))
                else:
                    v_list_padded = v_list[:len(self.generations_list)] 
                
                if not all(np.isnan(val) if isinstance(val, float) else False for val in v_list_padded):
                    valid_metrics_history[k] = v_list_padded
            
            if valid_metrics_history:
                 plot_generational_performance(self.generations_list, valid_metrics_history, title="Key Metrics Per Generation (Including Projections)")
            else: print("No valid generational metrics data to plot (all NaNs after alignment).")
        else: print("No generational data accumulated to plot (metrics_history values are empty).")

# --- Helper Functions (Unchanged from previous correct version) ---
# ... (resolve_column_names, fetch_data, calculate_features, add_lagged_features, split_data_by_days, normalize_data) ...
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
                _COL_OPEN_temp, _COL_HIGH_temp, _COL_LOW_temp, _COL_CLOSE_temp, _COL_VOLUME_temp = (p_info['cols']['Open'], p_info['cols']['High'], p_info['cols']['Low'], p_info['cols']['Close'], p_info['cols']['Volume'])
                matched_pattern = True; break
        if not matched_pattern:
            pot_price_metric = {'Open': ('Price', 'Open'), 'High': ('Price', 'High'), 'Low': ('Price', 'Low'), 'Close': ('Price', 'Close')}
            actual_vol_col = None
            if ('Price', 'Volume') in df_columns: actual_vol_col = ('Price', 'Volume')
            elif ('Volume', '') in df_columns: actual_vol_col = ('Volume', '')
            elif ('Volume', ticker_symbol_str) in df_columns: actual_vol_col = ('Volume', ticker_symbol_str)
            if all(c in df_columns for c in pot_price_metric.values()) and actual_vol_col:
                _COL_OPEN_temp, _COL_HIGH_temp, _COL_LOW_temp, _COL_CLOSE_temp, _COL_VOLUME_temp = pot_price_metric['Open'], pot_price_metric['High'], pot_price_metric['Low'], pot_price_metric['Close'], actual_vol_col
                matched_pattern = True
        if not matched_pattern: print(f"WARNING: Unhandled yfinance MultiIndex {df_columns.names}. Defaulting. Cols: {df_columns.tolist()}")
    elif not all(col_str in df_columns for col_str in [_COL_OPEN_temp, _COL_HIGH_temp, _COL_LOW_temp, _COL_CLOSE_temp, _COL_VOLUME_temp]):
        print(f"WARNING: Standard column names not found. Defaulting. Cols: {df_columns.tolist()}")
    else: matched_pattern = True
    COL_OPEN, COL_HIGH, COL_LOW, COL_CLOSE, COL_VOLUME = _COL_OPEN_temp, _COL_HIGH_temp, _COL_LOW_temp, _COL_CLOSE_temp, _COL_VOLUME_temp
    _COLUMN_NAMES_RESOLVED_FOR_CURRENT_FETCH = True

def fetch_data(ticker, period, interval):
    global _COLUMN_NAMES_RESOLVED_FOR_CURRENT_FETCH; _COLUMN_NAMES_RESOLVED_FOR_CURRENT_FETCH = False
    print(f"Fetching data for {ticker} ({period}, {interval})...")
    if interval == "1m" and pd.to_timedelta(period) > pd.to_timedelta("7d"):
        print(f"WARNING: yfinance may restrict '1m' interval data to 7 days for free API. Requested period '{period}' might be truncated.")
    
    if isinstance(ticker, str): data = yf.download(ticker, period=period, interval=interval, progress=False, auto_adjust=True)
    else: data = yf.download(ticker, period=period, interval=interval, progress=False, auto_adjust=True, group_by='ticker' if isinstance(ticker, list) and len(ticker) > 1 else None)
    
    if data.empty: raise ValueError(f"No data fetched for {ticker} with period {period} and interval {interval}.")
    
    print(f"Actual data range fetched: {data.index.min()} to {data.index.max()} ({len(data)} rows)")

    actual_ticker_for_resolve = ticker[0] if isinstance(ticker, list) else ticker
    resolve_column_names(data.columns, actual_ticker_for_resolve)
    required_cols = [COL_OPEN, COL_HIGH, COL_LOW, COL_CLOSE, COL_VOLUME]
    missing = [c for c in required_cols if c not in data.columns]
    if missing: raise KeyError(f"Cols missing after fetch: {missing}. DF cols: {data.columns.tolist()}")
    data.dropna(subset=required_cols, inplace=True)
    if data.empty: raise ValueError(f"Data became empty after dropping NA for {ticker}.")
    print(f"Data ready after NA drop: {data.shape[0]} rows")
    return data

def calculate_features(df_ohlcv): 
    print("Calculating Features (TAs and Multi-Timeframe Changes)...")
    df = df_ohlcv.copy() 

    if not all(col in df.columns for col in [COL_OPEN, COL_HIGH, COL_LOW, COL_CLOSE, COL_VOLUME]):
        raise ValueError("One or more required OHLCV columns are missing before TA calculation.")
    
    df_ta_calc = df.copy() 
    df_ta = ta.add_all_ta_features(df_ta_calc, COL_OPEN, COL_HIGH, COL_LOW, COL_CLOSE, COL_VOLUME, fillna=True)
    
    prefs = ['trend_macd', 'trend_macd_signal', 'trend_macd_diff', 'momentum_rsi', 'momentum_stoch_rsi',
             'volatility_atr', 'trend_ema_fast', 'trend_ema_slow', 'volume_obv', 'others_cr'] 
    
    out_df = pd.DataFrame(index=df.index)
    out_df['Close'] = df[COL_CLOSE].copy() 

    selected_ta_cols_for_output = []
    processed_ta_names = set() 

    for p in prefs:
        for col_name_ta in df_ta.columns:
            str_col_ta = str(col_name_ta)
            is_tuple_ta = isinstance(col_name_ta, tuple)
            
            base_name_ta = col_name_ta[0] if is_tuple_ta else str_col_ta
            
            if base_name_ta.lower().startswith(p.lower()) and base_name_ta not in processed_ta_names:
                unique_simp_name = p
                ctr = 1
                while unique_simp_name in out_df.columns: 
                    unique_simp_name = f"{p}_{ctr}"
                    ctr += 1
                out_df[unique_simp_name] = df_ta[col_name_ta].values
                selected_ta_cols_for_output.append(unique_simp_name) 
                processed_ta_names.add(base_name_ta) 
                break 
    
    print(f"Selected TA features: {selected_ta_cols_for_output}")

    timeframes_minutes = {
        '1H': 60, '2H': 120, '6H': 360, '12H': 720, '24H': 1440
    }
    added_mtf_features = []
    for tf_name, minutes in timeframes_minutes.items():
        feature_name = f'close_pct_chg_{tf_name}'
        shifted_close = df[COL_CLOSE].shift(minutes)
        shifted_close_safe = shifted_close.replace(0, np.nan) 
        out_df[feature_name] = (df[COL_CLOSE] / shifted_close_safe) - 1
        added_mtf_features.append(feature_name)
    print(f"Added Multi-Timeframe % change features: {added_mtf_features}")
    
    out_df.fillna(0.0, inplace=True) 
    out_df.replace([np.inf, -np.inf], 0.0, inplace=True)
    return out_df

def add_lagged_features(df, n_lags=1):
    print(f"Adding {n_lags} lags..."); lagged_df = df.copy()
    features_to_lag = [col for col in df.columns if col != 'Close'] 
    
    if not features_to_lag:
        print("Warning: No features to lag. Skipping lag addition.")
        return lagged_df

    for lag in range(1, n_lags + 1):
        shifted = df[features_to_lag].shift(lag)
        for col in features_to_lag:
            lagged_df[f'{col}_lag{lag}'] = shifted[col]
    
    lagged_df.dropna(inplace=True)
    if lagged_df.empty and n_lags > 0 and not df.empty : 
        print(f"WARNING: DataFrame became empty after adding {n_lags} lags and dropping NA. Original data (len {len(df)}) might be too short.")
    print(f"Shape after lags/dropna: {lagged_df.shape}"); return lagged_df

def split_data_by_days(df, train_days_count):
    print(f"Splitting: {train_days_count} train days..."); df = df.sort_index()
    unique_days = sorted(df.index.normalize().unique())
    if len(unique_days) == 0: print("ERROR split_data: No unique days found."); return None, None
    
    if train_days_count >= len(unique_days):
        print(f"WARNING: train_days_count ({train_days_count}) is >= total unique days ({len(unique_days)}).")
        if len(unique_days) > 1:
            train_days_count = len(unique_days) - 1 
            print(f"Adjusted train_days_count to {train_days_count} to allow for a minimal validation set.")
        else: 
            print("Only one unique day of data. Using all for training, no validation set will be created.")
            train_df = df.copy()
            validation_df = pd.DataFrame(columns=df.columns, index=pd.to_datetime([])) 
            print(f"Train: {train_df.shape}, {train_df.index.min()} to {train_df.index.max() if not train_df.empty else 'N/A'}")
            print(f"Valid: {validation_df.shape}, Validation set is empty.")
            return train_df, validation_df

    if train_days_count <= 0: 
        print("ERROR: train_days_count is 0 or less. Cannot create training set.")
        return None, None

    split_date_boundary = unique_days[train_days_count -1] 
    train_df = df[df.index.normalize() <= split_date_boundary]
    validation_df = df[df.index.normalize() > split_date_boundary]
    
    print(f"Train: {train_df.shape}, {train_df.index.min()} to {train_df.index.max() if not train_df.empty else 'N/A'}")
    print(f"Valid: {validation_df.shape}, {validation_df.index.min()} to {validation_df.index.max() if not validation_df.empty else 'N/A'}")
    
    if train_df.empty: print("ERROR split_data: Training dataframe is empty."); return None, None
    if validation_df.empty: print("WARNING split_data: Validation dataframe is empty. Consider increasing DATA_PERIOD or reducing TRAIN_DAYS.");
    
    return train_df, validation_df

def normalize_data(train_df, val_df):
    print("Normalizing...");
    train_close_prices = train_df['Close'].copy() if 'Close' in train_df.columns else None
    val_close_prices = val_df['Close'].copy() if 'Close' in val_df.columns and not val_df.empty else None

    train_features = train_df.drop(columns=['Close'], errors='ignore')
    val_features = val_df.drop(columns=['Close'], errors='ignore') if not val_df.empty else pd.DataFrame(columns=train_features.columns, index=val_df.index)


    if train_features.empty:
        print("Warning: Training features are empty after dropping 'Close'. Returning original-like DFs.")
        train_scaled_df = pd.DataFrame(index=train_df.index)
        if train_close_prices is not None: train_scaled_df['Close'] = train_close_prices
        
        val_scaled_df = pd.DataFrame(index=val_df.index)
        if val_close_prices is not None: val_scaled_df['Close'] = val_close_prices
        
        return train_scaled_df, val_scaled_df, None 

    scaler = MinMaxScaler()
    train_features_scaled_np = scaler.fit_transform(train_features)
    
    if not val_features.empty and not val_features.columns.empty: 
        val_features_scaled_np = scaler.transform(val_features)
    else:
        val_features_scaled_np = np.array([]).reshape(0, train_features.shape[1]) 

    train_scaled_df = pd.DataFrame(train_features_scaled_np, columns=train_features.columns, index=train_features.index)
    
    if not val_features.empty and not val_features.columns.empty:
        val_scaled_df = pd.DataFrame(val_features_scaled_np, columns=val_features.columns, index=val_features.index)
    else: 
        val_scaled_df = pd.DataFrame(columns=train_features.columns, index=val_features.index, dtype=float)

    if train_close_prices is not None:
        train_scaled_df['Close'] = train_close_prices
    if val_close_prices is not None and not val_df.empty: 
        val_scaled_df['Close'] = val_close_prices
    elif val_df.empty and 'Close' in train_scaled_df.columns : 
        val_scaled_df['Close'] = pd.Series(dtype=float, index=val_scaled_df.index)

    return train_scaled_df, val_scaled_df, scaler


# --- FITNESS FUNCTION (New Approach v8 - "Aggressive Culling & High Profit Pursuit") ---
def eval_genomes(genomes, config):
    global train_data_scaled_np_global, train_data_raw_prices_global, \
           current_eval_window_start_index, max_portfolio_ever_achieved_in_training, \
           num_input_features_from_data_global, FITNESS_SCALER 

    # --- Data preparation ---
    full_train_len = len(train_data_scaled_np_global)
    window_size_for_eval = EVAL_WINDOW_SIZE_MINUTES
    min_eval_window_size = ATTENTION_SEQUENCE_LENGTH + 30 
    if window_size_for_eval >= full_train_len or window_size_for_eval <= min_eval_window_size :
        window_size_for_eval = max(min_eval_window_size + 10, full_train_len // 2) 
    start_idx_global_for_this_gen_eval = current_eval_window_start_index
    end_idx_global_for_this_gen_eval = min(start_idx_global_for_this_gen_eval + window_size_for_eval, full_train_len)
    actual_window_len = end_idx_global_for_this_gen_eval - start_idx_global_for_this_gen_eval
    if actual_window_len < min_eval_window_size : 
        if full_train_len > window_size_for_eval :
             start_idx_global_for_this_gen_eval = max(ATTENTION_SEQUENCE_LENGTH -1, full_train_len - window_size_for_eval)
        else: 
            start_idx_global_for_this_gen_eval = ATTENTION_SEQUENCE_LENGTH -1 
        end_idx_global_for_this_gen_eval = min(start_idx_global_for_this_gen_eval + window_size_for_eval, full_train_len)
        actual_window_len = end_idx_global_for_this_gen_eval - start_idx_global_for_this_gen_eval
    if actual_window_len < min_eval_window_size: 
        for _, genome in genomes: genome.fitness = - (FITNESS_SCALER * 2000) 
        return
    current_eval_scaled_features_window = train_data_scaled_np_global[start_idx_global_for_this_gen_eval : end_idx_global_for_this_gen_eval]
    current_eval_raw_prices_df_window = train_data_raw_prices_global.iloc[start_idx_global_for_this_gen_eval : end_idx_global_for_this_gen_eval]

    # --- Fitness Parameters v8 (ALL DEFINED HERE - THOROUGHLY CHECKED) ---
    MANDATORY_IMPROVEMENT_PENALTY = FITNESS_SCALER * 0.75 
    RUIN_THRESHOLD = 0.10 
    RUIN_PENALTY = FITNESS_SCALER * 1000.0 
    SEVERE_LOSS_THRESHOLD = 0.50  
    SEVERE_LOSS_PENALTY_FACTOR = 8.0 
    
    PROFIT_BASE_SCALER = FITNESS_SCALER * 1.0 
    HIGH_PROFIT_THRESHOLD = 0.05 
    HIGH_PROFIT_EXP_BONUS_SCALER = FITNESS_SCALER * 5.0 

    ALPHA_SCALER = FITNESS_SCALER * 6.0  
    
    PROFIT_FACTOR_SCALER = FITNESS_SCALER * 0.5 
    PROFIT_FACTOR_TARGET = 1.35 
    MAX_PROFIT_FACTOR_REWARD_CAP = FITNESS_SCALER * 1.5 
    POOR_PROFIT_FACTOR_PENALTY_SCALER = 1.75 

    REALIZED_PNL_SCALER = FITNESS_SCALER * 1.5 

    window_hours = actual_window_len / 60.0
    MIN_ACTIVE_TRADES = max(3, int(window_hours * 0.75)) 
    MIN_SURVIVAL_TRADES = max(2, int(window_hours * 0.25)) 
    MIN_PROFIT_SURVIVAL_THRESHOLD = 0.001 
    CATASTROPHIC_ACTIVE_LOSS = -0.05 
    EXCESSIVE_CHURN_THRESHOLD_CULL = actual_window_len // 3 
    MODEST_PROFIT_FOR_HIGH_TRADES = 0.005 

    VOLATILITY_THRESHOLD_FOR_INACTION = 0.0020 
    PROFITABLE_BH_THRESHOLD = 0.002 
    MISSED_OPPORTUNITY_PENALTY = FITNESS_SCALER * 4.0 
    FAILURE_TO_EXPLORE_PENALTY = FITNESS_SCALER * 2.0 
    INACTION_OPPORTUNITY_COST_SCALER = FITNESS_SCALER * 1.5 # THIS WAS THE MISSING ONE from previous error

    ACTIVE_LOSS_THRESHOLD = -0.002 
    ACTIVE_LOSS_PENALTY_SCALER = FITNESS_SCALER * 2.5

    MAX_FEE_TO_REALIZED_PROFIT_RATIO = 0.15 
    FEE_EXCESS_PENALTY_SCALER = FITNESS_SCALER * 3.0 

    INTRA_WINDOW_GROWTH_SCALER = FITNESS_SCALER * 0.4

    EXCEPTIONAL_PROFIT_RATIO_WINDOW = 0.05 
    EXCEPTIONAL_PROFIT_POWER = 1.75 
    EXCEPTIONAL_PROFIT_BONUS = FITNESS_SCALER * 6.0 
    B_AND_H_EXCEPTIONAL_THRESHOLD = 0.03 

    GLOBAL_RECORD_BREAK_MASSIVE_BONUS = FITNESS_SCALER * 30.0
    GLOBAL_RECORD_BREAK_PROPORTIONAL_BONUS = FITNESS_SCALER * 12.0 

    MAX_ABS_FITNESS_CAP = FITNESS_SCALER * 120.0 
    VERY_LOW_FITNESS_FOR_CULLING = - (FITNESS_SCALER * 5000.0) 
    # End of Fitness Parameter Definitions


    for genome_id, genome in genomes:
        net = neat.nn.FeedForwardNetwork.create(genome, config)
        trader = Trader(INITIAL_STARTING_CAPITAL, INITIAL_STARTING_HOLDINGS, trading_fee_percent=TRADING_FEE_PERCENT)
        
        gross_profit_from_trades = 0
        gross_loss_from_trades = 0
        window_peak_portfolio_value = INITIAL_STARTING_CAPITAL 
        
        for i_window in range(len(current_eval_scaled_features_window)): 
            current_global_idx = start_idx_global_for_this_gen_eval + i_window
            if current_global_idx < ATTENTION_SEQUENCE_LENGTH -1:
                attention_context_np = np.zeros(ATTENTION_OUTPUT_DIM) 
            else:
                start_seq_global_idx = current_global_idx - ATTENTION_SEQUENCE_LENGTH + 1
                sequence_for_attention_np = train_data_scaled_np_global[start_seq_global_idx : current_global_idx + 1]
                attention_context_np = get_attention_output(
                    sequence_for_attention_np,
                    current_seq_len=sequence_for_attention_np.shape[0],
                    target_seq_len=ATTENTION_SEQUENCE_LENGTH,
                    feature_dim=num_input_features_from_data_global
                )
            current_step_features_np = current_eval_scaled_features_window[i_window]
            price = current_eval_raw_prices_df_window.iloc[i_window][COL_CLOSE]
            ts = current_eval_raw_prices_df_window.index[i_window]
            state = trader.get_state_for_nn(price, MAX_EXPECTED_CREDIT, MAX_EXPECTED_HOLDINGS_VALUE)
            nn_in = np.concatenate((current_step_features_np, attention_context_np, state));
            action_raw, amount_raw = net.activate(nn_in)
            amount_to_use = np.clip(amount_raw, 0.01, 1.0) 

            if not trader.is_alive: break 

            trade_executed_profit = 0 
            if action_raw > 0.55: 
                trade_executed_profit = trader.buy(amount_to_use * trader.credit, price, ts, return_trade_profit=True)
            elif action_raw < 0.45: 
                trade_executed_profit = trader.sell(amount_to_use * trader.holdings_shares, price, ts, return_trade_profit=True)
            
            if isinstance(trade_executed_profit, (float, int)) and trade_executed_profit != 0: 
                if trade_executed_profit > 0:
                    gross_profit_from_trades += trade_executed_profit
                else:
                    gross_loss_from_trades += abs(trade_executed_profit)
            trader.update_history(ts, price)
            current_portfolio = trader.get_portfolio_value(price)
            if current_portfolio > window_peak_portfolio_value:
                window_peak_portfolio_value = current_portfolio
        
        final_portfolio = trader.get_portfolio_value(current_eval_raw_prices_df_window.iloc[-1][COL_CLOSE])
        total_trades = len(trader.trade_log) 
        profit_ratio_overall = (final_portfolio - INITIAL_STARTING_CAPITAL) / INITIAL_STARTING_CAPITAL if INITIAL_STARTING_CAPITAL > 1e-6 else 0.0
        realized_pnl_window_ratio = trader.realized_gains_this_evaluation / INITIAL_STARTING_CAPITAL if INITIAL_STARTING_CAPITAL > 1e-6 else 0.0
        
        buy_hold_profit_ratio = 0.0 
        if len(current_eval_raw_prices_df_window) > 1:
            initial_price_window = current_eval_raw_prices_df_window.iloc[0][COL_CLOSE]
            final_price_window = current_eval_raw_prices_df_window.iloc[-1][COL_CLOSE]
            if initial_price_window > 1e-6:
                bh_start_capital = INITIAL_STARTING_CAPITAL
                bh_shares = (bh_start_capital * (1-TRADING_FEE_PERCENT)) / initial_price_window
                bh_final_value_gross = bh_shares * final_price_window
                bh_final_value_net = bh_final_value_gross * (1 - TRADING_FEE_PERCENT)
                buy_hold_profit_ratio = (bh_final_value_net - bh_start_capital) / bh_start_capital if bh_start_capital > 1e-6 else 0.0

        # --- Fitness Calculation v8 ---
        fitness_score = -MANDATORY_IMPROVEMENT_PENALTY 

        if final_portfolio < INITIAL_STARTING_CAPITAL * RUIN_THRESHOLD:
            fitness_score -= RUIN_PENALTY 
            genome.fitness = fitness_score; continue
        if not trader.is_alive: 
            fitness_score -= RUIN_PENALTY * 0.75
            genome.fitness = fitness_score; continue
        
        if final_portfolio < INITIAL_STARTING_CAPITAL * SEVERE_LOSS_THRESHOLD:
             fitness_score -= (1 - (final_portfolio / (INITIAL_STARTING_CAPITAL + 1e-9) )) * FITNESS_SCALER * SEVERE_LOSS_PENALTY_FACTOR

        core_fitness_contribution = profit_ratio_overall * PROFIT_BASE_SCALER
        
        if profit_ratio_overall > EXCEPTIONAL_PROFIT_RATIO_WINDOW:
            profit_factor_for_bonus = (profit_ratio_overall / EXCEPTIONAL_PROFIT_RATIO_WINDOW if EXCEPTIONAL_PROFIT_RATIO_WINDOW > 0 else 1)
            capped_profit_factor_for_bonus = min(profit_factor_for_bonus, 4.0) 
            bonus = (capped_profit_factor_for_bonus ** EXCEPTIONAL_PROFIT_POWER) * EXCEPTIONAL_PROFIT_BONUS 
            core_fitness_contribution += bonus
        
        fitness_score += core_fitness_contribution
            
        true_alpha_profit_ratio = profit_ratio_overall - buy_hold_profit_ratio # This was 'alpha'
        fitness_score += true_alpha_profit_ratio * ALPHA_SCALER 

        if total_trades >= 2 : 
            profit_factor_val = gross_profit_from_trades / (gross_loss_from_trades + 1e-9) 
            profit_factor_val = min(profit_factor_val, 30.0) 
            if profit_factor_val > PROFIT_FACTOR_TARGET:
                reward = min((profit_factor_val - PROFIT_FACTOR_TARGET) * PROFIT_FACTOR_SCALER, MAX_PROFIT_FACTOR_REWARD_CAP)
                fitness_score += reward
            elif profit_factor_val < 1.0 and gross_loss_from_trades > 1e-6: 
                penalty = (1.0 - profit_factor_val) * PROFIT_FACTOR_SCALER * POOR_PROFIT_FACTOR_PENALTY_SCALER
                fitness_score -= penalty
        
        if realized_pnl_window_ratio > 0: 
            fitness_score += realized_pnl_window_ratio * REALIZED_PNL_SCALER
        elif realized_pnl_window_ratio < 0: 
             fitness_score += realized_pnl_window_ratio * REALIZED_PNL_SCALER * 1.75 

        # --- "Hard Cull" Criteria Application ---
        cull_genome = False
        window_price_std_dev_for_cull = current_eval_raw_prices_df_window[COL_CLOSE].std() / (current_eval_raw_prices_df_window[COL_CLOSE].mean() + 1e-9)
        market_moved_for_cull = window_price_std_dev_for_cull > VOLATILITY_THRESHOLD_FOR_INACTION

        if total_trades < MIN_SURVIVAL_TRADES: 
            if profit_ratio_overall < MIN_PROFIT_SURVIVAL_THRESHOLD: 
                missed_bh_opportunity_for_cull = buy_hold_profit_ratio > PROFITABLE_BH_THRESHOLD and \
                                                 profit_ratio_overall < buy_hold_profit_ratio * 0.75
                failed_to_explore_for_cull = market_moved_for_cull and \
                                             buy_hold_profit_ratio <= PROFITABLE_BH_THRESHOLD
                if missed_bh_opportunity_for_cull:
                    fitness_score -= MISSED_OPPORTUNITY_PENALTY * (1 + min(buy_hold_profit_ratio * 50, 4.0)) 
                    cull_genome = True 
                elif failed_to_explore_for_cull:
                    fitness_score -= FAILURE_TO_EXPLORE_PENALTY * (1 + min(window_price_std_dev_for_cull * 100, 3.0))
                    cull_genome = True

        if not cull_genome and total_trades >= MIN_SURVIVAL_TRADES and profit_ratio_overall < CATASTROPHIC_ACTIVE_LOSS:
            fitness_score -= abs(profit_ratio_overall - CATASTROPHIC_ACTIVE_LOSS) * ACTIVE_LOSS_PENALTY_SCALER * 2.0 
            cull_genome = True
        
        if not cull_genome and total_trades > EXCESSIVE_CHURN_THRESHOLD_CULL and \
           profit_ratio_overall < MODEST_PROFIT_FOR_HIGH_TRADES:
            if trader.realized_gains_this_evaluation <= 1e-6 or \
               (trader.total_fees_paid / (trader.realized_gains_this_evaluation + 1e-9)) > MAX_FEE_TO_REALIZED_PROFIT_RATIO * 2.0: 
                cull_genome = True

        if cull_genome and fitness_score < (FITNESS_SCALER * 0.1) : 
            genome.fitness = VERY_LOW_FITNESS_FOR_CULLING
            continue 
        
        # Global Record Breaking (only for non-culled genomes)
        if window_peak_portfolio_value > max_portfolio_ever_achieved_in_training and profit_ratio_overall > 0.02: 
             current_max_pf_for_ratio = max(INITIAL_STARTING_CAPITAL, max_portfolio_ever_achieved_in_training)
             improvement_ratio_over_record = (window_peak_portfolio_value - max_portfolio_ever_achieved_in_training) / current_max_pf_for_ratio
             fitness_score += GLOBAL_RECORD_BREAK_MASSIVE_BONUS 
             fitness_score += improvement_ratio_over_record * GLOBAL_RECORD_BREAK_PROPORTIONAL_BONUS

        if abs(fitness_score) > MAX_ABS_FITNESS_CAP:
            fitness_score = np.sign(fitness_score) * MAX_ABS_FITNESS_CAP
            
        genome.fitness = fitness_score
        if np.isnan(genome.fitness) or np.isinf(genome.fitness): 
            genome.fitness = - (FITNESS_SCALER * 2000) 


# --- run_simulation_and_plot (Unchanged from previous correct version) ---
# ... (Insert the full run_simulation_and_plot function here) ...
def run_simulation_and_plot(genome, config, data_scaled_features_np_segment, data_raw_prices_df_segment, title_prefix, is_validation_run=False):
    global num_input_features_from_data_global 

    if data_raw_prices_df_segment.empty or COL_CLOSE not in data_raw_prices_df_segment.columns:
        print(f"Plot Info: Raw price data empty or missing '{COL_CLOSE}' for '{title_prefix}'. Skipping.")
        return
    if data_scaled_features_np_segment is None or len(data_scaled_features_np_segment) == 0:
        print(f"Simulation Info: No scaled feature data for '{title_prefix}'. Cannot run simulation.")
        plot_backtest_results(data_raw_prices_df_segment, [], [], f"{title_prefix} Results for {TICKER} (No Sim)", COL_CLOSE)
        return
    if len(data_scaled_features_np_segment) != len(data_raw_prices_df_segment):
        print(f"Data Mismatch Error for '{title_prefix}': Scaled features length ({len(data_scaled_features_np_segment)}) != Raw prices length ({len(data_raw_prices_df_segment)}). Skipping.")
        return
    
    if len(data_scaled_features_np_segment) < ATTENTION_SEQUENCE_LENGTH :
        print(f"Simulation Info: Data segment too short ({len(data_scaled_features_np_segment)}) for attention sequence ({ATTENTION_SEQUENCE_LENGTH}) in '{title_prefix}'. Skipping simulation for plotting.")
        plot_backtest_results(data_raw_prices_df_segment, [], [], f"{title_prefix} Results for {TICKER} (Sim Skipped - Short Data)", COL_CLOSE)
        return

    print(f"\n--- {title_prefix} Evaluation ---")
    net = neat.nn.FeedForwardNetwork.create(genome, config)
    tr = Trader(INITIAL_STARTING_CAPITAL, INITIAL_STARTING_HOLDINGS, trading_fee_percent=TRADING_FEE_PERCENT)
    buys,sells = 0,0
    
    final_val_idx = max(0, ATTENTION_SEQUENCE_LENGTH -1) 
    if final_val_idx >= len(data_raw_prices_df_segment): 
        final_val_idx = len(data_raw_prices_df_segment) -1

    sim_loop_start_index = ATTENTION_SEQUENCE_LENGTH -1
    
    if sim_loop_start_index >= len(data_scaled_features_np_segment):
        print(f"  {title_prefix}: Data segment too short to run simulation loop (len {len(data_scaled_features_np_segment)}, needs > {sim_loop_start_index}).")
    else:
        for i_sim_segment in range(sim_loop_start_index, len(data_scaled_features_np_segment)):
            if not tr.is_alive:
                final_val_idx = i_sim_segment 
                break
            
            current_step_features_segment = data_scaled_features_np_segment[i_sim_segment]
            start_seq_segment_idx = i_sim_segment - ATTENTION_SEQUENCE_LENGTH + 1
            sequence_for_attention_segment_np = data_scaled_features_np_segment[start_seq_segment_idx : i_sim_segment + 1]

            attention_context_np = get_attention_output(
                sequence_for_attention_segment_np,
                current_seq_len=sequence_for_attention_segment_np.shape[0],
                target_seq_len=ATTENTION_SEQUENCE_LENGTH,
                feature_dim=num_input_features_from_data_global 
            )

            current_price = data_raw_prices_df_segment.iloc[i_sim_segment][COL_CLOSE]
            current_ts = data_raw_prices_df_segment.index[i_sim_segment]
            trader_state_inputs = tr.get_state_for_nn(current_price, MAX_EXPECTED_CREDIT, MAX_EXPECTED_HOLDINGS_VALUE)
            nn_in = np.concatenate((current_step_features_segment, attention_context_np, trader_state_inputs))
            
            action_raw, amount_raw = net.activate(nn_in)
            amount_to_use = np.clip(amount_raw, 0.01, 1.0) 

            if action_raw > 0.55: 
                if tr.buy(amount_to_use * tr.credit, current_price, current_ts, return_trade_profit=False): 
                    buys+=1
            elif action_raw < 0.45: 
                if tr.sell(amount_to_use * tr.holdings_shares, current_price, current_ts, return_trade_profit=False):
                    sells+=1
            
            tr.update_history(current_ts, current_price)
            final_val_idx = i_sim_segment 
    
    final_val = INITIAL_STARTING_CAPITAL 
    if final_val_idx >= 0 and final_val_idx < len(data_raw_prices_df_segment) and not data_raw_prices_df_segment.empty :
         final_val = tr.get_portfolio_value(data_raw_prices_df_segment.iloc[final_val_idx][COL_CLOSE])
    elif not data_raw_prices_df_segment.empty: 
        final_val = tr.get_portfolio_value(data_raw_prices_df_segment.iloc[-1][COL_CLOSE])

    print(f"{title_prefix} - Initial: ${INITIAL_STARTING_CAPITAL:.2f}, Final Portfolio: ${final_val:.2f}, Final Credit: ${tr.credit:.2f}")
    print(f"Trades Logged: {len(tr.trade_log)} (Sim Buys: {buys}, Sells: {sells}), Realized PnL: ${tr.realized_gains_this_evaluation:.2f}, Fees Paid: ${tr.total_fees_paid:.2f}")

    profit_abs = final_val - INITIAL_STARTING_CAPITAL
    profit_percentage_on_segment = 0.0
    if INITIAL_STARTING_CAPITAL > 1e-6:
        profit_percentage_on_segment = (profit_abs / INITIAL_STARTING_CAPITAL) * 100
        print(f"Profit/Loss (Portfolio on this data segment): {profit_percentage_on_segment:.2f}%")
    else:
        print(f"Profit/Loss (Portfolio): N/A (Initial capital too small)")

    if is_validation_run and not data_raw_prices_df_segment.empty and INITIAL_STARTING_CAPITAL > 1e-6:
        start_time = data_raw_prices_df_segment.index.min()
        end_time = data_raw_prices_df_segment.index.max()
        duration_seconds = (end_time - start_time).total_seconds()
        
        sim_duration_for_projection_text = f"Duration: {pd.to_timedelta(duration_seconds, unit='s')}"
        if sim_loop_start_index >= len(data_scaled_features_np_segment): 
            sim_duration_for_projection_text = f"Duration: {pd.to_timedelta(duration_seconds, unit='s')} (Sim Loop Skipped)"
        print(f"  Validation Data Timeframe: {start_time.strftime('%Y-%m-%d %H:%M')} to {end_time.strftime('%Y-%m-%d %H:%M')} ({sim_duration_for_projection_text})")

        if duration_seconds > 60 * 30 and sim_loop_start_index < len(data_scaled_features_np_segment): 
            profit_ratio_for_projection = profit_abs / INITIAL_STARTING_CAPITAL
            seconds_in_week = 7 * 24 * 3600; seconds_in_month = 30 * 24 * 3600 
            periods_in_week = seconds_in_week / duration_seconds
            periods_in_month = seconds_in_month / duration_seconds
            
            projected_weekly_return_pct = (math.pow(1 + profit_ratio_for_projection, periods_in_week) - 1) * 100
            projected_monthly_return_pct = (math.pow(1 + profit_ratio_for_projection, periods_in_month) - 1) * 100
            
            print(f"  Projected Performance (based on this validation segment, COMPOUNDED):")
            print(f"    Projected Weekly Return: {projected_weekly_return_pct:.2f}%")
            print(f"    Projected Monthly Return: {projected_monthly_return_pct:.2f}%")
            if abs(projected_weekly_return_pct) > 1000 or abs(projected_monthly_return_pct) > 5000 :
                 print("    WARNING: Projections are very high/low. Interpret with caution.")
        else:
            print(f"  Validation period or sim ({pd.to_timedelta(duration_seconds, unit='s')}) too short for meaningful weekly/monthly projection.")
    
    plot_backtest_results(data_raw_prices_df_segment, tr.trade_log, tr.history, f"{title_prefix} Results for {TICKER}", COL_CLOSE)


# --- run_neat_trader (Main Logic - Unchanged from previous correct version) ---
# ... (Insert the full run_neat_trader function here) ...
def run_neat_trader(config_file):
    global train_data_scaled_np_global, train_data_raw_prices_global, \
           current_eval_window_start_index, max_portfolio_ever_achieved_in_training, \
           best_record_breaker_details, current_eval_window_raw_data_for_plotting, \
           num_input_features_from_data_global, FITNESS_SCALER, FITNESS_THRESHOLD_CONFIG

    raw_df_full = fetch_data(TICKER, DATA_PERIOD, DATA_INTERVAL)
    if raw_df_full.empty: print("ERROR: No data fetched. Exiting."); return
    
    feats_df_full = calculate_features(raw_df_full.copy()) 
    if feats_df_full.empty: print("ERROR: Feature calculation resulted in empty dataframe. Exiting."); return
    
    feats_lags_df_full = add_lagged_features(feats_df_full, N_LAGS)
    if feats_lags_df_full.empty: print("ERROR: Adding lags resulted in empty dataframe. Exiting."); return
    
    if len(feats_lags_df_full) < ATTENTION_SEQUENCE_LENGTH + N_LAGS + 60 : 
        print(f"ERROR: Data too short after lags ({len(feats_lags_df_full)}) for processing. Need at least {ATTENTION_SEQUENCE_LENGTH + N_LAGS + 60} rows. Exiting.")
        return

    train_feature_df, val_feature_df = split_data_by_days(feats_lags_df_full, TRAIN_DAYS)

    if train_feature_df is None or train_feature_df.empty:
        print("ERROR: Training feature dataframe is empty or None after split. Exiting."); return
    if val_feature_df is None: 
        val_feature_df = pd.DataFrame(columns=train_feature_df.columns, index=pd.to_datetime([]))
    
    if len(train_feature_df) < ATTENTION_SEQUENCE_LENGTH + EVAL_WINDOW_SIZE_MINUTES: 
        print(f"ERROR: Training data ({len(train_feature_df)}) too short for attention sequence and a full evaluation window. Needs at least {ATTENTION_SEQUENCE_LENGTH + EVAL_WINDOW_SIZE_MINUTES}. Exiting.")
        return

    train_data_raw_prices_global = raw_df_full[[COL_CLOSE]].loc[train_feature_df.index].copy()
    val_data_raw_prices_df = raw_df_full[[COL_CLOSE]].loc[val_feature_df.index].copy() if not val_feature_df.empty else pd.DataFrame(columns=[COL_CLOSE])

    if train_data_raw_prices_global.empty :
        print("ERROR: Training raw prices dataframe is empty after aligning with feature indices."); return

    train_scaled_features_df, val_scaled_features_df, scaler_obj = normalize_data(
        train_feature_df.copy(), 
        val_feature_df.copy() if not val_feature_df.empty else pd.DataFrame(columns=train_feature_df.columns, index=val_feature_df.index) 
    )
    
    if 'Close' in train_scaled_features_df.columns:
        train_data_scaled_np_global = train_scaled_features_df.drop(columns=['Close'], errors='ignore').to_numpy()
    else: 
        train_data_scaled_np_global = train_scaled_features_df.to_numpy()

    if not val_scaled_features_df.empty and 'Close' in val_scaled_features_df.columns:
        val_scaled_np_features = val_scaled_features_df.drop(columns=['Close'], errors='ignore').to_numpy()
    elif not val_scaled_features_df.empty:
        val_scaled_np_features = val_scaled_features_df.to_numpy()
    else:
        val_scaled_np_features = np.array([])


    if len(train_data_scaled_np_global) == 0:
        print("ERROR: Scaled training features (train_data_scaled_np_global) are empty."); return
    if not val_feature_df.empty and \
       len(val_scaled_np_features) == 0 and \
       len(val_feature_df.drop(columns=['Close'], errors='ignore').columns) > 0 :
        print("WARNING: Scaled validation features are empty, though validation data and features (other than 'Close') were present.")


    num_input_features_from_data_global = train_data_scaled_np_global.shape[1] 
    num_trader_state_features = 3 
    
    initialize_shared_attention(
        input_dim=num_input_features_from_data_global, 
        attention_dim=ATTENTION_OUTPUT_DIM,      
        attention_heads=ATTENTION_HEADS,
        dropout_rate=0.1 
    )
    
    total_nn_inputs = num_input_features_from_data_global + ATTENTION_OUTPUT_DIM + num_trader_state_features
    
    print(f"Number of input features from data (current step): {num_input_features_from_data_global}")
    print(f"Number of attention output features: {ATTENTION_OUTPUT_DIM}")
    print(f"Number of trader state features: {num_trader_state_features}")
    print(f"Total NN inputs: {total_nn_inputs}. CONFIG FILE ('{CONFIG_FILE_PATH}') 'num_inputs' SHOULD BE: {total_nn_inputs}.")

    cfg = neat.Config(neat.DefaultGenome, neat.DefaultReproduction, neat.DefaultSpeciesSet, neat.DefaultStagnation, config_file)
    
    if hasattr(cfg, 'fitness_threshold'):
        FITNESS_THRESHOLD_CONFIG = cfg.fitness_threshold 
        print(f"Using fitness_threshold from config: {FITNESS_THRESHOLD_CONFIG}")
    else: 
        cfg.fitness_threshold = FITNESS_THRESHOLD_CONFIG 
        print(f"Fitness_threshold not in config, using default: {FITNESS_THRESHOLD_CONFIG}")


    if cfg.genome_config.num_inputs != total_nn_inputs:
        print(f"CONFIG ERROR: num_inputs mismatch! Script calculated: {total_nn_inputs}, Config file has: {cfg.genome_config.num_inputs}. PLEASE UPDATE THE CONFIG FILE and restart."); return

    current_eval_window_start_index = ATTENTION_SEQUENCE_LENGTH -1 
    max_portfolio_ever_achieved_in_training = INITIAL_STARTING_CAPITAL
    best_record_breaker_details = { "genome_obj": None, "window_fitness": -float('inf'), "portfolio_achieved_on_full_train": -float('inf')}
    current_eval_window_raw_data_for_plotting = None

    pop = neat.Population(cfg)
    gen_reporter = GenerationReporter(
        plot_interval=PLOT_BEST_OF_GENERATION_EVERY,
        train_data_scaled_for_reporter_features=train_data_scaled_np_global, 
        train_data_raw_for_reporter_prices=train_data_raw_prices_global,     
        neat_config=cfg,
        initial_capital=INITIAL_STARTING_CAPITAL,
        trading_fee=TRADING_FEE_PERCENT
    )
    pop.add_reporter(gen_reporter)
    pop.add_reporter(neat.StdOutReporter(True))
    stats = neat.StatisticsReporter(); pop.add_reporter(stats)

    checkpointer = neat.Checkpointer(generation_interval=10, time_interval_seconds=3600, 
                                     filename_prefix=f'neat_outputs/neat-checkpoint-{TICKER}-')
    pop.add_reporter(checkpointer)


    print("\nStarting NEAT evolution with Attention & New Fitness...");
    winner = None 
    try:
        winner = pop.run(eval_genomes, N_GENERATIONS) 
    except KeyboardInterrupt:
        print("Evolution run interrupted by user.")
    except Exception as e:
        print(f"ERROR during NEAT evolution: {e}")
        import traceback; traceback.print_exc()
    finally:
        print("Evolution finished or interrupted. Plotting final generation metrics...")
        gen_reporter.plot_generational_metrics()


    winner_to_evaluate = None; winner_fitness_note = -float('inf'); final_report_source = "None"
    
    if winner is not None and hasattr(winner, 'fitness') and winner.fitness is not None:
        if winner.fitness >= cfg.fitness_threshold: 
            winner_to_evaluate = winner
            winner_fitness_note = winner.fitness
            final_report_source = "NEAT pop.run() returned winner (threshold met)"
            print(f"Winner from pop.run (threshold met): ID {winner.key}, Fitness: {winner.fitness:.2f}")

    if winner_to_evaluate is None:
        grb_genome = best_record_breaker_details["genome_obj"]
        grb_portfolio_full_train = best_record_breaker_details["portfolio_achieved_on_full_train"]
        grb_window_fitness = best_record_breaker_details["window_fitness"]
        
        cond1_grb_profitable = grb_genome and grb_portfolio_full_train > (INITIAL_STARTING_CAPITAL * 1.001) 
        cond2_grb_decent_window_fit = grb_window_fitness is not None and grb_window_fitness > - (FITNESS_SCALER * 1.0) 

        if cond1_grb_profitable and cond2_grb_decent_window_fit:
            winner_to_evaluate = grb_genome; winner_fitness_note = grb_window_fitness
            final_report_source = "Global Record Breaker (Min Profit & Decent WindowFit)"
        elif gen_reporter.neat_overall_best_genome_obj and gen_reporter.neat_best_fitness_so_far > -float('inf'):
            winner_to_evaluate = gen_reporter.neat_overall_best_genome_obj; winner_fitness_note = gen_reporter.neat_best_fitness_so_far
            final_report_source = "Reporter's Overall Best (by window fitness)"
        else: 
            best_fitness_fallback = -float('inf'); temp_winner_fallback = None
            # Check if stats object and population exist before trying to access them
            if stats and hasattr(pop, 'population') and pop.population:
                all_genomes_final_pop = list(pop.population.values())
                for g_val_iter in all_genomes_final_pop:
                    if hasattr(g_val_iter, 'fitness') and g_val_iter.fitness is not None and g_val_iter.fitness > best_fitness_fallback:
                        best_fitness_fallback = g_val_iter.fitness; temp_winner_fallback = g_val_iter
            
            if temp_winner_fallback: 
                winner_to_evaluate = temp_winner_fallback
                winner_fitness_note = best_fitness_fallback
                final_report_source = "NEAT Population Fallback (End of Run)"
            elif stats and hasattr(stats, 'best_genome') and callable(stats.best_genome): 
                 best_from_stats = stats.best_genome()
                 if best_from_stats: 
                    winner_to_evaluate = best_from_stats
                    winner_fitness_note = winner_to_evaluate.fitness if hasattr(winner_to_evaluate, 'fitness') and winner_to_evaluate.fitness is not None else "N/A"
                    final_report_source = "NEAT Statistics Reporter Best Genome"
                 else:
                    print("CRITICAL: No winner genome could be determined after evolution (stats.best_genome() was None).")
                    # return # Removed return to allow plotting if this path is hit after interruption
            else:
                print("CRITICAL: No winner genome could be determined after evolution (no fallback options worked).")
                # return # Removed return to allow plotting if this path is hit after interruption
    
    fitness_display_string = f"{winner_fitness_note:.2f}" if isinstance(winner_fitness_note, (int, float)) and not (np.isinf(winner_fitness_note) or np.isnan(winner_fitness_note)) else str(winner_fitness_note)
    print(f"\nOverall Best Genome Selected ({final_report_source}): ID {winner_to_evaluate.key if winner_to_evaluate else 'N/A'}, Fitness Context: {fitness_display_string}")

    if winner_to_evaluate:
        output_dir = "neat_outputs"
        os.makedirs(output_dir, exist_ok=True)
        winner_path = os.path.join(output_dir, f"winner_genome_attention_{TICKER}.pkl")
        with open(winner_path, "wb") as f: pickle.dump(winner_to_evaluate, f)
        print(f"Saved overall best genome to {winner_path}")

    if winner_to_evaluate:
        print("\n--- Final Evaluation of Selected Overall Best Genome ---")
        if train_data_scaled_np_global is not None and len(train_data_scaled_np_global) >= ATTENTION_SEQUENCE_LENGTH and \
           train_data_raw_prices_global is not None and not train_data_raw_prices_global.empty:
            run_simulation_and_plot(winner_to_evaluate, cfg,
                                    train_data_scaled_np_global, 
                                    train_data_raw_prices_global,    
                                    "Selected Best Genome on Training Data (Full)")
        else:
            print("Skipping final training data evaluation due to insufficient data length for attention.")
        
        if val_scaled_np_features is not None and len(val_scaled_np_features) >= ATTENTION_SEQUENCE_LENGTH and \
           val_data_raw_prices_df is not None and not val_data_raw_prices_df.empty:
            run_simulation_and_plot(winner_to_evaluate, cfg,
                                    val_scaled_np_features,   
                                    val_data_raw_prices_df,   
                                    "Selected Best Genome on Validation Data",
                                    is_validation_run=True)
        elif not val_feature_df.empty : 
             print("WARNING: Validation data was present but resulted in empty/short scaled features or raw prices for final eval. Skipping validation plot.")
        else: 
             print("INFO: No validation data was available or it was too short for final evaluation.")
    else:
        print("No best genome found to evaluate after evolution.")

if __name__ == "__main__":
    os.makedirs("neat_outputs", exist_ok=True)

    if not os.path.exists(CONFIG_FILE_PATH): 
        print(f"Config file not found: {CONFIG_FILE_PATH}")
    else: 
        run_neat_trader(CONFIG_FILE_PATH)
