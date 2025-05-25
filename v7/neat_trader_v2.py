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
import math # For pow

from trader import Trader
from functionalities import plot_backtest_results, plot_generational_performance

# --- Configuration ---
TICKER = "BTC-USD"
DATA_PERIOD = "7d" # yfinance limit for 1m interval
DATA_INTERVAL = "1m"
TRAIN_DAYS = 5     # Uses 5 days for training, rest for validation (approx 2 days)
INITIAL_STARTING_CAPITAL = 200.0
INITIAL_STARTING_HOLDINGS = 0.0
N_LAGS = 5
CONFIG_FILE_PATH = "./config-feedforward"
N_GENERATIONS = 50 # Adjusted for quicker example runs, you likely want more
MAX_EXPECTED_CREDIT = INITIAL_STARTING_CAPITAL * 3
MAX_EXPECTED_HOLDINGS_VALUE = INITIAL_STARTING_CAPITAL * 3
PLOT_BEST_OF_GENERATION_EVERY = 10
PLOT_WINDOW_PERFORMANCE = True
TRADING_FEE_PERCENT = 0.001
EVAL_WINDOW_SIZE_MINUTES = 4 * 60 # 4 hours

_COLUMN_NAMES_RESOLVED_FOR_CURRENT_FETCH = False
COL_OPEN, COL_HIGH, COL_LOW, COL_CLOSE, COL_VOLUME = 'Open', 'High', 'Low', 'Close', 'Volume'
current_eval_window_start_index = 0
max_portfolio_ever_achieved_in_training = INITIAL_STARTING_CAPITAL
best_record_breaker_details = {
    "genome_obj": None, "window_fitness": -float('inf'), "portfolio_achieved_on_full_train": -float('inf')
}
current_eval_window_raw_data_for_plotting = None

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
        self.metrics_history = {
            "Best Fitness (Window)": [],
            "Max Portfolio Ever ($) (Training Record)": [],
            "Best Gen Genome's Portfolio ($) (Full Train Sim)": [],
            "Best Gen Genome's Net Profit ($) (Full Train Sim)": [],
            "Best Gen Genome's Credit ($) (Full Train Sim)": [],
            "Best Gen Genome's Total Trades (Full Train Sim)": [],
            "Best Gen Genome's Buys (Full Train Sim)": [],
            "Best Gen Genome's Sells (Full Train Sim)": [],
            "Best Gen Genome's Fees Paid ($) (Full Train Sim)": []
        }

    def start_generation(self, generation):
        global current_eval_window_start_index, current_eval_window_raw_data_for_plotting
        self.generation_count = generation
        full_train_len = len(self.train_data_scaled_for_reporter)
        window_size_for_eval = EVAL_WINDOW_SIZE_MINUTES

        if window_size_for_eval >= full_train_len or window_size_for_eval <= 10:
            window_size_for_eval = max(10, full_train_len // 3)

        if full_train_len > window_size_for_eval:
            advance_step = max(1, int(window_size_for_eval * 0.80))
            total_advanceable_range = max(0, full_train_len - window_size_for_eval)
            if total_advanceable_range > 0 :
                current_eval_window_start_index = (generation * advance_step) % total_advanceable_range
            else:
                current_eval_window_start_index = 0
        else:
            current_eval_window_start_index = 0

        current_eval_window_start_index = max(0, current_eval_window_start_index)
        current_eval_window_start_index = min(current_eval_window_start_index, max(0, full_train_len - window_size_for_eval))

        eval_win_start_idx = current_eval_window_start_index
        eval_win_end_idx = min(eval_win_start_idx + window_size_for_eval, full_train_len)
        
        current_eval_window_raw_data_for_plotting = self.train_data_raw_for_reporter.iloc[eval_win_start_idx:eval_win_end_idx]

        if len(self.train_data_raw_for_reporter) > eval_win_start_idx and \
           len(self.train_data_raw_for_reporter) >= eval_win_end_idx and \
           eval_win_end_idx > eval_win_start_idx:
            start_date_str = self.train_data_raw_for_reporter.index[eval_win_start_idx].strftime('%Y-%m-%d %H:%M')
            end_date_str = self.train_data_raw_for_reporter.index[eval_win_end_idx-1].strftime('%Y-%m-%d %H:%M')
            print(f"  Gen {generation}: eval_genomes window: idx {eval_win_start_idx}-{eval_win_end_idx-1} ({start_date_str} to {end_date_str}), size={eval_win_end_idx - eval_win_start_idx}")
        else:
            print(f"  Gen {generation}: eval_genomes window: idx {eval_win_start_idx}-{eval_win_end_idx-1}, size={eval_win_end_idx - eval_win_start_idx}.")


    def end_generation(self, config, population_genomes_dict, species_set):
        self._actual_end_of_generation_logic(config, population_genomes_dict, species_set)

    def _actual_end_of_generation_logic(self, config, population_genomes_dict, species_set_object):
        global max_portfolio_ever_achieved_in_training, best_record_breaker_details, current_eval_window_raw_data_for_plotting
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
        self.metrics_history["Max Portfolio Ever ($) (Training Record)"].append(max_portfolio_ever_achieved_in_training)

        if best_genome_this_gen_by_window_fitness:
            self.metrics_history["Best Fitness (Window)"].append(current_gen_max_window_fitness)
            
            net_full_sim = neat.nn.FeedForwardNetwork.create(best_genome_this_gen_by_window_fitness, self.neat_config)
            rep_trader_full_sim = Trader(self.initial_capital, INITIAL_STARTING_HOLDINGS, trading_fee_percent=self.trading_fee)
            buys_rep_sim, sells_rep_sim = 0,0
            for i in range(len(self.train_data_scaled_for_reporter)):
                if not rep_trader_full_sim.is_alive: break
                feat = self.train_data_scaled_for_reporter[i]
                price = self.train_data_raw_for_reporter.iloc[i][COL_CLOSE]
                ts = self.train_data_raw_for_reporter.index[i]
                state = rep_trader_full_sim.get_state_for_nn(price, MAX_EXPECTED_CREDIT, MAX_EXPECTED_HOLDINGS_VALUE)
                nn_in = np.concatenate((feat, state)); action, amount = net_full_sim.activate(nn_in)
                if action > 0.55:
                    if rep_trader_full_sim.buy(amount * rep_trader_full_sim.credit, price, ts): buys_rep_sim +=1
                elif action < 0.45:
                    if rep_trader_full_sim.sell(amount * rep_trader_full_sim.holdings_shares, price, ts): sells_rep_sim +=1
                rep_trader_full_sim.update_history(ts, price)
            final_pf_rep = rep_trader_full_sim.get_portfolio_value(self.train_data_raw_for_reporter.iloc[-1][COL_CLOSE])
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
            
            # --- ADDED: Projected returns for this generation's best on full training data ---
            if not self.train_data_raw_for_reporter.empty and self.initial_capital > 1e-6:
                train_start_time = self.train_data_raw_for_reporter.index.min()
                train_end_time = self.train_data_raw_for_reporter.index.max()
                train_duration_seconds = (train_end_time - train_start_time).total_seconds()

                # Only project if training duration is somewhat meaningful (e.g., > 1 hour for training data)
                if train_duration_seconds > 3600: 
                    train_profit_ratio_for_projection = net_profit_rep / self.initial_capital
                    
                    seconds_in_week = 7 * 24 * 3600
                    seconds_in_month = 30 * 24 * 3600 # Approximate month

                    periods_in_week = seconds_in_week / train_duration_seconds
                    periods_in_month = seconds_in_month / train_duration_seconds
                    
                    # Projected returns (compounded)
                    # (1 + profit_ratio) is the growth factor over the period
                    projected_weekly_return_pct_train = (math.pow(1 + train_profit_ratio_for_projection, periods_in_week) - 1) * 100
                    projected_monthly_return_pct_train = (math.pow(1 + train_profit_ratio_for_projection, periods_in_month) - 1) * 100
                    
                    print(f"      REPORTER Projected (Full Train Sim, Gen {self.generation_count} Best, COMPOUNDED): "
                          f"Weekly: {projected_weekly_return_pct_train:.2f}%, Monthly: {projected_monthly_return_pct_train:.2f}%")
                    if abs(projected_weekly_return_pct_train) > 1000 or abs(projected_monthly_return_pct_train) > 5000:
                        print("      WARNING (Full Train Proj): Projections very high/low. Interpret with caution (in-sample performance).")
                # else:
                #     print(f"      REPORTER: Full training data period ({pd.to_timedelta(train_duration_seconds, unit='s')}) too short for weekly/monthly projection.")
            # --- END ADDED SECTION ---


            if self.plot_interval > 0 and (self.generation_count + 1) % self.plot_interval == 0:
                print(f"      REPORTER: Plotting for Gen {self.generation_count}...")
                run_simulation_and_plot(best_genome_this_gen_by_window_fitness, self.neat_config,
                                        self.train_data_scaled_for_reporter, self.train_data_raw_for_reporter,
                                        title_prefix=f"Gen {self.generation_count} Best (Full Train Sim)")
                
                if PLOT_WINDOW_PERFORMANCE and current_eval_window_raw_data_for_plotting is not None and not current_eval_window_raw_data_for_plotting.empty:
                    eval_win_start_idx = current_eval_window_start_index
                    window_size_for_eval_plot = len(current_eval_window_raw_data_for_plotting)
                    current_window_scaled_data_for_plot = self.train_data_scaled_for_reporter[eval_win_start_idx : eval_win_start_idx + window_size_for_eval_plot]
                    
                    if len(current_window_scaled_data_for_plot) > 0 and len(current_window_scaled_data_for_plot) == len(current_eval_window_raw_data_for_plotting) :
                        run_simulation_and_plot(best_genome_this_gen_by_window_fitness, self.neat_config,
                                                current_window_scaled_data_for_plot,
                                                current_eval_window_raw_data_for_plotting,
                                                title_prefix=f"Gen {self.generation_count} Best (Actual Eval Window)")
                    else:
                        print(f"      REPORTER: Skipped plotting eval window for Gen {self.generation_count} due to data mismatch/empty. Scaled len: {len(current_window_scaled_data_for_plot)}, Raw len: {len(current_eval_window_raw_data_for_plotting)}")
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
        if best_found_by_neat and best_found_by_neat.fitness is not None and \
           (self.neat_overall_best_genome_obj is None or best_found_by_neat.fitness > self.neat_best_fitness_so_far):
            self.neat_best_fitness_so_far = best_found_by_neat.fitness
            self.neat_overall_best_genome_obj = best_found_by_neat
    def info(self, msg): pass
    def plot_generational_metrics(self):
        metrics_to_plot = {k: v for k, v in self.metrics_history.items() if "Full Train Sim" in k or "Best Fitness" in k or "Max Portfolio Ever" in k}
        if self.generations_list and any(len(v) > 0 for v in metrics_to_plot.values()):
            valid_metrics_history = {k: v for k, v in metrics_to_plot.items() if not all(np.isnan(val) if isinstance(val, float) else False for val in v)}
            if valid_metrics_history:
                 plot_generational_performance(self.generations_list, valid_metrics_history, title="Key Metrics Per Generation")
            else: print("No valid generational metrics data to plot (all NaNs).")
        else: print("No generational data accumulated to plot.")

# --- Helper Functions (Unchanged) ---
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
        if not matched_pattern: print(f"WARNING: Unhandled yfinance MultiIndex. Defaulting: {df_columns.tolist()}")
    elif not all(col_str in df_columns for col_str in [_COL_OPEN_temp, _COL_HIGH_temp, _COL_LOW_temp, _COL_CLOSE_temp, _COL_VOLUME_temp]):
        print(f"WARNING: Standard column names not found. Defaulting: {df_columns.tolist()}")
    else: matched_pattern = True
    COL_OPEN, COL_HIGH, COL_LOW, COL_CLOSE, COL_VOLUME = _COL_OPEN_temp, _COL_HIGH_temp, _COL_LOW_temp, _COL_CLOSE_temp, _COL_VOLUME_temp
    _COLUMN_NAMES_RESOLVED_FOR_CURRENT_FETCH = True

def fetch_data(ticker, period, interval):
    global _COLUMN_NAMES_RESOLVED_FOR_CURRENT_FETCH; _COLUMN_NAMES_RESOLVED_FOR_CURRENT_FETCH = False
    print(f"Fetching data for {ticker} ({period}, {interval})...")
    if interval == "1m" and pd.to_timedelta(period) > pd.to_timedelta("7d"):
        print(f"WARNING: yfinance may restrict '1m' interval data to 7 days. Requested period '{period}' might be truncated.")
    
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

def calculate_technical_indicators(df):
    print("Calculating TAs..."); df_for_ta = df.copy()
    if not all(col in df_for_ta.columns for col in [COL_OPEN, COL_HIGH, COL_LOW, COL_CLOSE, COL_VOLUME]):
        raise ValueError("One or more required OHLCV columns are missing before TA calculation.")
    df_ta = ta.add_all_ta_features(df_for_ta, COL_OPEN, COL_HIGH, COL_LOW, COL_CLOSE, COL_VOLUME, fillna=True)
    prefs = ['trend_macd', 'trend_macd_signal', 'trend_macd_diff', 'momentum_rsi', 'momentum_stoch_rsi',
             'volatility_atr', 'trend_ema_fast', 'trend_ema_slow', 'volume_obv', 'others_cr']
    actual_names, simple_names = [], []
    for p in prefs:
        for col_name in df_ta.columns:
            str_col, is_tuple = str(col_name), isinstance(col_name, tuple)
            if str_col.lower().startswith(p.lower()) and col_name not in actual_names:
                 actual_names.append(col_name); simple_names.append(p); break
            elif is_tuple and col_name[0].lower().startswith(p.lower()) and col_name not in actual_names:
                 actual_names.append(col_name); simple_names.append(p); break
    out_df = pd.DataFrame(index=df_ta.index)
    out_df['Close'] = df[COL_CLOSE].reindex(out_df.index) 

    selected_ta_cols_for_output = []
    for act, simp in zip(actual_names, simple_names):
        unique_simp = simp; ctr = 1
        while unique_simp in out_df.columns: unique_simp = f"{simp}_{ctr}"; ctr+=1
        out_df[unique_simp] = df_ta[act].values
        selected_ta_cols_for_output.append(unique_simp)
    print(f"Selected TA features for model: {selected_ta_cols_for_output}"); return out_df

def add_lagged_features(df, n_lags=1):
    print(f"Adding {n_lags} lags..."); lagged_df = df.copy()
    features_to_lag = [col for col in df.columns if col != 'Close']
    
    if not features_to_lag:
        print("Warning: No features to lag (is 'Close' the only column?). Skipping lag addition.")
        return lagged_df

    for lag in range(1, n_lags + 1):
        shifted = df[features_to_lag].shift(lag)
        for col in features_to_lag:
            lagged_df[f'{col}_lag{lag}'] = shifted[col]
    
    lagged_df.dropna(inplace=True)
    if lagged_df.empty:
        print("WARNING: DataFrame became empty after adding lags and dropping NA. Original data might be too short for N_LAGS.")
    print(f"Shape after lags/dropna: {lagged_df.shape}"); return lagged_df

def split_data_by_days(df, train_days_count):
    print(f"Splitting: {train_days_count} train days..."); df = df.sort_index()
    unique_days = sorted(df.index.normalize().unique())
    if len(unique_days) == 0: print("ERROR split_data: No unique days found."); return None, None
    
    if train_days_count >= len(unique_days):
        print(f"ERROR: train_days_count ({train_days_count}) is >= total unique days ({len(unique_days)}). Cannot create validation set.")
        if len(unique_days) > 1:
            train_days_count = len(unique_days) - 1
            print(f"Adjusted train_days_count to {train_days_count} to allow for a minimal validation set.")
        else:
            return None, None

    if train_days_count == 0:
        print("ERROR: train_days_count is 0. Cannot create training set.")
        return None, None

    split_date_boundary = unique_days[train_days_count - 1]
    train_df = df[df.index.normalize() <= split_date_boundary]
    validation_df = df[df.index.normalize() > split_date_boundary]
    
    print(f"Train: {train_df.shape}, {train_df.index.min()} to {train_df.index.max() if not train_df.empty else 'N/A'}")
    print(f"Valid: {validation_df.shape}, {validation_df.index.min()} to {validation_df.index.max() if not validation_df.empty else 'N/A'}")
    
    if train_df.empty: print("ERROR split_data: Training dataframe is empty."); return None, None
    if validation_df.empty: print("WARNING split_data: Validation dataframe is empty. Consider increasing DATA_PERIOD or reducing TRAIN_DAYS.");
    
    return train_df, validation_df

def normalize_data(train_df, val_df):
    print("Normalizing...");
    train_features = train_df.drop(columns=['Close'], errors='ignore')
    val_features = val_df.drop(columns=['Close'], errors='ignore')

    scaler = MinMaxScaler()
    train_features_scaled_np = scaler.fit_transform(train_features)
    
    if not val_features.empty:
        val_features_scaled_np = scaler.transform(val_features)
    else:
        val_features_scaled_np = np.array([])

    train_scaled = pd.DataFrame(train_features_scaled_np, columns=train_features.columns, index=train_features.index)
    
    if not val_features.empty:
        val_scaled = pd.DataFrame(val_features_scaled_np, columns=val_features.columns, index=val_features.index)
    else:
        val_scaled = pd.DataFrame(columns=train_features.columns)

    if 'Close' in train_df.columns:
        train_scaled['Close'] = train_df['Close']
    if 'Close' in val_df.columns and not val_df.empty:
        val_scaled['Close'] = val_df['Close']
    elif 'Close' in val_df.columns and val_df.empty:
         val_scaled['Close'] = pd.Series(dtype=float)

    return train_scaled, val_scaled, scaler


train_data_scaled_np_global = None
train_data_raw_prices_global = None

# --- FITNESS FUNCTION ---
def eval_genomes(genomes, config):
    global train_data_scaled_np_global, train_data_raw_prices_global, current_eval_window_start_index, \
           max_portfolio_ever_achieved_in_training

    if train_data_scaled_np_global is None or train_data_raw_prices_global is None:
        raise RuntimeError("Global training data (scaled features or raw prices) not set.")

    full_train_len = len(train_data_scaled_np_global)
    window_size_for_eval = EVAL_WINDOW_SIZE_MINUTES
    if window_size_for_eval >= full_train_len or window_size_for_eval <= 10 :
        window_size_for_eval = max(10, full_train_len // 2)

    start_idx_for_this_gen_eval = current_eval_window_start_index
    end_idx_for_this_gen_eval = min(start_idx_for_this_gen_eval + window_size_for_eval, full_train_len)

    actual_window_len = end_idx_for_this_gen_eval - start_idx_for_this_gen_eval
    if actual_window_len < min(10, window_size_for_eval / 2) :
        if full_train_len > window_size_for_eval :
             start_idx_for_this_gen_eval = max(0, full_train_len - window_size_for_eval)
        else:
            start_idx_for_this_gen_eval = 0
        end_idx_for_this_gen_eval = min(start_idx_for_this_gen_eval + window_size_for_eval, full_train_len)
        actual_window_len = end_idx_for_this_gen_eval - start_idx_for_this_gen_eval

    if actual_window_len < 10:
        for _, genome in genomes: genome.fitness = -1e12
        return

    current_eval_scaled_features = train_data_scaled_np_global[start_idx_for_this_gen_eval : end_idx_for_this_gen_eval]
    current_eval_raw_prices_df = train_data_raw_prices_global.iloc[start_idx_for_this_gen_eval : end_idx_for_this_gen_eval]

    PROFIT_TARGET_BREAKEVEN = 0.002; PROFIT_TARGET_MODEST = 0.01; PROFIT_TARGET_GOOD = 0.025; PROFIT_TARGET_EXCEPTIONAL = 0.05
    FITNESS_PROFIT_SCALER = INITIAL_STARTING_CAPITAL * 10.0; FITNESS_LOSS_SCALER = INITIAL_STARTING_CAPITAL * 15.0
    RECORD_BREAK_BASE_BONUS = INITIAL_STARTING_CAPITAL * 50.0; RECORD_BREAK_IMPROVEMENT_SCALER = INITIAL_STARTING_CAPITAL * 200.0
    MIN_TRADES_THRESHOLD = 2; REALIZED_PROFIT_BONUS_MULTIPLIER = 0.15; BEAT_BUY_HOLD_PENALTY_FACTOR = 0.1
    INACTION_PENALTY_FACTOR_BH_NEGATIVE = 0.3
    AMOUNT_NUDGE_FACTOR = 0.05

    for genome_id, genome in genomes:
        net = neat.nn.FeedForwardNetwork.create(genome, config)
        trader = Trader(INITIAL_STARTING_CAPITAL, INITIAL_STARTING_HOLDINGS, trading_fee_percent=TRADING_FEE_PERCENT)
        genome.fitness = 0.0
        num_buys, num_sells = 0,0
        window_peak_portfolio = INITIAL_STARTING_CAPITAL
        
        buy_hold_profit_ratio = 0.0
        if len(current_eval_raw_prices_df) > 1:
            initial_price_window = current_eval_raw_prices_df.iloc[0][COL_CLOSE]
            final_price_window = current_eval_raw_prices_df.iloc[-1][COL_CLOSE]
            if initial_price_window > 1e-6:
                bh_shares = (INITIAL_STARTING_CAPITAL / (1 + TRADING_FEE_PERCENT)) / initial_price_window
                bh_final_value_gross = bh_shares * final_price_window
                bh_final_value_net = bh_final_value_gross * (1 - TRADING_FEE_PERCENT)
                buy_hold_profit_ratio = (bh_final_value_net - INITIAL_STARTING_CAPITAL) / INITIAL_STARTING_CAPITAL

        for i in range(len(current_eval_scaled_features)):
            if not trader.is_alive: break
            features_for_nn = current_eval_scaled_features[i]
            price = current_eval_raw_prices_df.iloc[i][COL_CLOSE]
            ts = current_eval_raw_prices_df.index[i]
            
            state = trader.get_state_for_nn(price, MAX_EXPECTED_CREDIT, MAX_EXPECTED_HOLDINGS_VALUE)
            nn_in = np.concatenate((features_for_nn, state));
            action_raw, amount_raw = net.activate(nn_in)
            
            amount_to_use = amount_raw
            if action_raw > 0.55 or action_raw < 0.45:
                amount_to_use = np.clip(amount_raw + AMOUNT_NUDGE_FACTOR, 0.0, 1.0)

            if action_raw > 0.55:
                if trader.buy(amount_to_use * trader.credit, price, ts): num_buys+=1
            elif action_raw < 0.45:
                if trader.sell(amount_to_use * trader.holdings_shares, price, ts): num_sells+=1
            trader.update_history(ts, price)
            if trader.get_portfolio_value(price) > window_peak_portfolio:
                window_peak_portfolio = trader.get_portfolio_value(price)
        
        final_portfolio = trader.get_portfolio_value(current_eval_raw_prices_df.iloc[-1][COL_CLOSE])
        final_credit = trader.credit
        total_trades = num_buys + num_sells
        net_profit_abs = final_portfolio - INITIAL_STARTING_CAPITAL
        profit_ratio = net_profit_abs / INITIAL_STARTING_CAPITAL if INITIAL_STARTING_CAPITAL > 0 else 0.0
        
        fitness_score = 0.0

        if final_portfolio < INITIAL_STARTING_CAPITAL * 0.65:
            fitness_score = -FITNESS_PROFIT_SCALER * 1000.0
            genome.fitness = fitness_score; continue
        if not trader.is_alive or (final_credit < INITIAL_STARTING_CAPITAL * 0.05 and final_portfolio < INITIAL_STARTING_CAPITAL * 0.8):
            fitness_score = -FITNESS_PROFIT_SCALER * 500.0
            genome.fitness = fitness_score; continue

        if profit_ratio >= PROFIT_TARGET_EXCEPTIONAL:
            fitness_score = (profit_ratio ** 2.0) * FITNESS_PROFIT_SCALER * 5.0
        elif profit_ratio >= PROFIT_TARGET_GOOD:
            fitness_score = (profit_ratio ** 1.8) * FITNESS_PROFIT_SCALER * 3.0
        elif profit_ratio >= PROFIT_TARGET_MODEST:
            fitness_score = (profit_ratio ** 1.5) * FITNESS_PROFIT_SCALER * 1.5
        elif profit_ratio > PROFIT_TARGET_BREAKEVEN:
            fitness_score = profit_ratio * FITNESS_PROFIT_SCALER * 0.5
        elif profit_ratio > -0.01:
            fitness_score = profit_ratio * FITNESS_PROFIT_SCALER * 0.1
        else:
            fitness_score = - (abs(profit_ratio) ** 1.5) * FITNESS_LOSS_SCALER * 2.0

        if window_peak_portfolio > max_portfolio_ever_achieved_in_training and profit_ratio >= PROFIT_TARGET_MODEST :
            improvement_over_record_abs = window_peak_portfolio - max_portfolio_ever_achieved_in_training
            fitness_score += (improvement_over_record_abs / INITIAL_STARTING_CAPITAL) * RECORD_BREAK_IMPROVEMENT_SCALER
            fitness_score += RECORD_BREAK_BASE_BONUS
        
        if trader.realized_gains_this_evaluation > (INITIAL_STARTING_CAPITAL * 0.005):
             fitness_score += (trader.realized_gains_this_evaluation / INITIAL_STARTING_CAPITAL) * FITNESS_PROFIT_SCALER * REALIZED_PROFIT_BONUS_MULTIPLIER

        if total_trades < MIN_TRADES_THRESHOLD :
            if profit_ratio < PROFIT_TARGET_MODEST:
                fitness_score -= FITNESS_PROFIT_SCALER * 0.2
            if buy_hold_profit_ratio < -0.005:
                 fitness_score -= FITNESS_PROFIT_SCALER * INACTION_PENALTY_FACTOR_BH_NEGATIVE

        max_trades_for_churn = max(3, len(current_eval_scaled_features) // 20)
        if total_trades > max_trades_for_churn and profit_ratio < PROFIT_TARGET_GOOD:
            excess_churn = total_trades - max_trades_for_churn
            fitness_score -= excess_churn * (FITNESS_PROFIT_SCALER * 0.001)
            
        if total_trades > 0 and profit_ratio < buy_hold_profit_ratio and buy_hold_profit_ratio > PROFIT_TARGET_BREAKEVEN :
            difference_from_bh = buy_hold_profit_ratio - profit_ratio
            fitness_score -= difference_from_bh * FITNESS_PROFIT_SCALER * BEAT_BUY_HOLD_PENALTY_FACTOR

        genome.fitness = fitness_score
        if np.isnan(genome.fitness) or np.isinf(genome.fitness): genome.fitness = -1e12


# --- run_simulation_and_plot ---
def run_simulation_and_plot(genome, config, data_scaled_features_np, data_raw_prices_df, title_prefix, is_validation_run=False):
    if data_raw_prices_df.empty or COL_CLOSE not in data_raw_prices_df.columns:
        print(f"Plot Info: Raw price data empty or missing '{COL_CLOSE}' for '{title_prefix}'. Skipping.")
        return
    if data_scaled_features_np is None or len(data_scaled_features_np) == 0:
        print(f"Simulation Info: No scaled feature data for '{title_prefix}'. Cannot run simulation.")
        plot_backtest_results(data_raw_prices_df, [], [], f"{title_prefix} Results for {TICKER} (No Sim)", COL_CLOSE)
        return
    if len(data_scaled_features_np) != len(data_raw_prices_df):
        print(f"Data Mismatch Error for '{title_prefix}': Scaled features length ({len(data_scaled_features_np)}) != Raw prices length ({len(data_raw_prices_df)}). Skipping.")
        return

    print(f"\n--- {title_prefix} Evaluation ---")
    net = neat.nn.FeedForwardNetwork.create(genome, config)
    tr = Trader(INITIAL_STARTING_CAPITAL, INITIAL_STARTING_HOLDINGS, trading_fee_percent=TRADING_FEE_PERCENT)
    buys,sells = 0,0
    i_sim = 0
    final_val = tr.credit

    buy_threshold_eff = 0.55
    sell_threshold_eff = 0.45
    if is_validation_run:
        # print("   (Validation Run Diagnostic: Using wider action thresholds: Buy > 0.50, Sell < 0.50)") # Example, not active
        pass

    print_debug_limit = 0 # Set to > 0 for step-by-step debug prints
    
    for i_sim_loop in range(len(data_scaled_features_np)):
        i_sim = i_sim_loop
        if not tr.is_alive: break
        
        current_features = data_scaled_features_np[i_sim]
        current_price = data_raw_prices_df.iloc[i_sim][COL_CLOSE]
        current_ts = data_raw_prices_df.index[i_sim]

        trader_state_inputs = tr.get_state_for_nn(current_price, MAX_EXPECTED_CREDIT, MAX_EXPECTED_HOLDINGS_VALUE)
        nn_in = np.concatenate((current_features, trader_state_inputs))
        
        action_raw, amount_raw = net.activate(nn_in)
        amount_to_use = np.clip(amount_raw, 0.01, 1.0)

        if (is_validation_run or print_debug_limit > 0) and i_sim < print_debug_limit: # Modified for general debug
            print(f"  Step {i_sim}: Price={current_price:.2f}, Timestamp={current_ts}")
            print(f"    Trader State: CreditNorm={trader_state_inputs[0]:.3f}, HoldValNorm={trader_state_inputs[1]:.3f}, UPL%Norm={trader_state_inputs[2]:.3f}")
            print(f"    NN Raw Out: Action={action_raw:.4f}, Amount={amount_raw:.4f} (Using Amount={amount_to_use:.4f})")

        trade_made = False
        if action_raw > buy_threshold_eff:
            if tr.buy(amount_to_use * tr.credit, current_price, current_ts):
                buys+=1; trade_made = True
                if (is_validation_run or print_debug_limit > 0) and i_sim < print_debug_limit: print(f"    ACTION: BUY executed. Shares: {tr.trade_log[-1]['shares']:.4f}")
        elif action_raw < sell_threshold_eff:
            if tr.sell(amount_to_use * tr.holdings_shares, current_price, current_ts):
                sells+=1; trade_made = True
                if (is_validation_run or print_debug_limit > 0) and i_sim < print_debug_limit: print(f"    ACTION: SELL executed. Shares: {tr.trade_log[-1]['shares']:.4f}")
        
        if (is_validation_run or print_debug_limit > 0) and i_sim < print_debug_limit and not trade_made:
            print(f"    ACTION: HOLD (Action {action_raw:.4f} not crossing thresholds {buy_threshold_eff:.2f}/{sell_threshold_eff:.2f})")

        tr.update_history(current_ts, current_price)
    
    if not data_raw_prices_df.empty:
        valid_last_idx = min(i_sim, len(data_raw_prices_df) - 1)
        if valid_last_idx >= 0 :
             final_val = tr.get_portfolio_value(data_raw_prices_df.iloc[valid_last_idx][COL_CLOSE])

    print(f"{title_prefix} - Initial: ${INITIAL_STARTING_CAPITAL:.2f}, Final Portfolio: ${final_val:.2f}, Final Credit: ${tr.credit:.2f}")
    print(f"Trades Logged: {len(tr.trade_log)} (Sim Buys: {buys}, Sells: {sells}), Realized PnL: ${tr.realized_gains_this_evaluation:.2f}, Fees Paid: ${tr.total_fees_paid:.2f}")

    profit_abs = final_val - INITIAL_STARTING_CAPITAL
    profit_percentage_on_segment = 0.0
    if INITIAL_STARTING_CAPITAL > 1e-6:
        profit_percentage_on_segment = (profit_abs / INITIAL_STARTING_CAPITAL) * 100
        print(f"Profit/Loss (Portfolio on this data segment): {profit_percentage_on_segment:.2f}%")
    else:
        print(f"Profit/Loss (Portfolio): N/A (Initial capital too small)")

    # --- ADDED: Timeframe and Projections for Validation Run ---
    if is_validation_run and not data_raw_prices_df.empty and INITIAL_STARTING_CAPITAL > 1e-6:
        start_time = data_raw_prices_df.index.min()
        end_time = data_raw_prices_df.index.max()
        duration_seconds = (end_time - start_time).total_seconds()
        
        print(f"  Validation Data Timeframe: {start_time.strftime('%Y-%m-%d %H:%M')} to {end_time.strftime('%Y-%m-%d %H:%M')} (Duration: {pd.to_timedelta(duration_seconds, unit='s')})")

        if duration_seconds > 60 * 5: # Only project if duration is somewhat meaningful (e.g., > 5 minutes for validation)
            profit_ratio_for_projection = profit_abs / INITIAL_STARTING_CAPITAL

            seconds_in_week = 7 * 24 * 3600
            seconds_in_month = 30 * 24 * 3600 # Approximate month

            periods_in_week = seconds_in_week / duration_seconds
            periods_in_month = seconds_in_month / duration_seconds
            
            projected_weekly_return_pct = (math.pow(1 + profit_ratio_for_projection, periods_in_week) - 1) * 100
            projected_monthly_return_pct = (math.pow(1 + profit_ratio_for_projection, periods_in_month) - 1) * 100
            
            print(f"  Projected Performance (based on this validation segment, COMPOUNDED):")
            print(f"    Projected Weekly Return: {projected_weekly_return_pct:.2f}%")
            print(f"    Projected Monthly Return: {projected_monthly_return_pct:.2f}%")
            if abs(projected_weekly_return_pct) > 1000 or abs(projected_monthly_return_pct) > 5000 :
                 print("    WARNING: Projections are very high/low, likely due to short validation period or extreme performance. Interpret with caution.")
        else:
            print(f"  Validation period ({pd.to_timedelta(duration_seconds, unit='s')}) too short for meaningful weekly/monthly projection.")
    # --- END ADDED SECTION ---

    plot_backtest_results(data_raw_prices_df, tr.trade_log, tr.history, f"{title_prefix} Results for {TICKER}", COL_CLOSE)


# --- run_neat_trader ---
def run_neat_trader(config_file):
    global train_data_scaled_np_global, train_data_raw_prices_global, current_eval_window_start_index, \
           max_portfolio_ever_achieved_in_training, best_record_breaker_details, current_eval_window_raw_data_for_plotting

    raw_df_full = fetch_data(TICKER, DATA_PERIOD, DATA_INTERVAL)
    if raw_df_full.empty: print("ERROR: No data fetched. Exiting."); return
    
    feats_df_full = calculate_technical_indicators(raw_df_full.copy())
    if feats_df_full.empty: print("ERROR: Feature calculation resulted in empty dataframe. Exiting."); return
    
    feats_lags_df_full = add_lagged_features(feats_df_full, N_LAGS)
    if feats_lags_df_full.empty: print("ERROR: Adding lags resulted in empty dataframe. Exiting."); return

    train_feature_df, val_feature_df = split_data_by_days(feats_lags_df_full, TRAIN_DAYS)

    if train_feature_df is None or train_feature_df.empty:
        print("ERROR: Training feature dataframe is empty or None after split. Exiting."); return
    if val_feature_df is None:
        print("WARNING: Validation feature dataframe is None. Will proceed without validation if it's empty later.");
        val_feature_df = pd.DataFrame(columns=train_feature_df.columns)

    # Align raw prices with feature DFs (important!)
    # train_data_raw_prices_global will contain 'Close' and index for the training portion.
    # Used by reporter for its full train sim and by eval_genomes for its window sim.
    train_data_raw_prices_global = raw_df_full[[COL_CLOSE]].loc[train_feature_df.index].copy()
    # val_data_raw_prices_df will contain 'Close' and index for the validation portion.
    # Used for the final validation run.
    val_data_raw_prices_df = raw_df_full[[COL_CLOSE]].loc[val_feature_df.index].copy() if not val_feature_df.empty else pd.DataFrame(columns=[COL_CLOSE])


    if train_data_raw_prices_global.empty :
        print("ERROR: Training raw prices dataframe is empty after aligning with feature indices."); return

    train_scaled_features_df, val_scaled_features_df, scaler_obj = normalize_data(
        train_feature_df.copy(),
        val_feature_df.copy() if not val_feature_df.empty else pd.DataFrame(columns=train_feature_df.columns)
    )
    
    # train_data_scaled_np_global will contain ONLY scaled features (no 'Close') for NEAT input.
    train_data_scaled_np_global = train_scaled_features_df.drop(columns=[COL_CLOSE], errors='ignore').to_numpy()
    val_scaled_np_features = val_scaled_features_df.drop(columns=[COL_CLOSE], errors='ignore').to_numpy() if not val_scaled_features_df.empty else np.array([])

    if len(train_data_scaled_np_global) == 0:
        print("ERROR: Scaled training features are empty."); return
    if not val_feature_df.empty and len(val_scaled_np_features) == 0 :
        print("WARNING: Scaled validation features are empty, though validation data was present.")


    current_eval_window_start_index = 0
    max_portfolio_ever_achieved_in_training = INITIAL_STARTING_CAPITAL
    best_record_breaker_details = { "genome_obj": None, "window_fitness": -float('inf'), "portfolio_achieved_on_full_train": -float('inf')}
    current_eval_window_raw_data_for_plotting = None

    num_trader_state_features = 3
    num_input_features_from_data = train_data_scaled_np_global.shape[1]
    total_nn_inputs = num_input_features_from_data + num_trader_state_features
    
    print(f"Number of input features from data: {num_input_features_from_data}")
    print(f"Number of trader state features: {num_trader_state_features}")
    print(f"Total NN inputs: {total_nn_inputs}. CHECK YOUR CONFIG FILE ('num_inputs')!")

    cfg = neat.Config(neat.DefaultGenome, neat.DefaultReproduction, neat.DefaultSpeciesSet, neat.DefaultStagnation, config_file)
    if cfg.genome_config.num_inputs != total_nn_inputs:
        print(f"CONFIG ERROR: num_inputs mismatch! Script: {total_nn_inputs}, Config: {cfg.genome_config.num_inputs}. UPDATE CONFIG."); return

    pop = neat.Population(cfg)
    gen_reporter = GenerationReporter(
        plot_interval=PLOT_BEST_OF_GENERATION_EVERY,
        train_data_scaled=train_data_scaled_np_global,      # Scaled features for training window eval
        train_data_raw=train_data_raw_prices_global,        # Raw prices for training (used by reporter's full sim & eval_genomes window)
        neat_config=cfg,
        initial_capital=INITIAL_STARTING_CAPITAL,
        trading_fee=TRADING_FEE_PERCENT
    )
    pop.add_reporter(gen_reporter)
    pop.add_reporter(neat.StdOutReporter(True))
    stats = neat.StatisticsReporter(); pop.add_reporter(stats)

    print("\nStarting NEAT evolution...");
    try:
        pop.run(eval_genomes, N_GENERATIONS)
    except Exception as e:
        print(f"ERROR during NEAT evolution: {e}")
        import traceback; traceback.print_exc()
    finally: # Ensure metrics are plotted even if evolution is interrupted
        gen_reporter.plot_generational_metrics()


    winner_to_evaluate = None; winner_fitness_note = -float('inf'); final_report_source = "None"
    grb_genome = best_record_breaker_details["genome_obj"]
    grb_portfolio_full_train = best_record_breaker_details["portfolio_achieved_on_full_train"]
    grb_window_fitness = best_record_breaker_details["window_fitness"]
    cond1_grb_profitable = grb_genome and grb_portfolio_full_train > (INITIAL_STARTING_CAPITAL * 1.01)
    cond2_grb_decent_window_fit = grb_window_fitness is not None and grb_window_fitness >= 0.0

    if cond1_grb_profitable and cond2_grb_decent_window_fit:
        winner_to_evaluate = grb_genome; winner_fitness_note = grb_window_fitness
        final_report_source = "Global Record Breaker (Good FullTrainPf & WindowFit)"
    elif gen_reporter.neat_overall_best_genome_obj:
        winner_to_evaluate = gen_reporter.neat_overall_best_genome_obj; winner_fitness_note = gen_reporter.neat_best_fitness_so_far
        final_report_source = "Reporter's Best (by window fitness)"
    else:
        best_fitness_fallback = -float('inf'); temp_winner_fallback = None
        if pop.population:
            all_genomes_final_pop = list(pop.population.values())
            for g_val_iter in all_genomes_final_pop:
                if hasattr(g_val_iter, 'fitness') and g_val_iter.fitness is not None and g_val_iter.fitness > best_fitness_fallback:
                    best_fitness_fallback = g_val_iter.fitness; temp_winner_fallback = g_val_iter
        winner_to_evaluate = temp_winner_fallback
        winner_fitness_note = best_fitness_fallback if temp_winner_fallback and hasattr(temp_winner_fallback, 'fitness') and temp_winner_fallback.fitness is not None else -float('inf')
        final_report_source = "NEAT Population Fallback"
        if winner_to_evaluate is None: print("CRITICAL ERROR: No winner genome could be determined after evolution."); return # Added return
    
    fitness_display_string = f"{winner_fitness_note:.2f}" if winner_fitness_note is not None and not np.isinf(winner_fitness_note) else "N/A"
    print(f"\nOverall Best Genome Selected ({final_report_source}): ID {winner_to_evaluate.key if winner_to_evaluate else 'N/A'}, Fitness Context: {fitness_display_string}")

    if winner_to_evaluate:
        with open(f"winner_genome_{TICKER}.pkl", "wb") as f: pickle.dump(winner_to_evaluate, f)
        print(f"Saved overall best genome to winner_genome_{TICKER}.pkl")

    # gen_reporter.plot_generational_metrics() # Moved to finally block

    if winner_to_evaluate:
        print("\n--- Final Evaluation of Selected Overall Best Genome ---")
        if train_data_scaled_np_global is not None and len(train_data_scaled_np_global) > 0 and \
           train_data_raw_prices_global is not None and not train_data_raw_prices_global.empty:
            run_simulation_and_plot(winner_to_evaluate, cfg,
                                    train_data_scaled_np_global,
                                    train_data_raw_prices_global,
                                    "Selected Best Genome on Training Data (Full)")
        
        if val_scaled_np_features is not None and len(val_scaled_np_features) > 0 and \
           val_data_raw_prices_df is not None and not val_data_raw_prices_df.empty:
            run_simulation_and_plot(winner_to_evaluate, cfg,
                                    val_scaled_np_features,
                                    val_data_raw_prices_df, # Pass the correct raw prices DF for validation
                                    "Selected Best Genome on Validation Data",
                                    is_validation_run=True)
        elif not val_feature_df.empty :
             print("WARNING: Validation data was present but resulted in empty scaled features or raw prices for final eval. Skipping validation plot.")
        else:
             print("INFO: No validation data was available for final evaluation (likely due to short overall data period).")
    else:
        print("No best genome found to evaluate after evolution.")

if __name__ == "__main__":
    if not os.path.exists(CONFIG_FILE_PATH): print(f"Config file not found: {CONFIG_FILE_PATH}")
    else: run_neat_trader(CONFIG_FILE_PATH)
