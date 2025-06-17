# main.py
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
import math # For pow, isnan, isinf
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
N_GENERATIONS = 50 
MAX_EXPECTED_CREDIT = INITIAL_STARTING_CAPITAL * 10
MAX_EXPECTED_HOLDINGS_VALUE = INITIAL_STARTING_CAPITAL * 10
PLOT_BEST_OF_GENERATION_EVERY = 10
PLOT_WINDOW_PERFORMANCE = True
TRADING_FEE_PERCENT = 0.001
EVAL_WINDOW_SIZE_MINUTES = 4 * 60

# --- Attention Mechanism Configuration ---
ATTENTION_SEQUENCE_LENGTH = 15
ATTENTION_OUTPUT_DIM = 24
ATTENTION_HEADS = 3
if ATTENTION_OUTPUT_DIM % ATTENTION_HEADS != 0:
    raise ValueError("ATTENTION_OUTPUT_DIM must be divisible by ATTENTION_HEADS for multi-head attention internal splitting.")

_COLUMN_NAMES_RESOLVED_FOR_CURRENT_FETCH = False
COL_OPEN, COL_HIGH, COL_LOW, COL_CLOSE, COL_VOLUME = 'Open', 'High', 'Low', 'Close', 'Volume'
current_eval_window_start_index = 0

# --- GLOBAL RECORD TRACKING ---
max_final_portfolio_record_global = INITIAL_STARTING_CAPITAL 

best_record_breaker_details = {
    "genome_obj": None, "window_fitness": -float('inf'), "portfolio_achieved_on_full_train": -float('inf')
}
current_eval_window_raw_data_for_plotting = None

train_data_scaled_np_global = None
train_data_raw_prices_global = None
num_input_features_from_data_global = 0
FITNESS_THRESHOLD_CONFIG_FROM_FILE = 10000.0


# --- FITNESS CONSTANTS (Uncompromising Profit Engine V6) ---
# A. IMMEDIATE DISQUALIFICATIONS (DEATH PENALTIES)
RUIN_DEATH_SCORE = -1000.0
MINIMUM_TRADES_FOR_ACTIVITY = 4
INACTIVITY_DEATH_SCORE = -900.0
NO_SELLS_DEATH_SCORE = -800.0
VERY_LOW_FITNESS_UNSALVAGEABLE = -1100.0

# B. THE CORE PROFIT ENGINE
# The base reward is now directly tied to the percentage return.
PROFIT_SCALER = 500.0 
# ** NEW **: Rewards compound, making high-percentage gains exponentially more valuable.
PROFIT_COMPOUNDING_FACTOR = 1.5 

# C. RECORD-BREAKING
# A massive bonus for achieving a new all-time high FINAL portfolio value.
RECORD_BREAKER_BONUS = 5000.0

# D. BEHAVIORAL SHAPING
# Bonus for having a high ratio of profits to losses. Rewards clean trading.
PROFIT_FACTOR_BONUS = 150.0
# Penalty for ending with a large portion of value tied up in unrealized assets.
# This forces the agent to learn to take profit.
TERMINAL_HOLDING_PENALTY_FACTOR = 0.7 
# Penalty for high drawdown from peak portfolio value.
DRAWDOWN_PENALTY_FACTOR = 0.5
# --- END OF FITNESS CONSTANTS ---

class GenerationReporter(neat.reporting.BaseReporter):
    # This class is correct and does not need changes.
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
            "Max Final Portfolio Global Record ($)": [], # From window evals
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

        min_eval_window_size_for_reporter = ATTENTION_SEQUENCE_LENGTH + 30
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
        current_eval_window_start_index = min(current_eval_window_start_index, max(0, full_train_len - window_size_for_eval))

        eval_win_start_idx = current_eval_window_start_index
        eval_win_end_idx = min(eval_win_start_idx + window_size_for_eval, full_train_len)

        if eval_win_end_idx <= eval_win_start_idx or eval_win_start_idx >= len(self.train_data_raw_prices_global_ref):
             current_eval_window_raw_data_for_plotting = pd.DataFrame()
        else:
            safe_eval_win_end_idx = min(eval_win_end_idx, len(self.train_data_raw_prices_global_ref))
            current_eval_window_raw_data_for_plotting = self.train_data_raw_prices_global_ref.iloc[eval_win_start_idx:safe_eval_win_end_idx]

        if not current_eval_window_raw_data_for_plotting.empty:
            start_date_str = current_eval_window_raw_data_for_plotting.index[0].strftime('%Y-%m-%d %H:%M')
            end_date_str = current_eval_window_raw_data_for_plotting.index[-1].strftime('%Y-%m-%d %H:%M')
            print(f"  Gen {generation}: eval_genomes window: Global Idx {eval_win_start_idx}-{eval_win_start_idx + len(current_eval_window_raw_data_for_plotting)-1} ({start_date_str} to {end_date_str}), size={len(current_eval_window_raw_data_for_plotting)}")
        else:
            print(f"  Gen {generation}: eval_genomes window: Global Idx {eval_win_start_idx}-{eval_win_end_idx-1}, size={eval_win_end_idx - eval_win_start_idx}. (Note: Window data is empty or invalid)")


    def _actual_end_of_generation_logic(self, config, population_genomes_dict, species_set_object):
        global max_final_portfolio_record_global, best_record_breaker_details, \
               current_eval_window_raw_data_for_plotting, train_data_scaled_np_global, \
               num_input_features_from_data_global

        best_genome_this_gen_by_window_fitness, current_gen_max_window_fitness = None, -float('inf')
        all_current_genomes = list(population_genomes_dict.values())
        for g_obj in all_current_genomes:
            if hasattr(g_obj, 'fitness') and g_obj.fitness is not None and g_obj.fitness > current_gen_max_window_fitness:
                current_gen_max_window_fitness, best_genome_this_gen_by_window_fitness = g_obj.fitness, g_obj

        if best_genome_this_gen_by_window_fitness and (current_gen_max_window_fitness > self.neat_best_fitness_so_far):
            self.neat_best_fitness_so_far = current_gen_max_window_fitness
            self.neat_overall_best_genome_obj = copy.deepcopy(best_genome_this_gen_by_window_fitness)
            print(f"  REPORTER: ** New best NEAT genome (by window fitness)! ** Gen: {self.generation_count}, ID: {best_genome_this_gen_by_window_fitness.key}, Fitness: {current_gen_max_window_fitness:.2f}")

        self.generations_list.append(self.generation_count)

        gen_metrics_values = {key: np.nan for key in self.metrics_history.keys()}
        gen_metrics_values["Max Final Portfolio Global Record ($)"] = max_final_portfolio_record_global

        if best_genome_this_gen_by_window_fitness:
            gen_metrics_values["Best Fitness (Window)"] = current_gen_max_window_fitness

            net_full_sim = neat.nn.FeedForwardNetwork.create(best_genome_this_gen_by_window_fitness, self.neat_config)
            rep_trader_full_sim = Trader(self.initial_capital, INITIAL_STARTING_HOLDINGS, trading_fee_percent=self.trading_fee)
            buys_rep_sim, sells_rep_sim = 0,0

            sim_start_index_reporter_global = ATTENTION_SEQUENCE_LENGTH - 1
            final_pf_rep = self.initial_capital
            net_profit_rep = 0.0

            if sim_start_index_reporter_global >= len(self.train_data_scaled_features_global_ref):
                print(f"    REPORTER: Full train sim skipped for Gen {self.generation_count}'s best, data too short.")
            else:
                for i_global in range(sim_start_index_reporter_global, len(self.train_data_scaled_features_global_ref)):
                    if not rep_trader_full_sim.is_alive: break

                    current_step_features_np = self.train_data_scaled_features_global_ref[i_global]
                    start_seq_global_idx = i_global - ATTENTION_SEQUENCE_LENGTH + 1
                    sequence_for_attention_np = self.train_data_scaled_features_global_ref[start_seq_global_idx : i_global+1]

                    attention_context_np = get_attention_output(
                        sequence_for_attention_np,
                        current_seq_len=sequence_for_attention_np.shape[0],
                        target_seq_len=ATTENTION_SEQUENCE_LENGTH,
                        feature_dim=num_input_features_from_data_global
                    )

                    price = self.train_data_raw_prices_global_ref.iloc[i_global][COL_CLOSE]
                    ts = self.train_data_raw_prices_global_ref.index[i_global]
                    state = rep_trader_full_sim.get_state_for_nn(price, MAX_EXPECTED_CREDIT, MAX_EXPECTED_HOLDINGS_VALUE)

                    nn_in = np.concatenate((current_step_features_np, attention_context_np.flatten(), state))
                    action, amount = net_full_sim.activate(nn_in)

                    if action > 0.55:
                        if rep_trader_full_sim.buy(np.clip(amount,0.01,1.0) * rep_trader_full_sim.credit, price, ts): buys_rep_sim +=1
                    elif action < 0.45:
                        if rep_trader_full_sim.sell(np.clip(amount,0.01,1.0) * rep_trader_full_sim.holdings_shares, price, ts): sells_rep_sim +=1

                    rep_trader_full_sim.update_history(ts, price)
                
                if not self.train_data_raw_prices_global_ref.empty:
                    last_price_full_sim = self.train_data_raw_prices_global_ref.iloc[len(self.train_data_scaled_features_global_ref)-1][COL_CLOSE]
                    final_pf_rep = rep_trader_full_sim.get_portfolio_value(last_price_full_sim)
                net_profit_rep = final_pf_rep - self.initial_capital

            if rep_trader_full_sim.max_portfolio_value_achieved > best_record_breaker_details["portfolio_achieved_on_full_train"]:
                 best_record_breaker_details["genome_obj"] = copy.deepcopy(best_genome_this_gen_by_window_fitness)
                 best_record_breaker_details["window_fitness"] = current_gen_max_window_fitness
                 best_record_breaker_details["portfolio_achieved_on_full_train"] = rep_trader_full_sim.max_portfolio_value_achieved
                 print(f"    REPORTER: !!! NEW BEST PEAK PORTFOLIO (on Full Train Sim): ${rep_trader_full_sim.max_portfolio_value_achieved:.2f} "
                       f"by Gen {self.generation_count}'s Best (ID {best_genome_this_gen_by_window_fitness.key}, WindowFit {current_gen_max_window_fitness:.2f}) !!!")

            gen_metrics_values["Best Gen Genome's Portfolio ($) (Full Train Sim)"] = final_pf_rep
            gen_metrics_values["Best Gen Genome's Net Profit ($) (Full Train Sim)"] = net_profit_rep
            gen_metrics_values["Best Gen Genome's Credit ($) (Full Train Sim)"] = rep_trader_full_sim.credit
            gen_metrics_values["Best Gen Genome's Total Trades (Full Train Sim)"] = len(rep_trader_full_sim.trade_log)
            gen_metrics_values["Best Gen Genome's Buys (Full Train Sim)"] = buys_rep_sim
            gen_metrics_values["Best Gen Genome's Sells (Full Train Sim)"] = sells_rep_sim
            gen_metrics_values["Best Gen Genome's Fees Paid ($) (Full Train Sim)"] = rep_trader_full_sim.total_fees_paid

            print(f"    REPORTER Gen {self.generation_count} Best (ID {best_genome_this_gen_by_window_fitness.key}, WindowFit: {current_gen_max_window_fitness:.2f}) "
                  f"SimOnFullTrain: NetProfit: ${net_profit_rep:.2f}, Pf: ${final_pf_rep:.2f}, Cr: ${rep_trader_full_sim.credit:.2f}, "
                  f"Trades: {len(rep_trader_full_sim.trade_log)} (B:{buys_rep_sim}/S:{sells_rep_sim}), Fees: ${rep_trader_full_sim.total_fees_paid:.2f}")

            if not self.train_data_raw_prices_global_ref.empty and self.initial_capital > 1e-6 and \
               sim_start_index_reporter_global < len(self.train_data_scaled_features_global_ref):

                train_start_time = self.train_data_raw_prices_global_ref.index[sim_start_index_reporter_global]
                train_end_time = self.train_data_raw_prices_global_ref.index[len(self.train_data_scaled_features_global_ref)-1]
                train_duration_seconds = (train_end_time - train_start_time).total_seconds()

                if train_duration_seconds > 3600:
                    train_profit_ratio_for_projection = net_profit_rep / self.initial_capital
                    seconds_in_week = 7 * 24 * 3600
                    seconds_in_month = 30 * 24 * 3600

                    if 1 + train_profit_ratio_for_projection > 0:
                        periods_in_week = seconds_in_week / train_duration_seconds
                        periods_in_month = seconds_in_month / train_duration_seconds
                        projected_weekly_return_pct_train_val = (math.pow(1 + train_profit_ratio_for_projection, periods_in_week) - 1) * 100
                        projected_monthly_return_pct_train_val = (math.pow(1 + train_profit_ratio_for_projection, periods_in_month) - 1) * 100
                    else:
                        projected_weekly_return_pct_train_val = -100.0 
                        projected_monthly_return_pct_train_val = -100.0

                    gen_metrics_values["Projected Weekly Return (%) (Full Train Sim)"] = projected_weekly_return_pct_train_val
                    gen_metrics_values["Projected Monthly Return (%) (Full Train Sim)"] = projected_monthly_return_pct_train_val

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
            print(f"  REPORTER: No genome with reportable fitness (valid, > -inf) in gen {self.generation_count}.")

        for key in self.metrics_history.keys():
            self.metrics_history[key].append(gen_metrics_values.get(key, np.nan))

    def end_generation(self, config, population_genomes_dict, species_set):
        self._actual_end_of_generation_logic(config, population_genomes_dict, species_set)

    def post_evaluate(self, config, population_object, species_set_object, best_genome_from_neat):
        if best_genome_from_neat and hasattr(best_genome_from_neat, 'fitness') and best_genome_from_neat.fitness is not None:
            if self.neat_overall_best_genome_obj is None or best_genome_from_neat.fitness > self.neat_best_fitness_so_far:
                self.neat_best_fitness_so_far = best_genome_from_neat.fitness
                self.neat_overall_best_genome_obj = copy.deepcopy(best_genome_from_neat)
    
    def found_solution(self, config, generation, best_found_by_neat):
        print(f"REPORTER: Solution found by NEAT in generation {generation} by genome {best_found_by_neat.key} with fitness {best_found_by_neat.fitness:.2f}!")
        if best_found_by_neat and hasattr(best_found_by_neat, 'fitness') and best_found_by_neat.fitness is not None and \
           (self.neat_overall_best_genome_obj is None or best_found_by_neat.fitness > self.neat_best_fitness_so_far):
            self.neat_best_fitness_so_far = best_found_by_neat.fitness
            self.neat_overall_best_genome_obj = copy.deepcopy(best_found_by_neat)
    
    def info(self, msg):
        pass

    def plot_generational_metrics(self):
        metrics_to_plot = {
            k: v for k, v in self.metrics_history.items()
            if "Best Fitness (Window)" in k or \
               "Max Final Portfolio Global Record ($)" in k or \
               "Net Profit ($) (Full Train Sim)" in k or \
               "Portfolio ($) (Full Train Sim)" in k or \
               "Projected Weekly Return (%) (Full Train Sim)" in k
        }
        if self.generations_list and any(len(v_list) > 0 for v_list in metrics_to_plot.values()):
            valid_metrics_history = {}
            max_len = len(self.generations_list)

            for k, v_list in metrics_to_plot.items():
                if len(v_list) == 0: continue
                if len(v_list) != max_len:
                     print(f"Plotting Warning: Metric '{k}' has length {len(v_list)}, but expected {max_len} generations. Padding/truncating for plot.")
                if len(v_list) < max_len:
                    v_list_padded = v_list + [np.nan] * (max_len - len(v_list))
                else:
                    v_list_padded = v_list[:max_len]
                if not all(np.isnan(val) if isinstance(val, float) else False for val in v_list_padded):
                    valid_metrics_history[k] = v_list_padded

            if valid_metrics_history:
                 plot_generational_performance(self.generations_list, valid_metrics_history, title="Key Metrics Per Generation (Uncompromising Profit Engine V6)")
            else:
                 print("No valid generational metrics data to plot (all NaNs after alignment or selected metrics were empty).")
        else:
            print("No generational data accumulated to plot (all selected metrics_history values are empty or no generations run).")

def resolve_column_names(df_columns, ticker_symbol_str):
    # This function is correct and does not need changes.
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
        if not matched_pattern: print(f"WARNING: Unhandled yfinance MultiIndex {df_columns.names}. Defaulting to standard names ('Open', 'High', etc.). Columns available: {df_columns.tolist()}")
    elif not all(col_str in df_columns for col_str in [_COL_OPEN_temp, _COL_HIGH_temp, _COL_LOW_temp, _COL_CLOSE_temp, _COL_VOLUME_temp]):
        if all(col_str.lower() in (c.lower() for c in df_columns) for col_str in [_COL_OPEN_temp, _COL_HIGH_temp, _COL_LOW_temp, _COL_CLOSE_temp, _COL_VOLUME_temp]):
            df_cols_lower = {c.lower(): c for c in df_columns}
            _COL_OPEN_temp = df_cols_lower['open']
            _COL_HIGH_temp = df_cols_lower['high']
            _COL_LOW_temp = df_cols_lower['low']
            _COL_CLOSE_temp = df_cols_lower['close']
            _COL_VOLUME_temp = df_cols_lower['volume']
            matched_pattern = True
            print("INFO: Found lowercase OHLCV column names.")
        else:
            print(f"WARNING: Standard column names ('Open', 'High', 'Low', 'Close', 'Volume') not all found in DataFrame. Defaulting. Columns available: {df_columns.tolist()}")
    else:
        matched_pattern = True
    COL_OPEN, COL_HIGH, COL_LOW, COL_CLOSE, COL_VOLUME = _COL_OPEN_temp, _COL_HIGH_temp, _COL_LOW_temp, _COL_CLOSE_temp, _COL_VOLUME_temp
    _COLUMN_NAMES_RESOLVED_FOR_CURRENT_FETCH = True
    if matched_pattern:
        print(f"INFO: Resolved column names: Open={COL_OPEN}, High={COL_HIGH}, Low={COL_LOW}, Close={COL_CLOSE}, Volume={COL_VOLUME}")

def fetch_data(ticker, period, interval):
    # This function is correct and does not need changes.
    global _COLUMN_NAMES_RESOLVED_FOR_CURRENT_FETCH; _COLUMN_NAMES_RESOLVED_FOR_CURRENT_FETCH = False
    print(f"Fetching data for {ticker} (period: {period}, interval: {interval})...")
    if interval == "1m" and pd.to_timedelta(period) > pd.to_timedelta("7d"):
        print(f"WARNING: yfinance may restrict '1m' interval data to 7 days for free API. Requested period '{period}' might be truncated or fail for older data.")

    if isinstance(ticker, str):
        data = yf.download(ticker, period=period, interval=interval, progress=False, auto_adjust=True)
    else:
        data = yf.download(ticker, period=period, interval=interval, progress=False, auto_adjust=True, group_by='ticker' if isinstance(ticker, list) and len(ticker) > 1 else None)

    if data.empty:
        raise ValueError(f"No data fetched for {ticker} with period {period} and interval {interval}. Check ticker, period, and internet connection.")

    print(f"Actual data range fetched: {data.index.min()} to {data.index.max()} ({len(data)} rows)")

    actual_ticker_for_resolve = ticker[0] if isinstance(ticker, list) and len(ticker) > 0 else ticker
    resolve_column_names(data.columns, actual_ticker_for_resolve)

    required_cols_resolved = [COL_OPEN, COL_HIGH, COL_LOW, COL_CLOSE, COL_VOLUME]
    missing_cols = [c for c in required_cols_resolved if c not in data.columns]
    if missing_cols:
        raise KeyError(f"One or more required OHLCV columns are missing after attempting to resolve names: {missing_cols}. DataFrame columns: {data.columns.tolist()}")

    data.dropna(subset=required_cols_resolved, inplace=True)
    if data.empty:
        raise ValueError(f"Data became empty after dropping NA values for essential OHLCV columns for {ticker}.")

    print(f"Data ready after NA drop for essential columns: {data.shape[0]} rows")
    return data

def calculate_features(df_ohlcv):
    # This function is correct and does not need changes.
    print("Calculating Features (TAs and Multi-Timeframe Changes)...")
    df = df_ohlcv.copy()
    if not all(col in df.columns for col in [COL_OPEN, COL_HIGH, COL_LOW, COL_CLOSE, COL_VOLUME]):
        raise ValueError(f"One or more required OHLCV columns ({COL_OPEN}, {COL_HIGH}, etc.) are missing before TA calculation.")
    
    df_ta_temp = pd.DataFrame(index=df.index)
    df_ta_temp['Open'] = df[COL_OPEN]
    df_ta_temp['High'] = df[COL_HIGH]
    df_ta_temp['Low'] = df[COL_LOW]
    df_ta_temp['Close'] = df[COL_CLOSE]
    df_ta_temp['Volume'] = df[COL_VOLUME]

    df_ta = ta.add_all_ta_features(df_ta_temp, "Open", "High", "Low", "Close", "Volume", fillna=True)

    prefs = ['trend_macd', 'trend_macd_signal', 'trend_macd_diff', 'momentum_rsi', 'momentum_stoch_rsi',
             'volatility_atr', 'trend_ema_fast', 'trend_ema_slow', 'volume_obv', 'others_cr']
    out_df = pd.DataFrame(index=df.index)
    out_df['Close'] = df[COL_CLOSE].copy()
    
    selected_ta_cols_for_output = []
    processed_ta_names = set()
    for p in prefs:
        for col_name_ta in df_ta.columns:
            str_col_ta = str(col_name_ta)
            if p.lower() in str_col_ta.lower():
                unique_simp_name = p
                ctr = 1
                while unique_simp_name in out_df.columns or unique_simp_name in processed_ta_names:
                    unique_simp_name = f"{p}_{ctr}"
                    ctr += 1
                
                if str_col_ta not in processed_ta_names:
                    out_df[unique_simp_name] = df_ta[str_col_ta].values
                    selected_ta_cols_for_output.append(unique_simp_name)
                    processed_ta_names.add(str_col_ta)
                    processed_ta_names.add(unique_simp_name)
                    break
    print(f"Selected TA features: {selected_ta_cols_for_output}")

    timeframes_minutes = {
        '1H': 60, '2H': 120, '6H': 360, '12H': 720, '24H': 1440
    }
    added_mtf_features = []
    for tf_name, minutes in timeframes_minutes.items():
        if minutes >= len(df):
            print(f"Skipping MTF feature '{tf_name}' ({minutes} min) as data length ({len(df)}) is insufficient.")
            continue
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
    # This function is correct and does not need changes.
    print(f"Adding {n_lags} lags..."); lagged_df = df.copy()
    features_to_lag = [col for col in df.columns if col != 'Close'] 
    if not features_to_lag:
        print("Warning: No features to lag (other than 'Close'). Skipping lag addition.")
        return lagged_df
        
    for lag in range(1, n_lags + 1):
        shifted = df[features_to_lag].shift(lag)
        for col in features_to_lag:
            lagged_df[f'{col}_lag{lag}'] = shifted[col]

    initial_len = len(lagged_df)
    lagged_df.dropna(inplace=True)
    if lagged_df.empty and n_lags > 0 and initial_len > 0 :
        print(f"WARNING: DataFrame became empty after adding {n_lags} lags and dropping NA. Original data (len {initial_len}) might be too short for this many lags.")
    print(f"Shape after lags/dropna: {lagged_df.shape}"); return lagged_df

def split_data_by_days(df, train_days_count):
    # This function is correct and does not need changes.
    print(f"Splitting data: attempting {train_days_count} train days..."); df = df.sort_index()
    if not isinstance(df.index, pd.DatetimeIndex):
        raise ValueError("DataFrame index must be a DatetimeIndex for split_data_by_days.")
        
    unique_days = sorted(df.index.normalize().unique())
    if len(unique_days) == 0:
        print("ERROR in split_data_by_days: No unique days found in the dataset."); return None, None

    if train_days_count >= len(unique_days):
        print(f"WARNING: train_days_count ({train_days_count}) is >= total unique days ({len(unique_days)}).")
        if len(unique_days) > 1:
            train_days_count = len(unique_days) - 1 
            print(f"Adjusted train_days_count to {train_days_count} to allow for a minimal validation set.")
        else:
            print("Only one unique day of data. Using all for training, no validation set will be created.")
            train_df = df.copy()
            validation_df = pd.DataFrame(columns=df.columns, index=pd.to_datetime([]))
            validation_df = validation_df.astype(df.dtypes)
            print(f"Train: {train_df.shape}, from {train_df.index.min() if not train_df.empty else 'N/A'} to {train_df.index.max() if not train_df.empty else 'N/A'}")
            print(f"Valid: {validation_df.shape}, Validation set is empty.")
            return train_df, validation_df

    if train_days_count <= 0:
        print("ERROR: train_days_count must be positive. Cannot create training set.")
        return None, None
        
    split_date_boundary = unique_days[train_days_count -1] 

    train_df = df[df.index.normalize() <= split_date_boundary]
    validation_df = df[df.index.normalize() > split_date_boundary]

    print(f"Train: {train_df.shape}, from {train_df.index.min() if not train_df.empty else 'N/A'} to {train_df.index.max() if not train_df.empty else 'N/A'}")
    print(f"Valid: {validation_df.shape}, from {validation_df.index.min() if not validation_df.empty else 'N/A'} to {validation_df.index.max() if not validation_df.empty else 'N/A'}")

    if train_df.empty:
        print("ERROR in split_data_by_days: Training dataframe is empty after split."); return None, None
    if validation_df.empty:
        print("WARNING in split_data_by_days: Validation dataframe is empty. Consider increasing DATA_PERIOD or reducing TRAIN_DAYS if validation is desired.");

    return train_df, validation_df

def normalize_data(train_df, val_df):
    # This function is correct and does not need changes.
    print("Normalizing data using MinMaxScaler (fitted on training features)...");

    train_close_prices = train_df['Close'].copy() if 'Close' in train_df.columns else None
    val_close_prices = val_df['Close'].copy() if 'Close' in val_df.columns and val_df is not None and not val_df.empty else None

    train_features = train_df.drop(columns=['Close'], errors='ignore')
    if val_df is not None and not val_df.empty:
        val_features = val_df.drop(columns=['Close'], errors='ignore')
    else:
        val_features = pd.DataFrame(columns=train_features.columns, index=val_df.index if val_df is not None else None)

    if train_features.empty:
        print("Warning: Training features DataFrame is empty (or only had 'Close' column). Returning DFs without feature scaling.")
        train_scaled_df = pd.DataFrame(index=train_df.index)
        if train_close_prices is not None: train_scaled_df['Close'] = train_close_prices
        
        val_scaled_df = pd.DataFrame(index=val_df.index if val_df is not None else None)
        if val_close_prices is not None: val_scaled_df['Close'] = val_close_prices
        return train_scaled_df, val_scaled_df, None

    scaler = MinMaxScaler()
    train_features_scaled_np = scaler.fit_transform(train_features)

    if not val_features.empty and not val_features.columns.empty:
        val_features_scaled_np = scaler.transform(val_features)
    else:
        val_features_scaled_np = np.array([]).reshape(0, train_features.shape[1])

    train_scaled_df = pd.DataFrame(train_features_scaled_np, columns=train_features.columns, index=train_features.index)
    
    if not val_features.empty and not val_features.columns.empty :
        val_scaled_df = pd.DataFrame(val_features_scaled_np, columns=val_features.columns, index=val_features.index)
    else:
        val_scaled_df = pd.DataFrame(columns=train_features.columns, index=val_features.index if val_features is not None else None, dtype=float)

    if train_close_prices is not None:
        train_scaled_df['Close'] = train_close_prices
    
    if val_close_prices is not None and not val_df.empty :
        val_scaled_df['Close'] = val_close_prices
    elif val_df is not None and val_df.empty and 'Close' in train_scaled_df.columns :
        val_scaled_df['Close'] = pd.Series(dtype=train_scaled_df['Close'].dtype, index=val_scaled_df.index)

    return train_scaled_df, val_scaled_df, scaler

def eval_genomes(genomes, config):
    # *** THIS IS THE MAIN CORRECTED FUNCTION ***
    global train_data_scaled_np_global, train_data_raw_prices_global, \
           current_eval_window_start_index, num_input_features_from_data_global, \
           max_final_portfolio_record_global

    full_train_len = len(train_data_scaled_np_global)
    window_size_for_eval = EVAL_WINDOW_SIZE_MINUTES
    
    min_required_data_points = ATTENTION_SEQUENCE_LENGTH + 50
    if window_size_for_eval >= full_train_len or window_size_for_eval < min_required_data_points:
        window_size_for_eval = max(min_required_data_points, full_train_len // 2)
        if full_train_len < min_required_data_points:
            for _, genome in genomes: genome.fitness = VERY_LOW_FITNESS_UNSALVAGEABLE
            return

    start_idx_global_for_this_gen_eval = current_eval_window_start_index
    end_idx_global_for_this_gen_eval = min(start_idx_global_for_this_gen_eval + window_size_for_eval, 
                                           full_train_len, 
                                           len(train_data_raw_prices_global))
    
    actual_window_len = end_idx_global_for_this_gen_eval - start_idx_global_for_this_gen_eval

    if actual_window_len < min_required_data_points:
        for _, genome in genomes: genome.fitness = VERY_LOW_FITNESS_UNSALVAGEABLE
        return

    current_eval_scaled_features_window_np = train_data_scaled_np_global[start_idx_global_for_this_gen_eval : end_idx_global_for_this_gen_eval]
    current_eval_raw_prices_df_window = train_data_raw_prices_global.iloc[start_idx_global_for_this_gen_eval : end_idx_global_for_this_gen_eval]

    for genome_id, genome in genomes:
        net = neat.nn.FeedForwardNetwork.create(genome, config)
        trader = Trader(INITIAL_STARTING_CAPITAL, INITIAL_STARTING_HOLDINGS, trading_fee_percent=TRADING_FEE_PERCENT)
        
        sim_loop_start_offset_in_window = ATTENTION_SEQUENCE_LENGTH - 1
        
        for i_window in range(sim_loop_start_offset_in_window, actual_window_len):
            if not trader.is_alive: break

            current_global_idx = start_idx_global_for_this_gen_eval + i_window
            start_seq_global_idx = current_global_idx - ATTENTION_SEQUENCE_LENGTH + 1
            sequence_for_attention_np = train_data_scaled_np_global[start_seq_global_idx : current_global_idx + 1]
            
            attention_context_np = get_attention_output(
                sequence_for_attention_np, current_seq_len=sequence_for_attention_np.shape[0],
                target_seq_len=ATTENTION_SEQUENCE_LENGTH, feature_dim=num_input_features_from_data_global)
            
            current_step_features_np_in_window = current_eval_scaled_features_window_np[i_window]
            price = current_eval_raw_prices_df_window.iloc[i_window][COL_CLOSE]
            ts = current_eval_raw_prices_df_window.index[i_window]
            state = trader.get_state_for_nn(price, MAX_EXPECTED_CREDIT, MAX_EXPECTED_HOLDINGS_VALUE)
            
            nn_in = np.concatenate((current_step_features_np_in_window, attention_context_np.flatten(), state))
            action_raw, amount_raw = net.activate(nn_in)
            amount_to_use = np.clip(amount_raw, 0.01, 1.0)
            
            if action_raw > 0.55:
                trader.buy(amount_to_use * trader.credit, price, ts)
            elif action_raw < 0.45:
                trader.sell(amount_to_use * trader.holdings_shares, price, ts)
            trader.update_history(ts, price)
            
        final_pf = trader.get_portfolio_value(current_eval_raw_prices_df_window.iloc[-1][COL_CLOSE])
        net_profit = final_pf - INITIAL_STARTING_CAPITAL
        
        # --- FITNESS CALCULATION (Uncompromising Profit Engine V6) ---
        
        # 1. IMMEDIATE DISQUALIFICATIONS
        if not trader.is_alive:
            genome.fitness = RUIN_DEATH_SCORE
            continue
        
        num_buys = sum(1 for t in trader.trade_log if t['type'] == 'buy')
        num_sells = sum(1 for t in trader.trade_log if t['type'] == 'sell')

        if (num_buys + num_sells) < MINIMUM_TRADES_FOR_ACTIVITY:
            genome.fitness = INACTIVITY_DEATH_SCORE
            continue
            
        if num_buys > 0 and num_sells == 0:
            genome.fitness = NO_SELLS_DEATH_SCORE
            continue

        # 2. PROFIT-DRIVEN FITNESS CALCULATION
        fitness = 0.0
        profit_percentage = net_profit / INITIAL_STARTING_CAPITAL
        
        # Core reward: scaled and compounded profit percentage
        if profit_percentage > 0:
            fitness = (pow(1 + profit_percentage, PROFIT_COMPOUNDING_FACTOR) - 1) * PROFIT_SCALER
        else:
            # If not profitable, fitness is negative, proportional to the loss
            genome.fitness = net_profit # Directly assign the loss as negative fitness
            continue

        # --- From here, we are only dealing with profitable genomes ---

        # 3. RECORD-BREAKING REWARD
        if final_pf > max_final_portfolio_record_global:
            fitness += RECORD_BREAKER_BONUS
            print(f"      *** NEW FINAL PROFIT RECORD: ${final_pf:.2f} by Genome {genome_id} (Profit: ${net_profit:.2f}) ***")
            max_final_portfolio_record_global = final_pf

        # 4. BEHAVIORAL BONUSES
        gross_profit = sum(t['profit'] for t in trader.trade_log if t['type'] == 'sell' and t.get('profit', 0) > 0)
        gross_loss = sum(abs(t['profit']) for t in trader.trade_log if t['type'] == 'sell' and t.get('profit', 0) < 0)
        if gross_profit > 0 and gross_loss > 1e-6:
            profit_factor = gross_profit / gross_loss
            fitness += profit_factor * PROFIT_FACTOR_BONUS
        
        # 5. BEHAVIORAL PENALTIES
        # Drawdown Penalty
        max_pf_in_window = trader.max_portfolio_value_achieved
        if max_pf_in_window > final_pf:
             drawdown_ratio = (max_pf_in_window - final_pf) / max_pf_in_window
             # Penalty is proportional to the fitness itself
             fitness *= (1 - (drawdown_ratio * DRAWDOWN_PENALTY_FACTOR))

        # Terminal Holding Penalty
        last_price = current_eval_raw_prices_df_window.iloc[-1][COL_CLOSE]
        holdings_value = trader.holdings_shares * last_price
        holdings_ratio = holdings_value / final_pf if final_pf > 0 else 1
        fitness *= (1 - (holdings_ratio * TERMINAL_HOLDING_PENALTY_FACTOR))

        # 6. FINAL ASSIGNMENT
        # ** UNCOMPROMISING CHECK **: If fitness somehow became negative after penalties, it's a failure.
        if fitness <= 0:
            genome.fitness = net_profit # Punish it by its (small) initial profit
        else:
            genome.fitness = fitness

def run_simulation_and_plot(genome, config, data_scaled_features_np_segment, data_raw_prices_df_segment, title_prefix, is_validation_run=False):
    # This function is correct and does not need changes.
    global num_input_features_from_data_global

    if data_raw_prices_df_segment.empty or COL_CLOSE not in data_raw_prices_df_segment.columns:
        print(f"Plot Info: Raw price data for '{title_prefix}' is empty or missing '{COL_CLOSE}' column. Skipping plot.")
        return

    if data_scaled_features_np_segment is None or len(data_scaled_features_np_segment) == 0:
        print(f"Simulation Info: No scaled feature data provided for '{title_prefix}'. Cannot run simulation for plotting.")
        plot_backtest_results(data_raw_prices_df_segment, [], [], f"{title_prefix} Results for {TICKER} (No Sim Data)", COL_CLOSE)
        return

    if len(data_scaled_features_np_segment) != len(data_raw_prices_df_segment):
        print(f"Data Mismatch Error for '{title_prefix}': Scaled features length ({len(data_scaled_features_np_segment)}) "
              f"!= Raw prices length ({len(data_raw_prices_df_segment)}). Skipping simulation for plotting.")
        return

    min_len_for_sim_plot_attention = ATTENTION_SEQUENCE_LENGTH 
    if len(data_scaled_features_np_segment) < min_len_for_sim_plot_attention :
        print(f"Simulation Info: Data segment too short ({len(data_scaled_features_np_segment)}, need {min_len_for_sim_plot_attention}) "
              f"for attention sequence in '{title_prefix}'. Skipping simulation for plotting.")
        plot_backtest_results(data_raw_prices_df_segment, [], [], f"{title_prefix} Results for {TICKER} (Sim Skipped - Short Data for Attention)", COL_CLOSE)
        return

    print(f"\n--- {title_prefix} Evaluation (for plotting/reporting) ---")
    net = neat.nn.FeedForwardNetwork.create(genome, config)
    tr = Trader(INITIAL_STARTING_CAPITAL, INITIAL_STARTING_HOLDINGS, trading_fee_percent=TRADING_FEE_PERCENT)
    buys,sells = 0,0

    final_val_idx = len(data_raw_prices_df_segment) - 1 if not data_raw_prices_df_segment.empty else 0
    
    actual_sim_loop_start_idx_in_segment = ATTENTION_SEQUENCE_LENGTH - 1

    if actual_sim_loop_start_idx_in_segment >= len(data_scaled_features_np_segment):
        print(f"  {title_prefix}: Data segment (len {len(data_scaled_features_np_segment)}) too short to run plotting simulation loop (needs >= {actual_sim_loop_start_idx_in_segment+1} for attention history).")
    else:
        for i_sim_segment in range(actual_sim_loop_start_idx_in_segment, len(data_scaled_features_np_segment)):
            if not tr.is_alive:
                final_val_idx = i_sim_segment
                break

            current_step_features_segment = data_scaled_features_np_segment[i_sim_segment]

            start_seq_idx_in_segment = i_sim_segment - ATTENTION_SEQUENCE_LENGTH + 1
            sequence_for_attention_segment_np = data_scaled_features_np_segment[start_seq_idx_in_segment : i_sim_segment + 1]

            attention_context_np = get_attention_output(
                sequence_for_attention_segment_np,
                current_seq_len=sequence_for_attention_segment_np.shape[0],
                target_seq_len=ATTENTION_SEQUENCE_LENGTH,
                feature_dim=num_input_features_from_data_global
            )

            current_price = data_raw_prices_df_segment.iloc[i_sim_segment][COL_CLOSE]
            current_ts = data_raw_prices_df_segment.index[i_sim_segment]
            trader_state_inputs = tr.get_state_for_nn(current_price, MAX_EXPECTED_CREDIT, MAX_EXPECTED_HOLDINGS_VALUE)

            nn_in = np.concatenate((current_step_features_segment, attention_context_np.flatten(), trader_state_inputs))
            action_raw, amount_raw = net.activate(nn_in)
            amount_to_use = np.clip(amount_raw, 0.01, 1.0)

            if action_raw > 0.55:
                if tr.buy(amount_to_use * tr.credit, current_price, current_ts): buys+=1
            elif action_raw < 0.45:
                if tr.sell(amount_to_use * tr.holdings_shares, current_price, current_ts): sells+=1

            tr.update_history(current_ts, current_price)
            final_val_idx = i_sim_segment

    final_val_pf = INITIAL_STARTING_CAPITAL
    if final_val_idx >= 0 and final_val_idx < len(data_raw_prices_df_segment) and not data_raw_prices_df_segment.empty :
         final_val_pf = tr.get_portfolio_value(data_raw_prices_df_segment.iloc[final_val_idx][COL_CLOSE])
    elif not data_raw_prices_df_segment.empty:
        final_val_pf = tr.get_portfolio_value(data_raw_prices_df_segment.iloc[-1][COL_CLOSE])

    print(f"{title_prefix} - Initial Capital: ${INITIAL_STARTING_CAPITAL:.2f}, Final Portfolio Value: ${final_val_pf:.2f}, Final Credit: ${tr.credit:.2f}")
    print(f"  Trades Logged: {len(tr.trade_log)} (Sim Buys: {buys}, Sells: {sells}), Realized PnL: ${tr.realized_gains_this_evaluation:.2f}, Total Fees Paid: ${tr.total_fees_paid:.2f}")

    profit_abs = final_val_pf - INITIAL_STARTING_CAPITAL
    profit_percentage_on_segment = 0.0
    if INITIAL_STARTING_CAPITAL > 1e-6:
        profit_percentage_on_segment = (profit_abs / INITIAL_STARTING_CAPITAL) * 100
        print(f"  Profit/Loss (Portfolio on this data segment): {profit_percentage_on_segment:.2f}%")
    else:
        print(f"  Profit/Loss (Portfolio on this data segment): N/A (Initial capital too small for percentage)")

    if is_validation_run and not data_raw_prices_df_segment.empty and INITIAL_STARTING_CAPITAL > 1e-6:
        start_time_val_segment = data_raw_prices_df_segment.index.min()
        end_time_val_segment = data_raw_prices_df_segment.index.max()
        duration_seconds_val_segment = (end_time_val_segment - start_time_val_segment).total_seconds()
        
        sim_duration_for_projection_text = f"Segment Duration: {pd.to_timedelta(duration_seconds_val_segment, unit='s')}"
        if actual_sim_loop_start_idx_in_segment >= len(data_scaled_features_np_segment):
            sim_duration_for_projection_text += " (Sim Loop Skipped - Data Too Short)"

        print(f"  Validation Data Timeframe: {start_time_val_segment.strftime('%Y-%m-%d %H:%M')} to {end_time_val_segment.strftime('%Y-%m-%d %H:%M')} ({sim_duration_for_projection_text})")

        if duration_seconds_val_segment > (60 * 30) and actual_sim_loop_start_idx_in_segment < len(data_scaled_features_np_segment) :
            profit_ratio_for_projection = profit_abs / INITIAL_STARTING_CAPITAL
            seconds_in_week = 7 * 24 * 3600; seconds_in_month = 30 * 24 * 3600
            
            projected_weekly_return_pct_val = -100.0
            projected_monthly_return_pct_val = -100.0

            if 1 + profit_ratio_for_projection > 0:
                periods_in_week_val = seconds_in_week / duration_seconds_val_segment
                periods_in_month_val = seconds_in_month / duration_seconds_val_segment
                projected_weekly_return_pct_val = (math.pow(1 + profit_ratio_for_projection, periods_in_week_val) - 1) * 100
                projected_monthly_return_pct_val = (math.pow(1 + profit_ratio_for_projection, periods_in_month_val) - 1) * 100
            
            print(f"  Projected Performance (based on this validation segment, COMPOUNDED):")
            print(f"    Projected Weekly Return: {projected_weekly_return_pct_val:.2f}%")
            print(f"    Projected Monthly Return: {projected_monthly_return_pct_val:.2f}%")
            if abs(projected_weekly_return_pct_val) > 1000 or abs(projected_monthly_return_pct_val) > 5000 :
                 print("    WARNING (Validation Proj): Projections are very high/low. Interpret with extreme caution.")
        else:
            print(f"  Validation period ({pd.to_timedelta(duration_seconds_val_segment, unit='s')}) or sim data too short for meaningful weekly/monthly projection.")

    plot_backtest_results(data_raw_prices_df_segment, tr.trade_log, tr.history, f"{title_prefix} Results for {TICKER}", COL_CLOSE)

def run_neat_trader(config_file):
    # This function is correct and does not need changes.
    global train_data_scaled_np_global, train_data_raw_prices_global, \
           current_eval_window_start_index, max_final_portfolio_record_global, \
           best_record_breaker_details, current_eval_window_raw_data_for_plotting, \
           num_input_features_from_data_global, FITNESS_THRESHOLD_CONFIG_FROM_FILE

    raw_df_full = fetch_data(TICKER, DATA_PERIOD, DATA_INTERVAL)
    if raw_df_full.empty: print("ERROR: No data fetched. Exiting."); return
    feats_df_full = calculate_features(raw_df_full.copy())
    if feats_df_full.empty: print("ERROR: Feature calculation resulted in empty dataframe. Exiting."); return
    feats_lags_df_full = add_lagged_features(feats_df_full, N_LAGS)
    if feats_lags_df_full.empty: print("ERROR: Adding lags resulted in empty dataframe. Exiting."); return

    min_data_length_needed_overall = ATTENTION_SEQUENCE_LENGTH + N_LAGS + EVAL_WINDOW_SIZE_MINUTES + 60
    if len(feats_lags_df_full) < min_data_length_needed_overall :
        print(f"ERROR: Data too short after lags ({len(feats_lags_df_full)}) for processing. Need at least {min_data_length_needed_overall} rows. Exiting.")
        return

    train_feature_df, val_feature_df = split_data_by_days(feats_lags_df_full, TRAIN_DAYS)
    if train_feature_df is None or train_feature_df.empty:
        print("ERROR: Training feature dataframe is empty or None after split. Exiting."); return
    if val_feature_df is None:
        val_feature_df = pd.DataFrame(columns=train_feature_df.columns, index=pd.to_datetime([]))
        val_feature_df = val_feature_df.astype(train_feature_df.dtypes)

    min_train_data_len_for_sliding_window = ATTENTION_SEQUENCE_LENGTH + EVAL_WINDOW_SIZE_MINUTES + 10
    if len(train_feature_df) < min_train_data_len_for_sliding_window:
        print(f"ERROR: Training data ({len(train_feature_df)}) too short after split for evaluation windows. Needs {min_train_data_len_for_sliding_window}. Exiting.")
        return

    train_data_raw_prices_global = raw_df_full.loc[train_feature_df.index, [COL_CLOSE]].copy()
    val_data_raw_prices_df = raw_df_full.loc[val_feature_df.index, [COL_CLOSE]].copy() if not val_feature_df.empty else pd.DataFrame(columns=[COL_CLOSE])

    if train_data_raw_prices_global.empty :
        print("ERROR: Training raw prices dataframe is empty. This should not happen if train_feature_df was populated."); return

    train_scaled_df, val_scaled_df, scaler_obj = normalize_data(
        train_feature_df.copy(),
        val_feature_df.copy() if not val_feature_df.empty else pd.DataFrame(columns=train_feature_df.columns, index=val_feature_df.index)
    )
    
    train_data_scaled_np_global = train_scaled_df.drop(columns=['Close'], errors='ignore').to_numpy()
    val_scaled_np_features_only = val_scaled_df.drop(columns=['Close'], errors='ignore').to_numpy() if not val_scaled_df.empty else np.array([])

    if len(train_data_scaled_np_global) == 0:
        print("ERROR: Scaled training features (train_data_scaled_np_global) are empty. Exiting."); return

    num_input_features_from_data_global = train_data_scaled_np_global.shape[1]
    num_trader_state_features = 4

    initialize_shared_attention(
        input_dim=num_input_features_from_data_global,
        attention_dim=ATTENTION_OUTPUT_DIM,
        attention_heads=ATTENTION_HEADS,
        dropout_rate=0.1
    )

    total_nn_inputs = num_input_features_from_data_global + ATTENTION_OUTPUT_DIM + num_trader_state_features
    print(f"Number of input features from data (current step): {num_input_features_from_data_global}")
    print(f"Number of attention output features (context vector): {ATTENTION_OUTPUT_DIM}")
    print(f"Number of trader state features: {num_trader_state_features}")
    print(f"Total NN inputs: {total_nn_inputs}. CONFIG FILE ('{CONFIG_FILE_PATH}') 'num_inputs' SHOULD BE: {total_nn_inputs}.")

    cfg = neat.Config(neat.DefaultGenome, neat.DefaultReproduction, neat.DefaultSpeciesSet, neat.DefaultStagnation, config_file)

    if hasattr(cfg.genome_config, 'fitness_threshold'):
        FITNESS_THRESHOLD_CONFIG_FROM_FILE = cfg.genome_config.fitness_threshold
    cfg.genome_config.fitness_threshold = FITNESS_THRESHOLD_CONFIG_FROM_FILE
    print(f"Fitness threshold set to: {FITNESS_THRESHOLD_CONFIG_FROM_FILE}")

    if cfg.genome_config.num_inputs != total_nn_inputs:
        print(f"CRITICAL CONFIG ERROR: 'num_inputs' in NEAT config file is {cfg.genome_config.num_inputs}, "
              f"but script calculated {total_nn_inputs}. PLEASE UPDATE THE CONFIG FILE and restart.")
        return

    max_final_portfolio_record_global = INITIAL_STARTING_CAPITAL
    current_eval_window_start_index = 0
    best_record_breaker_details = { "genome_obj": None, "window_fitness": -float('inf'), "portfolio_achieved_on_full_train": INITIAL_STARTING_CAPITAL}
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

    checkpointer_filename_prefix = f'neat_outputs/neat-checkpoint-{TICKER.replace("/", "_")}-'
    os.makedirs(os.path.dirname(checkpointer_filename_prefix), exist_ok=True)
    checkpointer = neat.Checkpointer(
        generation_interval=10,
        time_interval_seconds=3600,
        filename_prefix=checkpointer_filename_prefix
    )
    pop.add_reporter(checkpointer)

    print("\nStarting NEAT evolution with Uncompromising Profit Engine V6 Fitness Function...");
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
        if gen_reporter:
            gen_reporter.plot_generational_metrics()

    best_genome_overall = None
    best_fitness_overall = -float('inf')
    source_of_best = "None"

    if winner is not None and hasattr(winner, 'fitness') and winner.fitness is not None:
        if winner.fitness >= cfg.genome_config.fitness_threshold:
            best_genome_overall = winner
            best_fitness_overall = winner.fitness
            source_of_best = "NEAT pop.run() returned winner (threshold met)"

    if gen_reporter and gen_reporter.neat_overall_best_genome_obj:
        if best_genome_overall is None or gen_reporter.neat_best_fitness_so_far > best_fitness_overall:
            best_genome_overall = gen_reporter.neat_overall_best_genome_obj
            best_fitness_overall = gen_reporter.neat_best_fitness_so_far
            source_of_best = "GenerationReporter's Overall Best (by highest window fitness)"
    
    if stats and hasattr(stats, 'best_genome') and callable(stats.best_genome):
        try:
            stats_best = stats.best_genome()
            if stats_best and hasattr(stats_best, 'fitness') and stats_best.fitness is not None:
                if best_genome_overall is None or stats_best.fitness > best_fitness_overall:
                    best_genome_overall = stats_best
                    best_fitness_overall = stats_best.fitness
                    source_of_best = "StatisticsReporter's Best Genome Overall"
        except IndexError:
            pass
    
    if best_genome_overall is None and hasattr(pop, 'population') and pop.population:
        current_pop_genomes = list(pop.population.values())
        final_pop_best_fitness = -float('inf')
        final_pop_best_genome = None
        for g_iter in current_pop_genomes:
            if hasattr(g_iter, 'fitness') and g_iter.fitness is not None and g_iter.fitness > final_pop_best_fitness:
                final_pop_best_fitness = g_iter.fitness
                final_pop_best_genome = g_iter
        if final_pop_best_genome:
            if best_genome_overall is None or final_pop_best_fitness > best_fitness_overall:
                best_genome_overall = final_pop_best_genome
                best_fitness_overall = final_pop_best_fitness
                source_of_best = "Best from Final Population (Ultimate Fallback)"

    fitness_display_str = f"{best_fitness_overall:.2f}" if isinstance(best_fitness_overall, (int, float)) and not (math.isinf(best_fitness_overall) or math.isnan(best_fitness_overall)) else str(best_fitness_overall)
    print(f"\nOverall Best Genome Selected for Final Evaluation (Source: {source_of_best}):")
    if best_genome_overall:
        print(f"  ID: {best_genome_overall.key}, Contextual Fitness (Window-based): {fitness_display_str}")
    else:
        print("  No suitable genome found after evolution.")

    if best_genome_overall:
        output_dir = "neat_outputs"
        winner_filename = f"winner_genome_attention_{TICKER.replace('/', '_')}_UncompromisingV6.pkl"
        winner_path = os.path.join(output_dir, winner_filename)
        with open(winner_path, "wb") as f:
            pickle.dump(best_genome_overall, f)
        print(f"Saved overall best genome to {winner_path}")

        print("\n--- Final Evaluation of Selected Overall Best Genome ---")
        if train_data_scaled_np_global is not None and len(train_data_scaled_np_global) > 0 and \
           train_data_raw_prices_global is not None and not train_data_raw_prices_global.empty:
            run_simulation_and_plot(best_genome_overall, cfg,
                                    train_data_scaled_np_global,
                                    train_data_raw_prices_global,
                                    "Selected Best Genome on Full Training Data")
        else:
            print("Skipping final training data evaluation: insufficient data length or missing data.")

        if val_scaled_np_features_only is not None and len(val_scaled_np_features_only) > 0 and \
           val_data_raw_prices_df is not None and not val_data_raw_prices_df.empty:
            run_simulation_and_plot(best_genome_overall, cfg,
                                    val_scaled_np_features_only,
                                    val_data_raw_prices_df,
                                    "Selected Best Genome on Validation Data",
                                    is_validation_run=True)
        else:
            print("INFO: No validation data was available or it was too short for final evaluation.")
    else:
        print("No best genome found to evaluate after evolution.")

if __name__ == "__main__":
    os.makedirs("neat_outputs", exist_ok=True)

    if not os.path.exists(CONFIG_FILE_PATH):
        print(f"Config file not found: {CONFIG_FILE_PATH}. Please ensure it exists.")
    else:
        run_neat_trader(CONFIG_FILE_PATH)
