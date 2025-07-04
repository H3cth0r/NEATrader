# functionalities.py
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pandas as pd
import numpy as np # For np.nan handling in generational plots

def plot_backtest_results(df_price_data, trade_log, history_log, title="Backtest Results", price_col_name='Close'):
    fig = make_subplots(rows=3, cols=1, shared_xaxes=True, vertical_spacing=0.07,
                        subplot_titles=("Stock Price & Executed Trades", "Total Portfolio Value", "Liquid Credit"),
                        row_heights=[0.6, 0.2, 0.2])

    # Plot 1: Stock Price and Trades
    if not df_price_data.empty and price_col_name in df_price_data.columns:
        fig.add_trace(go.Scatter(x=df_price_data.index, y=df_price_data[price_col_name],
                                mode='lines', name='Stock Price',
                                line=dict(color='blue')), row=1, col=1)
    else:
        print(f"Plot Info: Price data for '{price_col_name}' is empty or column missing for '{title}'.")


    buys_plotted = 0
    sells_plotted = 0
    if trade_log:
        trades_df = pd.DataFrame(trade_log)
        if not trades_df.empty and 'timestamp' in trades_df.columns and 'price' in trades_df.columns and 'type' in trades_df.columns:
            trades_df['timestamp'] = pd.to_datetime(trades_df['timestamp'])

            buy_signals = trades_df[trades_df['type'] == 'buy']
            sell_signals = trades_df[trades_df['type'] == 'sell']

            if not buy_signals.empty:
                fig.add_trace(go.Scatter(x=buy_signals['timestamp'], y=buy_signals['price'], mode='markers', name='Executed Buy', marker=dict(color='green', size=8, symbol='triangle-up')), row=1, col=1)
                buys_plotted = len(buy_signals)
            if not sell_signals.empty:
                fig.add_trace(go.Scatter(x=sell_signals['timestamp'], y=sell_signals['price'], mode='markers', name='Executed Sell', marker=dict(color='red', size=8, symbol='triangle-down')), row=1, col=1)
                sells_plotted = len(sell_signals)
        # else: # Commented out to reduce console noise if trades_df is just empty
            # print("Plot Info: trade_log was empty or malformed for plotting signals.")
    # else: # Commented out to reduce console noise
        # print("Plot Info: trade_log was empty for plotting signals.")

    # print(f"Plot Debug: Plotted {buys_plotted} buys, {sells_plotted} sells for '{title}'")


    # Plot 2: Portfolio Value & Plot 3: Liquid Credit
    if history_log:
        history_df = pd.DataFrame(history_log)
        if not history_df.empty and 'timestamp' in history_df.columns:
            history_df['timestamp'] = pd.to_datetime(history_df['timestamp'])
            if 'portfolio_value' in history_df.columns:
                fig.add_trace(go.Scatter(x=history_df['timestamp'], y=history_df['portfolio_value'], mode='lines', name='Portfolio Value', line=dict(color='purple')), row=2, col=1)
            if 'credit' in history_df.columns:
                fig.add_trace(go.Scatter(x=history_df['timestamp'], y=history_df['credit'], mode='lines', name='Liquid Credit', line=dict(color='orange')), row=3, col=1)
        # else: # Commented out
            # print("Plot Info: history_log was empty or malformed for plotting portfolio/credit.")
    # else: # Commented out
        # print("Plot Info: history_log was empty for plotting portfolio/credit.")


    fig.update_layout(title_text=title, height=900, showlegend=True)
    fig.update_xaxes(title_text="Date", row=3, col=1)
    fig.update_yaxes(title_text="Price", row=1, col=1)
    fig.update_yaxes(title_text="Portfolio Value ($)", row=2, col=1)
    fig.update_yaxes(title_text="Liquid Credit ($)", row=3, col=1)

    try:
        fig.show()
    except Exception as e:
        print(f"Error displaying plot '{title}': {e}. Ensure you have a graphical environment if not running in a notebook.")


def plot_generational_performance(generations, metrics_history_dict, title="Performance Metrics Over Generations"):
    num_metrics = len(metrics_history_dict)
    if num_metrics == 0 or not generations:
        print("No metrics or generations to plot for generational performance.")
        return

    if num_metrics <= 4:
        plot_cols = 1
        plot_rows = num_metrics
    else:
        plot_cols = 2
        plot_rows = (num_metrics + 1) // 2


    fig = make_subplots(rows=plot_rows, cols=plot_cols, shared_xaxes=True,
                        subplot_titles=list(metrics_history_dict.keys()),
                        vertical_spacing=0.1 if plot_rows > 1 else 0,
                        horizontal_spacing=0.1 if plot_cols > 1 else 0)

    current_row, current_col = 1, 1
    for i, (metric_name, values) in enumerate(metrics_history_dict.items()):
        # Pad with NaN if lengths don't match (can happen if some generations had no valid genomes)
        if len(values) < len(generations):
            values_padded = values + [np.nan] * (len(generations) - len(values))
        elif len(values) > len(generations): # Should not happen if logic is correct
            print(f"Warning: Metric '{metric_name}' has more values ({len(values)}) than generations ({len(generations)}). Truncating.")
            values_padded = values[:len(generations)]
        else:
            values_padded = values

        fig.add_trace(go.Scatter(x=generations, y=values_padded, mode='lines+markers', name=metric_name),
                      row=current_row, col=current_col)

        current_col += 1
        if current_col > plot_cols:
            current_col = 1
            current_row += 1

    fig.update_layout(
        title_text=title,
        height=max(400, 200 * plot_rows),
        showlegend=False
    )
    fig.update_xaxes(title_text="Generation", matches='x')

    try:
        fig.show()
    except Exception as e:
        print(f"Error displaying generational plot '{title}': {e}. Ensure you have a graphical environment.")
