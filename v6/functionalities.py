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
    fig.add_trace(go.Scatter(x=df_price_data.index, y=df_price_data[price_col_name],
                             mode='lines', name='Stock Price',
                             line=dict(color='blue')), row=1, col=1)

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
        else:
            print("Plot Info: trade_log was empty or malformed for plotting signals.")
    else:
        print("Plot Info: trade_log was empty for plotting signals.")
    
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
        else:
            print("Plot Info: history_log was empty or malformed for plotting portfolio/credit.")
    else:
        print("Plot Info: history_log was empty for plotting portfolio/credit.")


    fig.update_layout(title_text=title, height=900, showlegend=True)
    fig.update_xaxes(title_text="Date", row=3, col=1) 
    fig.update_yaxes(title_text="Price", row=1, col=1)
    fig.update_yaxes(title_text="Portfolio Value ($)", row=2, col=1)
    fig.update_yaxes(title_text="Liquid Credit ($)", row=3, col=1)
    
    # Suppress a common Plotly warning about large number of discrete colors if legend has many items (not typical here)
    # import warnings
    # from plotly.graph_objs.layout import Legend
    # if isinstance(fig.layout.legend, Legend): # Check if legend object exists
    #    with warnings.catch_warnings():
    #        warnings.simplefilter("ignore", UserWarning) # This can be too broad
    #        try:
    #            fig.show() # Show the plot
    #        except Exception as e:
    #            print(f"Error showing plot: {e}")
    # else:
    fig.show()


def plot_generational_performance(generations, metrics_history_dict, title="Performance Metrics Over Generations"):
    num_metrics = len(metrics_history_dict)
    if num_metrics == 0 or not generations:
        print("No metrics or generations to plot for generational performance.")
        return

    # Determine rows and columns for subplots, aim for 2 columns if many metrics
    if num_metrics <= 4:
        plot_cols = 1
        plot_rows = num_metrics
    else:
        plot_cols = 2
        plot_rows = (num_metrics + 1) // 2


    fig = make_subplots(rows=plot_rows, cols=plot_cols, shared_xaxes=True,
                        subplot_titles=list(metrics_history_dict.keys()),
                        vertical_spacing=0.1 if plot_rows > 1 else 0, # More spacing if multiple rows
                        horizontal_spacing=0.1 if plot_cols > 1 else 0)
    
    current_row, current_col = 1, 1
    for i, (metric_name, values) in enumerate(metrics_history_dict.items()):
        # Ensure 'values' has same length as 'generations' by padding with NaN if necessary
        # This can happen if a generation had no valid genome to report.
        if len(values) < len(generations):
            values = values + [np.nan] * (len(generations) - len(values))
        elif len(values) > len(generations): # Should not happen if logic is correct
            values = values[:len(generations)]


        fig.add_trace(go.Scatter(x=generations, y=values, mode='lines+markers', name=metric_name),
                      row=current_row, col=current_col)
        # fig.update_yaxes(title_text=metric_name, row=current_row, col=current_col) # Y-axis title often redundant with subplot title

        current_col += 1
        if current_col > plot_cols:
            current_col = 1
            current_row += 1
    
    # Add X-axis title to the bottom-most row(s)
    for r in range(plot_rows):
        if r == plot_rows -1 : # Only the last row
            for c in range(plot_cols):
                 # Calculate subplot index for x_axis update
                # This depends on how Plotly numbers its axes internally for subplots.
                # Usually fig.update_xaxes(title_text="Generation", row=plot_rows, col=c+1) works per column in last row
                pass # Generally, shared_xaxes + one title on fig.update_layout is enough


    fig.update_layout(
        title_text=title, 
        height=max(400, 200 * plot_rows), # Adjust height based on number of rows
        showlegend=False # Individual traces are named by subplot titles
    )
    # Common x-axis title
    fig.update_xaxes(title_text="Generation", matches='x') # Apply to all shared x-axes

    fig.show()
