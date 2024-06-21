import plotly.graph_objs as go
import plotly.io as pio
from plotly.subplots import make_subplots

def plot_live_data(data, trader, step):
    # Create subplots
    fig = make_subplots(rows=3, cols=1, shared_xaxes=True, 
                        subplot_titles=("Stock Close Price", "Trader Credit", "Trader Holdings"))

    # Plot the stock "Close" price
    close_trace = go.Scatter(
        x=list(range(len(data))),
        y=data,
        mode='lines',
        name='Close Price'
    )
    fig.add_trace(close_trace, row=1, col=1)

    # Plot the trader's credit history
    credit_trace = go.Scatter(
        x=list(range(len(trader.credit_history))),
        y=trader.credit_history,
        mode='lines',
        name='Trader Credit'
    )
    fig.add_trace(credit_trace, row=2, col=1)

    # Plot the trader's holdings history
    holdings_trace = go.Scatter(
        x=list(range(len(trader.holdings_history))),
        y=trader.holdings_history,
        mode='lines',
        name='Trader Holdings'
    )
    fig.add_trace(holdings_trace, row=3, col=1)

    fig.update_layout(
        title=f'Step {step} - Stock Close Price and Trader Performance',
        height=900,
        showlegend=False
    )

    fig.update_xaxes(title_text="Time", row=3, col=1)
    fig.update_yaxes(title_text="Close Price", row=1, col=1)
    fig.update_yaxes(title_text="Credit", row=2, col=1)
    fig.update_yaxes(title_text="Holdings", row=3, col=1)

    pio.show(fig)
