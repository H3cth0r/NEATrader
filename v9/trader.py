import numpy as np
import pandas as pd

class Trader:
    def __init__(self, initial_credit=1000, initial_holdings=0, trading_fee_percent=0.001):
        self.initial_credit = initial_credit
        self.initial_holdings_shares = initial_holdings
        self.trading_fee_percent = trading_fee_percent

        self.credit = initial_credit
        self.holdings_shares = self.initial_holdings_shares
        self.is_alive = True

        self.total_cost_of_holdings = 0.0
        self.history = []
        self.trade_log = []
        self.realized_gains_this_evaluation = 0.0
        self.max_portfolio_value_achieved = initial_credit
        self.total_fees_paid = 0.0
        
        # This tracks steps since the last trade to give the agent a sense of time/duration.
        self.steps_since_last_trade = 0
        
        # NEW: Counter for winning trades
        self.winning_sells = 0


    def reset(self):
        self.credit = self.initial_credit
        self.holdings_shares = self.initial_holdings_shares
        self.is_alive = True
        self.history = []
        self.trade_log = []
        self.total_cost_of_holdings = 0.0
        self.realized_gains_this_evaluation = 0.0
        self.max_portfolio_value_achieved = self.get_portfolio_value(0)
        if self.holdings_shares == 0: self.max_portfolio_value_achieved = self.initial_credit
        self.total_fees_paid = 0.0
        self.steps_since_last_trade = 0
        # NEW: Reset counter
        self.winning_sells = 0

    def _check_alive(self):
        # A trader is considered "dead" or ruined if it has virtually no cash and no shares.
        min_meaningful_credit_abs = 0.01
        if self.credit < min_meaningful_credit_abs and self.holdings_shares < 1e-8:
            self.is_alive = False
        return self.is_alive

    def get_average_buy_price(self):
        if self.holdings_shares <= 1e-9: return 0.0
        return self.total_cost_of_holdings / self.holdings_shares

    def buy(self, cash_amount_to_spend_requested, price_per_share, timestamp, return_trade_profit=False):
        if not self.is_alive or price_per_share <= 1e-6: return False
        if cash_amount_to_spend_requested <= 1e-6: return False

        max_spend_on_asset_possible = self.credit / (1 + self.trading_fee_percent)
        actual_cash_to_spend_on_asset = min(cash_amount_to_spend_requested, max_spend_on_asset_possible)

        if actual_cash_to_spend_on_asset < 1e-3: return False

        fee_for_this_buy = actual_cash_to_spend_on_asset * self.trading_fee_percent
        total_cost_of_transaction = actual_cash_to_spend_on_asset + fee_for_this_buy

        if total_cost_of_transaction > self.credit + 1e-7: return False

        shares_to_buy = actual_cash_to_spend_on_asset / price_per_share
        if shares_to_buy < 1e-8: return False

        self.credit -= total_cost_of_transaction
        self.total_fees_paid += fee_for_this_buy
        self.total_cost_of_holdings += actual_cash_to_spend_on_asset
        self.holdings_shares += shares_to_buy

        self.trade_log.append({
            'timestamp': timestamp, 'type': 'buy', 'price': price_per_share,
            'amount_cash': actual_cash_to_spend_on_asset, 'shares': shares_to_buy, 'fee': fee_for_this_buy
        })
        self.steps_since_last_trade = 0 # Reset counter on trade
        self._check_alive()
        return True

    def sell(self, shares_to_sell_requested, price_per_share, timestamp, return_trade_profit=False):
        if not self.is_alive or price_per_share <= 1e-6: return False

        actual_shares_to_sell = min(shares_to_sell_requested, self.holdings_shares)
        if actual_shares_to_sell <= 1e-8: return False

        gross_cash_from_sale = actual_shares_to_sell * price_per_share
        fee_for_this_sell = gross_cash_from_sale * self.trading_fee_percent
        net_cash_gained = gross_cash_from_sale - fee_for_this_sell

        avg_buy_price_of_sold_shares = self.get_average_buy_price()
        cost_of_shares_sold = avg_buy_price_of_sold_shares * actual_shares_to_sell
        
        profit_from_this_sell = net_cash_gained - cost_of_shares_sold
        self.realized_gains_this_evaluation += profit_from_this_sell
        
        # NEW: Check if this was a winning trade
        if profit_from_this_sell > 0:
            self.winning_sells += 1

        self.credit += net_cash_gained
        self.total_fees_paid += fee_for_this_sell

        if abs(self.holdings_shares - actual_shares_to_sell) < 1e-9:
            self.total_cost_of_holdings = 0.0
            self.holdings_shares = 0.0
        else:
            proportion_sold = actual_shares_to_sell / self.holdings_shares if self.holdings_shares > 1e-9 else 0
            self.total_cost_of_holdings *= (1 - proportion_sold)
            self.holdings_shares -= actual_shares_to_sell
            if self.total_cost_of_holdings < 0: self.total_cost_of_holdings = 0.0

        if self.holdings_shares < 1e-9: self.holdings_shares = 0.0

        self.trade_log.append({
            'timestamp': timestamp, 'type': 'sell', 'price': price_per_share,
            'amount_cash': gross_cash_from_sale,
            'net_cash_gained': net_cash_gained,
            'shares': actual_shares_to_sell,
            'profit': profit_from_this_sell,
            'fee': fee_for_this_sell
        })
        self.steps_since_last_trade = 0 # Reset counter on trade
        self._check_alive()
        if return_trade_profit:
            return profit_from_this_sell
        return True

    def hold(self):
        pass

    def update_history(self, timestamp, current_price):
        self.steps_since_last_trade += 1 # Increment counter each step
        portfolio_value = self.get_portfolio_value(current_price)
        self.history.append({
            'timestamp': timestamp, 'credit': self.credit, 'holdings_shares': self.holdings_shares,
            'price': current_price, 'portfolio_value': portfolio_value,
            'avg_buy_price': self.get_average_buy_price(),
            'total_fees_paid': self.total_fees_paid,
            'realized_pnl_eval': self.realized_gains_this_evaluation
        })
        if portfolio_value > self.max_portfolio_value_achieved:
            self.max_portfolio_value_achieved = portfolio_value

        # Add another condition for being "dead" - catastrophic loss
        if portfolio_value < (self.initial_credit * 0.1): # e.g., if value drops below 10% of start
             self.is_alive = False

    def get_portfolio_value(self, current_price):
        if current_price <= 1e-6: return self.credit
        return self.credit + (self.holdings_shares * current_price)

    def get_state_for_nn(self, current_price, max_possible_credit, max_possible_holdings_value):
        norm_credit = self.credit / max_possible_credit if max_possible_credit > 1e-6 else 0

        current_holdings_value = self.holdings_shares * current_price
        norm_holdings_value = current_holdings_value / max_possible_holdings_value if max_possible_holdings_value > 1e-6 else 0

        unrealized_pl_percentage = 0.0
        if self.holdings_shares > 1e-8 and current_price > 1e-6:
            avg_buy = self.get_average_buy_price()
            if avg_buy > 1e-6:
                unrealized_pl_percentage = (current_price - avg_buy) / avg_buy
        
        # Clip to a reasonable range like [-1, 2] to prevent extreme values
        norm_unrealized_pl = np.clip(unrealized_pl_percentage, -1.0, 2.0)

        # Normalize steps since last trade. This helps the agent learn time-based strategies.
        # Max steps can be set to a value like the evaluation window size.
        max_steps_in_position = 4 * 60 # e.g., 4 hours of 1-min data
        norm_steps_since_trade = min(self.steps_since_last_trade / max_steps_in_position, 1.0)

        return [ np.clip(norm_credit, 0, 1), 
                 np.clip(norm_holdings_value, 0, 1), 
                 norm_unrealized_pl,
                 norm_steps_since_trade ]
