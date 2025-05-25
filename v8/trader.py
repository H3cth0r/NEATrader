# trader.py
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

    def reset(self):
        self.credit = self.initial_credit
        self.holdings_shares = self.initial_holdings_shares
        self.is_alive = True
        self.history = []
        self.trade_log = []
        self.total_cost_of_holdings = 0.0
        self.realized_gains_this_evaluation = 0.0
        current_price_at_reset_effectively_zero_for_holdings = 0 
        self.max_portfolio_value_achieved = self.credit + (self.holdings_shares * current_price_at_reset_effectively_zero_for_holdings)
        self.total_fees_paid = 0.0

    def _check_alive(self):
        min_meaningful_credit = self.initial_credit * 0.0001 # Reduced slightly
        self.is_alive = self.credit > min_meaningful_credit or self.holdings_shares > 1e-8 
        if self.credit <= 1e-7 and self.holdings_shares <= 1e-8: # If truly nothing left
            self.is_alive = False
        # Also consider if portfolio value is extremely low
        # current_price would be needed here, or do this check in update_history
        # For now, rely on credit and shares.
        return self.is_alive

    def get_average_buy_price(self):
        if self.holdings_shares <= 1e-9: return 0.0
        return self.total_cost_of_holdings / self.holdings_shares

    def buy(self, cash_amount_to_spend_requested, price_per_share, timestamp, return_trade_profit=False): # Added return_trade_profit
        if not self.is_alive: return False if not return_trade_profit else 0
        if cash_amount_to_spend_requested <= 1e-6 : return False if not return_trade_profit else 0
        
        max_spend_on_asset_possible = self.credit / (1 + self.trading_fee_percent)
        actual_cash_to_spend_on_asset = min(cash_amount_to_spend_requested, max_spend_on_asset_possible)
        
        if actual_cash_to_spend_on_asset < 1e-3 : 
            return False if not return_trade_profit else 0

        fee_for_this_buy = actual_cash_to_spend_on_asset * self.trading_fee_percent
        total_cost_of_transaction = actual_cash_to_spend_on_asset + fee_for_this_buy

        if total_cost_of_transaction > self.credit + 1e-6: 
            return False if not return_trade_profit else 0

        if price_per_share <= 1e-6: return False if not return_trade_profit else 0

        shares_to_buy = actual_cash_to_spend_on_asset / price_per_share
        if shares_to_buy < 1e-7: return False if not return_trade_profit else 0

        self.credit -= total_cost_of_transaction
        self.total_fees_paid += fee_for_this_buy
        
        self.total_cost_of_holdings += actual_cash_to_spend_on_asset # Cost is just asset value, fee is separate
        self.holdings_shares += shares_to_buy
        
        self.trade_log.append({
            'timestamp': timestamp, 'type': 'buy', 'price': price_per_share,
            'amount_cash': actual_cash_to_spend_on_asset, 'shares': shares_to_buy, 'fee': fee_for_this_buy
        })
        self._check_alive()
        if return_trade_profit:
            return 0 # Buys don't realize profit/loss immediately for profit factor calculation
        return True

    def sell(self, shares_to_sell_requested, price_per_share, timestamp, return_trade_profit=False): # Added return_trade_profit
        if not self.is_alive: return False if not return_trade_profit else 0

        actual_shares_to_sell = min(shares_to_sell_requested, self.holdings_shares)
        if actual_shares_to_sell <= 1e-7 or price_per_share <= 1e-6: 
            return False if not return_trade_profit else 0
            
        gross_cash_from_sale = actual_shares_to_sell * price_per_share
        fee_for_this_sell = gross_cash_from_sale * self.trading_fee_percent
        net_cash_gained = gross_cash_from_sale - fee_for_this_sell
        
        avg_buy_price_of_sold_shares = self.get_average_buy_price()
        # Cost of shares sold is based on the average buy price of *all* current holdings
        cost_of_shares_sold = avg_buy_price_of_sold_shares * actual_shares_to_sell
        
        profit_from_this_sell = net_cash_gained - cost_of_shares_sold # This is the actual realized P&L for these shares
        self.realized_gains_this_evaluation += profit_from_this_sell

        self.credit += net_cash_gained
        self.total_fees_paid += fee_for_this_sell

        if abs(self.holdings_shares - actual_shares_to_sell) < 1e-9 : 
            self.total_cost_of_holdings = 0.0
            self.holdings_shares = 0.0 
        else:
            # Reduce total_cost_of_holdings proportionally to shares sold
            # This maintains the average buy price for remaining shares
            self.total_cost_of_holdings *= (1 - (actual_shares_to_sell / self.holdings_shares) if self.holdings_shares > 1e-9 else 0)
            self.holdings_shares -= actual_shares_to_sell
            if self.total_cost_of_holdings < 0: self.total_cost_of_holdings = 0.0
        
        if self.holdings_shares < 1e-9: self.holdings_shares = 0.0

        self.trade_log.append({
            'timestamp': timestamp, 'type': 'sell', 'price': price_per_share,
            'amount_cash': gross_cash_from_sale, # Gross cash before fee
            'net_cash_gained': net_cash_gained,  # Net cash after fee
            'shares': actual_shares_to_sell,
            'profit': profit_from_this_sell,     # Realized profit/loss from this specific sale
            'fee': fee_for_this_sell
        })
        self._check_alive()
        if return_trade_profit:
            return profit_from_this_sell # Return the profit/loss of this specific trade
        return True

    def hold(self): pass 

    def update_history(self, timestamp, current_price):
        portfolio_value = self.get_portfolio_value(current_price)
        self.history.append({
            'timestamp': timestamp, 'credit': self.credit, 'holdings_shares': self.holdings_shares,
            'price': current_price, 'portfolio_value': portfolio_value,
            'avg_buy_price': self.get_average_buy_price(),
            'total_fees_paid': self.total_fees_paid,
            'realized_pnl_eval': self.realized_gains_this_evaluation # Track running realized PnL
        })
        if portfolio_value > self.max_portfolio_value_achieved:
            self.max_portfolio_value_achieved = portfolio_value
        
        # Check alive can also be done here based on portfolio value too
        if portfolio_value < (self.initial_credit * 0.001) and self.credit < (self.initial_credit * 0.001) : # If portfolio value is negligible
             self.is_alive = False

    def get_portfolio_value(self, current_price):
        if current_price <= 1e-6: return self.credit 
        return self.credit + (self.holdings_shares * current_price)

    def get_state_for_nn(self, current_price, max_possible_credit, max_possible_holdings_value):
        norm_credit = self.credit / max_possible_credit if max_possible_credit > 1e-6 else 0
        
        current_holdings_value = self.holdings_shares * current_price
        norm_holdings_value = current_holdings_value / max_possible_holdings_value if max_possible_holdings_value > 1e-6 else 0
        
        unrealized_pl_percentage = 0.0
        if self.holdings_shares > 1e-7 and current_price > 1e-6: 
            avg_buy = self.get_average_buy_price()
            if avg_buy > 1e-6: 
                unrealized_pl_percentage = (current_price - avg_buy) / avg_buy
        
        norm_unrealized_pl = np.clip(unrealized_pl_percentage, -0.75, 1.5) # Wider range: -75% to +150%
        
        return [ np.clip(norm_credit, 0, 1), np.clip(norm_holdings_value, 0, 1), norm_unrealized_pl ]
