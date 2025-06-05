# trader.py # (No changes from your original, shown for completeness of file structure)
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

        self.total_cost_of_holdings = 0.0 # Tracks cost basis of current shares
        self.history = []
        self.trade_log = []
        self.realized_gains_this_evaluation = 0.0
        self.max_portfolio_value_achieved = initial_credit # This will be the peak in the current eval window
        self.total_fees_paid = 0.0

    def reset(self): # Called if reusing Trader instance, but we create new ones per eval
        self.credit = self.initial_credit
        self.holdings_shares = self.initial_holdings_shares
        self.is_alive = True
        self.history = []
        self.trade_log = []
        self.total_cost_of_holdings = 0.0
        self.realized_gains_this_evaluation = 0.0
        # Max portfolio value for a fresh instance starts at initial capital
        self.max_portfolio_value_achieved = self.get_portfolio_value(0) # Effectively initial_credit if price is 0 for holdings
        if self.holdings_shares == 0: self.max_portfolio_value_achieved = self.initial_credit

        self.total_fees_paid = 0.0

    def _check_alive(self):
        # Consider bankrupt if credit is extremely low AND no significant holdings
        min_meaningful_credit_abs = 0.01 # e.g. 1 cent
        min_meaningful_holdings_value_approx = 0.1 # Approx value, hard to check without price
        
        # If portfolio value (credit + approx holdings) is near zero
        # For simplicity, if credit is negligible and holdings are negligible.
        if self.credit < min_meaningful_credit_abs and self.holdings_shares < 1e-8: # 1e-8 is a common threshold for "zero" shares
            self.is_alive = False
        # Also, if portfolio value is drastically reduced (e.g. below 1% of initial)
        # This is handled by RUIN_THRESHOLD in fitness function primarily.
        return self.is_alive

    def get_average_buy_price(self):
        if self.holdings_shares <= 1e-9: return 0.0 # Avoid division by zero
        return self.total_cost_of_holdings / self.holdings_shares

    def buy(self, cash_amount_to_spend_requested, price_per_share, timestamp, return_trade_profit=False): # return_trade_profit kept for API consistency if used elsewhere
        if not self.is_alive or price_per_share <= 1e-6: return False
        if cash_amount_to_spend_requested <= 1e-6 : return False # Not enough to buy anything meaningful

        # Max spendable considering fee: spend_on_asset + spend_on_asset*fee_rate <= credit
        # spend_on_asset * (1 + fee_rate) <= credit
        # spend_on_asset <= credit / (1 + fee_rate)
        max_spend_on_asset_possible = self.credit / (1 + self.trading_fee_percent)
        actual_cash_to_spend_on_asset = min(cash_amount_to_spend_requested, max_spend_on_asset_possible)

        if actual_cash_to_spend_on_asset < 1e-3 : # e.g. less than 0.1 cent
            return False

        fee_for_this_buy = actual_cash_to_spend_on_asset * self.trading_fee_percent
        total_cost_of_transaction = actual_cash_to_spend_on_asset + fee_for_this_buy

        # Due to float precision, check if total_cost slightly exceeds credit
        if total_cost_of_transaction > self.credit + 1e-7: # Allow tiny overdraft for precision
            # This case should ideally be caught by max_spend_on_asset_possible logic
            # print(f"DEBUG: Buy failed. Cost {total_cost_of_transaction} > Credit {self.credit}")
            return False

        shares_to_buy = actual_cash_to_spend_on_asset / price_per_share
        if shares_to_buy < 1e-8: return False # Buying negligible shares

        self.credit -= total_cost_of_transaction
        self.total_fees_paid += fee_for_this_buy

        # Update cost basis
        self.total_cost_of_holdings += actual_cash_to_spend_on_asset # Cost is asset value, fee is separate
        self.holdings_shares += shares_to_buy

        self.trade_log.append({
            'timestamp': timestamp, 'type': 'buy', 'price': price_per_share,
            'amount_cash': actual_cash_to_spend_on_asset, 'shares': shares_to_buy, 'fee': fee_for_this_buy
        })
        self._check_alive()
        return True


    def sell(self, shares_to_sell_requested, price_per_share, timestamp, return_trade_profit=False):
        if not self.is_alive or price_per_share <= 1e-6: return False

        actual_shares_to_sell = min(shares_to_sell_requested, self.holdings_shares)
        if actual_shares_to_sell <= 1e-8: # Selling negligible shares
            return False

        gross_cash_from_sale = actual_shares_to_sell * price_per_share
        fee_for_this_sell = gross_cash_from_sale * self.trading_fee_percent
        net_cash_gained = gross_cash_from_sale - fee_for_this_sell

        # Calculate profit from this specific sale
        avg_buy_price_of_sold_shares = self.get_average_buy_price()
        cost_of_shares_sold = avg_buy_price_of_sold_shares * actual_shares_to_sell
        
        profit_from_this_sell = net_cash_gained - cost_of_shares_sold # This is realized P&L for these shares
        self.realized_gains_this_evaluation += profit_from_this_sell

        self.credit += net_cash_gained
        self.total_fees_paid += fee_for_this_sell

        # Update cost basis
        if abs(self.holdings_shares - actual_shares_to_sell) < 1e-9 : # Selling all shares
            self.total_cost_of_holdings = 0.0
            self.holdings_shares = 0.0
        else: # Selling partial shares
            # Reduce total_cost_of_holdings proportionally
            proportion_sold = actual_shares_to_sell / self.holdings_shares if self.holdings_shares > 1e-9 else 0
            self.total_cost_of_holdings *= (1 - proportion_sold)
            self.holdings_shares -= actual_shares_to_sell
            if self.total_cost_of_holdings < 0: self.total_cost_of_holdings = 0.0 # Safety for precision

        if self.holdings_shares < 1e-9: self.holdings_shares = 0.0 # Ensure it's cleanly zero


        self.trade_log.append({
            'timestamp': timestamp, 'type': 'sell', 'price': price_per_share,
            'amount_cash': gross_cash_from_sale, # Gross proceeds before fee
            'net_cash_gained': net_cash_gained,  # Net cash to credit after fee
            'shares': actual_shares_to_sell,
            'profit': profit_from_this_sell,     # P&L for this specific trade
            'fee': fee_for_this_sell
        })
        self._check_alive()
        if return_trade_profit: # Kept for API consistency if needed elsewhere
            return profit_from_this_sell
        return True

    def hold(self): # Placeholder, no action
        pass

    def update_history(self, timestamp, current_price):
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

        # Check if agent is effectively bankrupt based on portfolio value
        # This is a more aggressive check than just credit/holdings separately
        if portfolio_value < (self.initial_credit * 0.001): # e.g. < 0.1% of initial capital
             self.is_alive = False


    def get_portfolio_value(self, current_price):
        if current_price <= 1e-6: return self.credit # If price is zero, holdings are worthless
        return self.credit + (self.holdings_shares * current_price)

    def get_state_for_nn(self, current_price, max_possible_credit, max_possible_holdings_value):
        # Normalize credit
        norm_credit = self.credit / max_possible_credit if max_possible_credit > 1e-6 else 0

        # Normalize current value of holdings
        current_holdings_value = self.holdings_shares * current_price
        norm_holdings_value = current_holdings_value / max_possible_holdings_value if max_possible_holdings_value > 1e-6 else 0

        # Normalized unrealized P&L percentage on current holdings
        unrealized_pl_percentage = 0.0
        if self.holdings_shares > 1e-8 and current_price > 1e-6: # If has holdings and price is valid
            avg_buy = self.get_average_buy_price()
            if avg_buy > 1e-6: # Avoid division by zero if avg_buy is zero (e.g. free shares)
                unrealized_pl_percentage = (current_price - avg_buy) / avg_buy
        
        # Clip P&L to a reasonable range (e.g., -75% loss to +150% gain) to prevent extreme values
        norm_unrealized_pl = np.clip(unrealized_pl_percentage, -0.75, 1.5) 

        return [ np.clip(norm_credit, 0, 1), np.clip(norm_holdings_value, 0, 1), norm_unrealized_pl ]
