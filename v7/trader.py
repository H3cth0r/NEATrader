# trader.py
import numpy as np
import pandas as pd

class Trader:
    def __init__(self, initial_credit=1000, initial_holdings=0, trading_fee_percent=0.001): # e.g., 0.1% fee
        self.initial_credit = initial_credit
        self.initial_holdings_shares = initial_holdings
        self.trading_fee_percent = trading_fee_percent # Store fee rate

        self.credit = initial_credit
        self.holdings_shares = self.initial_holdings_shares
        self.is_alive = True

        self.total_cost_of_holdings = 0.0
        self.history = []
        self.trade_log = []
        self.realized_gains_this_evaluation = 0.0
        self.max_portfolio_value_achieved = initial_credit # Initialize with starting portfolio
        self.total_fees_paid = 0.0 # Track fees

    def reset(self):
        self.credit = self.initial_credit
        self.holdings_shares = self.initial_holdings_shares
        self.is_alive = True
        self.history = []
        self.trade_log = []
        self.total_cost_of_holdings = 0.0
        self.realized_gains_this_evaluation = 0.0
        # Reset max portfolio to current initial portfolio value upon reset
        current_price_at_reset_effectively_zero_for_holdings = 0 # Or pass a price if holdings have value at reset
        self.max_portfolio_value_achieved = self.credit + (self.holdings_shares * current_price_at_reset_effectively_zero_for_holdings)
        self.total_fees_paid = 0.0


    def _check_alive(self):
        # Agent is alive if it has some credit or some holdings.
        # A very small amount of credit might not be enough to cover fees for any trade.
        # Let's say "alive" means having at least 0.1% of initial capital or any shares.
        min_meaningful_credit = self.initial_credit * 0.001
        self.is_alive = self.credit > min_meaningful_credit or self.holdings_shares > 1e-7 # 1e-7 for float precision
        
        # If truly nothing left.
        if self.credit <= 1e-6 and self.holdings_shares <= 1e-7:
            self.is_alive = False
        return self.is_alive

    def get_average_buy_price(self):
        if self.holdings_shares <= 1e-9: return 0.0
        return self.total_cost_of_holdings / self.holdings_shares

    def buy(self, cash_amount_to_spend_requested, price_per_share, timestamp):
        if not self.is_alive: return False
        if cash_amount_to_spend_requested <= 1e-6 : return False # Trying to spend nothing
        
        # Calculate fee for this potential buy based on requested spend
        # The actual spend might be less if credit is insufficient
        
        # Max possible spend on asset before fee, considering available credit
        # MaxAssetSpend = Credit / (1 + FeeRate)
        max_spend_on_asset_possible = self.credit / (1 + self.trading_fee_percent)

        # Actual amount to spend on asset cannot exceed available credit (adjusted for fee)
        # and cannot exceed requested amount
        actual_cash_to_spend_on_asset = min(cash_amount_to_spend_requested, max_spend_on_asset_possible)
        
        if actual_cash_to_spend_on_asset < 1e-3 : # If effectively spending dust, don't trade
            return False

        fee_for_this_buy = actual_cash_to_spend_on_asset * self.trading_fee_percent
        total_cost_of_transaction = actual_cash_to_spend_on_asset + fee_for_this_buy

        # Final check, though above logic should prevent this. Added for robustness.
        if total_cost_of_transaction > self.credit + 1e-6: # 1e-6 for float precision
            # print(f"DEBUG BUY: Not enough credit. Need {total_cost_of_transaction}, have {self.credit}, trying to spend {cash_amount_to_spend_requested}")
            return False

        if price_per_share <= 1e-6: return False # Avoid division by zero or invalid price

        shares_to_buy = actual_cash_to_spend_on_asset / price_per_share
        if shares_to_buy < 1e-7: return False # Avoid buying negligible shares (dust)

        self.credit -= total_cost_of_transaction
        self.total_fees_paid += fee_for_this_buy
        
        self.total_cost_of_holdings += actual_cash_to_spend_on_asset
        self.holdings_shares += shares_to_buy
        
        self.trade_log.append({
            'timestamp': timestamp, 'type': 'buy', 'price': price_per_share,
            'amount_cash': actual_cash_to_spend_on_asset, 'shares': shares_to_buy, 'fee': fee_for_this_buy
        })
        self._check_alive()
        return True

    def sell(self, shares_to_sell_requested, price_per_share, timestamp):
        if not self.is_alive: return False

        actual_shares_to_sell = min(shares_to_sell_requested, self.holdings_shares)
        if actual_shares_to_sell <= 1e-7 or price_per_share <= 1e-6: return False # Selling dust or invalid price
            
        gross_cash_from_sale = actual_shares_to_sell * price_per_share
        fee_for_this_sell = gross_cash_from_sale * self.trading_fee_percent
        net_cash_gained = gross_cash_from_sale - fee_for_this_sell
        
        avg_buy_price_of_sold_shares = self.get_average_buy_price()
        cost_of_shares_sold = avg_buy_price_of_sold_shares * actual_shares_to_sell
        
        profit_from_this_sell = net_cash_gained - cost_of_shares_sold
        self.realized_gains_this_evaluation += profit_from_this_sell

        self.credit += net_cash_gained
        self.total_fees_paid += fee_for_this_sell

        # Update cost basis
        if abs(self.holdings_shares - actual_shares_to_sell) < 1e-9 : # Selling all shares
            self.total_cost_of_holdings = 0.0
            self.holdings_shares = 0.0 # Explicitly set to 0
        else:
            self.total_cost_of_holdings -= cost_of_shares_sold
            self.holdings_shares -= actual_shares_to_sell
            # Ensure cost basis doesn't go negative due to float issues if selling almost all
            if self.total_cost_of_holdings < 0: self.total_cost_of_holdings = 0.0
        
        if self.holdings_shares < 1e-9: self.holdings_shares = 0.0 # Clean up dust shares


        self.trade_log.append({
            'timestamp': timestamp, 'type': 'sell', 'price': price_per_share,
            'amount_cash': gross_cash_from_sale,
            'net_cash_gained': net_cash_gained,
            'shares': actual_shares_to_sell,
            'profit': profit_from_this_sell,
            'fee': fee_for_this_sell
        })
        self._check_alive()
        return True

    def hold(self): pass # Explicit hold action if needed, currently passive

    def update_history(self, timestamp, current_price):
        portfolio_value = self.get_portfolio_value(current_price)
        self.history.append({
            'timestamp': timestamp, 'credit': self.credit, 'holdings_shares': self.holdings_shares,
            'price': current_price, 'portfolio_value': portfolio_value,
            'avg_buy_price': self.get_average_buy_price(),
            'total_fees_paid': self.total_fees_paid
        })
        if portfolio_value > self.max_portfolio_value_achieved:
            self.max_portfolio_value_achieved = portfolio_value
        
        # Check alive can also be done here based on portfolio value too
        if portfolio_value < (self.initial_credit * 0.001) and self.credit < (self.initial_credit * 0.001) : # If portfolio value is negligible
             self.is_alive = False # Could trigger is_alive check

    def get_portfolio_value(self, current_price):
        if current_price <= 1e-6: return self.credit # If price is invalid, portfolio is just credit
        return self.credit + (self.holdings_shares * current_price)

    def get_state_for_nn(self, current_price, max_possible_credit, max_possible_holdings_value):
        # Ensure denominators are not zero to prevent division by zero errors
        norm_credit = self.credit / max_possible_credit if max_possible_credit > 1e-6 else 0
        
        current_holdings_value = self.holdings_shares * current_price
        norm_holdings_value = current_holdings_value / max_possible_holdings_value if max_possible_holdings_value > 1e-6 else 0
        
        unrealized_pl_percentage = 0.0
        if self.holdings_shares > 1e-7 and current_price > 1e-6: # Min shares to consider having a position
            avg_buy = self.get_average_buy_price()
            if avg_buy > 1e-6: # Avoid division by zero if avg_buy is zero (e.g., no buys yet or error)
                unrealized_pl_percentage = (current_price - avg_buy) / avg_buy
        
        # Clip unrealized P/L. Range of -0.5 to 1.0 means -50% loss to +100% gain.
        # Consider if this range is appropriate or should be wider/different.
        norm_unrealized_pl = np.clip(unrealized_pl_percentage, -0.5, 1.0)
        
        return [ np.clip(norm_credit, 0, 1), np.clip(norm_holdings_value, 0, 1), norm_unrealized_pl ]
