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
        self.max_portfolio_value_achieved = initial_credit
        self.total_fees_paid = 0.0 # Track fees

    def reset(self):
        self.credit = self.initial_credit
        self.holdings_shares = self.initial_holdings_shares
        self.is_alive = True
        self.history = []
        self.trade_log = []
        self.total_cost_of_holdings = 0.0
        self.realized_gains_this_evaluation = 0.0
        self.max_portfolio_value_achieved = self.initial_credit
        self.total_fees_paid = 0.0


    def _check_alive(self):
        # Needs enough credit to cover potential fees for a minimal trade, or has holdings.
        # Simplified:
        self.is_alive = self.credit > (self.initial_credit * 0.01) or self.holdings_shares > 1e-6 # e.g. must have at least 1% of initial credit
        if self.credit <= 1e-6 and self.holdings_shares <= 1e-6:
            self.is_alive = False
        return self.is_alive

    def get_average_buy_price(self):
        if self.holdings_shares <= 1e-9: return 0.0
        return self.total_cost_of_holdings / self.holdings_shares

    def buy(self, cash_amount_to_spend, price_per_share, timestamp):
        if not self.is_alive: return False
        
        # Calculate fee for this potential buy
        fee_for_this_buy = cash_amount_to_spend * self.trading_fee_percent
        
        # Ensure trader has enough credit for the spend amount AND the fee
        if self.credit < (cash_amount_to_spend + fee_for_this_buy):
            # If not enough for requested amount + fee, try to buy with available credit minus fee
            available_for_spend_before_fee = self.credit / (1 + self.trading_fee_percent)
            if available_for_spend_before_fee < 1e-3 : # If available amount is negligible
                 return False 
            cash_amount_to_spend = available_for_spend_before_fee # Adjust spend amount
            fee_for_this_buy = cash_amount_to_spend * self.trading_fee_percent


        actual_cash_to_spend_on_asset = cash_amount_to_spend # This is what buys shares
        total_cost_of_transaction = actual_cash_to_spend_on_asset + fee_for_this_buy

        if total_cost_of_transaction > self.credit + 1e-6 : # Final check, 1e-6 for float precision
            # This should ideally be caught by the logic above
            # print(f"DEBUG BUY: Not enough credit. Need {total_cost_of_transaction}, have {self.credit}")
            return False


        if actual_cash_to_spend_on_asset <= 1e-6 or price_per_share <= 1e-6: return False

        shares_to_buy = actual_cash_to_spend_on_asset / price_per_share
        if shares_to_buy < 1e-6: return False # Avoid buying dust

        self.credit -= total_cost_of_transaction # Deduct asset cost + fee
        self.total_fees_paid += fee_for_this_buy
        
        # Cost basis includes the price of shares, fees are an expense
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
        if actual_shares_to_sell <= 1e-6 or price_per_share <= 1e-6: return False
            
        gross_cash_from_sale = actual_shares_to_sell * price_per_share
        fee_for_this_sell = gross_cash_from_sale * self.trading_fee_percent
        net_cash_gained = gross_cash_from_sale - fee_for_this_sell
        
        avg_buy_price_of_sold_shares = self.get_average_buy_price() 
        cost_of_shares_sold = avg_buy_price_of_sold_shares * actual_shares_to_sell
        
        # Profit is net cash gained minus cost of those shares
        profit_from_this_sell = net_cash_gained - cost_of_shares_sold 
        self.realized_gains_this_evaluation += profit_from_this_sell # Tracks net profit/loss

        self.credit += net_cash_gained # Add net cash after fee
        self.total_fees_paid += fee_for_this_sell

        if abs(self.holdings_shares - actual_shares_to_sell) < 1e-9 : 
            self.total_cost_of_holdings = 0.0
        else:
            self.total_cost_of_holdings -= cost_of_shares_sold 
            if self.total_cost_of_holdings < 0: self.total_cost_of_holdings = 0.0
        self.holdings_shares -= actual_shares_to_sell
        if self.holdings_shares < 1e-9: self.holdings_shares = 0.0

        self.trade_log.append({
            'timestamp': timestamp, 'type': 'sell', 'price': price_per_share,
            'amount_cash': gross_cash_from_sale, # Gross amount before fee
            'net_cash_gained': net_cash_gained, # Net cash after fee
            'shares': actual_shares_to_sell,
            'profit': profit_from_this_sell, # Net profit of this trade
            'fee': fee_for_this_sell
        })
        self._check_alive()
        return True

    def hold(self): pass

    def update_history(self, timestamp, current_price):
        portfolio_value = self.get_portfolio_value(current_price) 
        self.history.append({
            'timestamp': timestamp, 'credit': self.credit, 'holdings_shares': self.holdings_shares,
            'price': current_price, 'portfolio_value': portfolio_value,
            'avg_buy_price': self.get_average_buy_price(),
            'total_fees_paid': self.total_fees_paid
        })
        if portfolio_value > self.max_portfolio_value_achieved: self.max_portfolio_value_achieved = portfolio_value
        if portfolio_value < 1e-6 and self.credit < 1e-6 : self.is_alive = False

    def get_portfolio_value(self, current_price):
        if current_price <= 1e-6: return self.credit 
        return self.credit + (self.holdings_shares * current_price)

    def get_state_for_nn(self, current_price, max_possible_credit, max_possible_holdings_value):
        norm_credit = self.credit / max_possible_credit if max_possible_credit > 0 else 0
        current_holdings_value = self.holdings_shares * current_price
        norm_holdings_value = current_holdings_value / max_possible_holdings_value if max_possible_holdings_value > 0 else 0
        unrealized_pl_percentage = 0.0
        if self.holdings_shares > 1e-6 and current_price > 1e-6:
            avg_buy = self.get_average_buy_price()
            if avg_buy > 1e-6: unrealized_pl_percentage = (current_price - avg_buy) / avg_buy 
        norm_unrealized_pl = np.clip(unrealized_pl_percentage, -0.5, 1.0) # Clip from -50% to +100%
        return [ np.clip(norm_credit, 0, 1), np.clip(norm_holdings_value, 0, 1), norm_unrealized_pl ]
