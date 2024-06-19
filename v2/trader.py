import numpy as np

# class OrderBook(pandas.DataFrame):
#     def __init__(self):
#         columns = ["OrderID", "OrderType", "Ticker", "Quantity", "Holdings"]
#         super(OrderBook, self).__init__()

class Trader:
    def __init__(self, credit=100, holdings=0, current_state=np.zeros(5)):
        self.current_state = current_state
        self.is_alive = True
        self.goal = False
        self.time_spent = 0

        self.credit = credit
        self.past_credit = credit
        self.holdings = holdings 

        self.credit_history = []
        self.holdings_history = []

    def check_alive(self):
        self.is_alive = self.credit > 0
        return self.is_alive

    def buy(self, quantity, price):
        self.past_credit = self.credit
        if self.credit >= quantity:
            self.credit -= quantity
            self.holdings += quantity / price
        self.check_alive()
        self.credit_history.append(self.credit)
        self.holdings_history.append(self.holdings)

    def sell(self, quantity, price):
        self.past_credit = self.credit
        if self.holdings >= quantity / price:
            self.credit += quantity
            self.holdings -= quantity / price
        self.check_alive()
        self.credit_history.append(self.credit)
        self.holdings_history.append(self.holdings)

    def update_state(self, new_state):
        self.current_state = new_state
