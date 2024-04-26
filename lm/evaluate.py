from math import sqrt

"""
    Evaluate the trading models
"""

class Evaluate:
    def __init__(self, trade_df, balance):
        self.trade_df = self.eod(trade_df)
        self.initial_balance = balance

    def eod(self, trade_df):
        trade_df["Date_1"] = trade_df["Date"].shift(-1)
        trade_df["EOD"] = (trade_df["Date_1"] != trade_df["Date"])
        trade_df.drop(columns=["Date_1"], inplace=True)
        return trade_df

    def day_wise_returns(self, trade_df):
        trade_df = trade_df[trade_df["EOD"] == True]
        trade_df["Balance_1"] = trade_df["Balance"].shift(1)
        trade_df["Balance_1"] = trade_df["Balance_1"].fillna(value=self.initial_balance)
        trade_df["Return"] = (trade_df["Balance"] - trade_df["Balance_1"])/trade_df["Balance_1"]
        returns = trade_df["Return"]
        return returns

    def calc_sharpe_ratio(self, risk_free_rate=None, annualized_coefficient=None):
        '''
            Calculate the Sharpe Ratio
        '''
        returns = self.day_wise_returns(self.trade_df)
        sharpe_ratio = returns.mean()/returns.std()
        return sharpe_ratio

    def calc_sortino_ratio(self, risk_free_rate=None, annualized_coefficient=None):
        '''
            Calculate the Sortino Ratio
        '''
        returns = self.day_wise_returns(self.trade_df)
        neg_returns = returns[returns<0]
        sortino_ratio = returns.mean()/neg_returns.std()
        return sortino_ratio

    def calc_profit(self):
        '''
            Calculate the Total Profit
        '''
        profit  = self.trade_df.iloc[-1]["Balance"] - self.initial_balance
        return profit

    def calc_transactions(self):
        self.trade_df["Transactions_1"] = self.trade_df["Transactions"].shift(1)
        self.trade_df.dropna(inplace=True)
        self.trade_df = self.trade_df[self.trade_df["Transactions_1"] != self.trade_df["Transactions"]]
        self.trade_df = self.trade_df[self.trade_df["Transactions"] != "Hold"]
        num_transactions = len(self.trade_df)
        return num_transactions

    def calc_max_drawdown(self):
        '''
            Calculate the Maximum Drawdown
        '''
        returns = self.day_wise_returns(self.trade_df)
        cumulative_returns = (returns + 1).cumprod()
        peak_returns = cumulative_returns.expanding(min_periods=1).max()
        draw_downs = (cumulative_returns/peak_returns) - 1
        max_draw_down = draw_downs.min() * 100
        draw_down_duration = draw_downs.argmin()
        return max_draw_down, draw_down_duration
    
    def calc_annualized_return(self):
        '''
            Calculate the Return (%)
        '''
        final_balance = self.trade_df.iloc[-1]["Balance"]
        annualized_return = ((final_balance - self.initial_balance)/self.initial_balance) * 100
        return annualized_return
    
    def calc_annualized_volatility(self):
        '''
            Calculate the Volatility
        '''
        returns = self.day_wise_returns(self.trade_df)
        volatility = returns.std() * sqrt(len(returns))
        return volatility
