import gym
from gym import spaces
import pandas as pd
from evaluate import Evaluate
import warnings
import numpy as np
import os
from transformers import BertTokenizerFast, BertModel
import torch
from torch import cuda

warnings.filterwarnings(action="ignore")

"""
    Custom environment for simulating futures trading
"""

class Env_CNN_News_Price(gym.Env):
    """Custom Environment that follows gym interface"""
    metadata = {'render.modes': ['human']}

    def __init__(
                    self, nifty_df, news_df, save_file, action_dim, observation_dim,
                    margin_pct, max_num_lots, lot_size, num_lots_held, window_size,
                    num_news_articles, language_model
                ):
        super(Env_CNN_News_Price, self).__init__()
        self.nifty_df = self.eod(nifty_df)
        self.news_df = news_df
        self.num_news_articles = num_news_articles
        self.init_params(margin_pct, max_num_lots, lot_size, num_lots_held)
        self.action_dim = action_dim
        self.observation_dim = observation_dim
        self.window_size = window_size
        self.episode = 0

        if language_model == "bert":
            self.tokenizer = BertTokenizerFast.from_pretrained('bert-base-uncased')
            self.bert_model = BertModel.from_pretrained('bert-base-uncased')
        if language_model == "finbert":
            self.tokenizer = BertTokenizerFast.from_pretrained('ProsusAI/finbert')
            self.bert_model = BertModel.from_pretrained('ProsusAI/finbert')

        self.bert_model.to(self.device)

        # Define action and observation space
        self.action_space = spaces.Box(low=-1, high=1, shape=(self.action_dim,))
        self.observation_space = spaces.Dict({
                                    "news_embeddings": spaces.Box(low=-np.inf, high=np.inf,
                                                               shape=(self.num_news_articles, 768),
                                                               dtype=np.float64
                                                            ),
                                    "market_value": spaces.Box(low=-np.inf, high=np.inf,
                                                               shape=(self.window_size, self.observation_dim),
                                                               dtype=np.float64
                                                            ),
                                    "action": spaces.Box(low=-self.max_num_lots, high=self.max_num_lots,
                                                         shape=(1,),
                                                         dtype=np.float64
                                                        )
                                })
        
        self.save_file = save_file
        self.init_csv_file()

    def init_params(self, margin_pct, max_num_lots, lot_size, num_lots_held):
        """
            Initialize the parameters
        """
        self.idx = 0
        self.balance = max_num_lots * lot_size * self.nifty_df.loc[0]["Close_1"]
        self.prev_close_price = self.nifty_df.loc[0]["Close_1"]
        self.initial_balance = self.balance
        self.margin_pct = margin_pct
        self.max_num_lots = max_num_lots
        self.lot_size = lot_size
        self.num_lots_held = num_lots_held
        self.max_num_lots_held = max_num_lots
        self.device = 'cuda' if cuda.is_available() else 'cpu'
        self.embeddings = None

    def init_csv_file(self):
        """
            Initialize the csv file
        """
        self.dates = []
        self.times = []
        self.balances = []
        self.actions = []
        self.lots = []
        self.current_prices = []
        self.df = pd.DataFrame([], columns=["Date", "Time", "Balance", "Actions", "Lots", "Current Price"])

    def eod(self, nifty_df):
        """
            Return day and times which indicate end of day
        """
        nifty_df["Date_1"] = nifty_df["Date"].shift(-1)
        nifty_df["EOD"] = (nifty_df["Date_1"] != nifty_df["Date"])
        nifty_df = nifty_df.drop(columns=["Date_1"])
        return nifty_df        

    def insert_array_element(self, date, time, action, balance, current_price):
        """
            Insert the values in rows
        """
        self.dates.append(date)
        self.times.append(time)
        self.actions.append(action)
        self.balances.append(balance)
        self.lots.append(self.num_lots_held)
        self.current_prices.append(current_price)

    def insert_to_df(self):
        self.df["Date"] = self.dates
        self.df["Time"] = self.times
        self.df["Balance"] = self.balances
        self.df["Actions"] = self.actions
        self.df["Lots"] = self.lots
        self.df["Current Price"] = self.current_prices
        self.df.to_csv(self.save_file, index=False, mode='a', header=not os.path.exists(self.save_file))

    def execute_action(self, action, current_price, prev_price, eod_done, eoc_done):
        """
            Execute the action of the RL agent
        """
        if ((self.num_lots_held+action) > self.max_num_lots_held) or ((self.num_lots_held+action) < -self.max_num_lots_held):
            action = 0
        if eoc_done == True:
            if self.num_lots_held < 0:
                action = abs(self.num_lots_held)
            elif self.num_lots_held > 0:
                action = -self.num_lots_held
            elif self.num_lots_held == 0:
                action = 0

        # Calculate the contract value
        contract_value = action * self.lot_size * current_price
        margin_value = self.margin_pct * contract_value
        new_balance = self.balance - margin_value
        self.num_lots_held += action

        # Calculate mark to market
        if eod_done == True:
            new_balance += self.num_lots_held * self.lot_size * (current_price - self.prev_close_price)
            self.prev_close_price = current_price

        # Calculate the reward
        reward = 0.85 * action * (current_price - prev_price) + 0.15 * (new_balance - self.balance)
        self.balance = new_balance
        return reward, action

    def get_news_embeddings(self, titles):
        """
            Get news embedding of the news titles
        """
        tokens = self.tokenizer(titles, truncation=True, max_length=40, padding='max_length')
        input_ids = torch.LongTensor(tokens["input_ids"])
        attention_mask = torch.FloatTensor(tokens["attention_mask"])
        input_ids = input_ids.to(self.device)
        attention_mask = attention_mask.to(self.device)
        with torch.no_grad():
            output = self.bert_model(input_ids, attention_mask)
            last_hidden_state = output.last_hidden_state
            news_embeddings = torch.sum(last_hidden_state, dim=1)
        news_embeddings = news_embeddings.cpu().numpy()
        
        # Perform padding of the sequence if required
        if news_embeddings.shape[0] < self.num_news_articles:
            diff = self.num_news_articles - news_embeddings.shape[0]
            temp = np.zeros((diff, news_embeddings.shape[1]))
            news_embeddings = np.concatenate((news_embeddings, temp), axis=0)
        return news_embeddings

    def step(self, action):
        """
            Execute the action of the RL agent
        """

        # Convert the continuous value to discrete value
        action = action * self.max_num_lots
        action = action.astype(int)
        action = action[0]

        # Get the previous price 
        prev_price = self.nifty_df.iloc[self.idx + self.window_size - 1]["Close_1"]

        self.idx += 1
        current_price = self.nifty_df.iloc[self.idx + self.window_size]["Close_1"]
        date = self.nifty_df.iloc[self.idx + self.window_size]["Date"]
        time = self.nifty_df.iloc[self.idx + self.window_size]["Time"]

        titles = self.news_df[self.idx + self.window_size]
        
        # Add zero embeddings if no news articles are present
        # Or truncate the sequence if it exceeds the required sequence length
        if len(titles) == 0:
            news_embeddings = np.zeros((self.num_news_articles, 768))
            self.embeddings = news_embeddings
        elif titles == self.news_df[self.idx + self.window_size - 1] and self.embeddings is not None:
            news_embeddings = self.embeddings
        else:
            titles = titles[-self.num_news_articles:]
            news_embeddings = self.get_news_embeddings(titles)
            self.embeddings = news_embeddings

        # Get the end of day indicator and end of contract indicator
        eod_indicator = self.nifty_df.iloc[self.idx + self.window_size]["EOD"]
        eoc_indicator = self.nifty_df.iloc[self.idx + self.window_size]["EOC"]
        
        # Execute the action and get the reward
        reward, action = self.execute_action(action, current_price, prev_price, eod_indicator, eoc_indicator)
        market_values = np.array(self.nifty_df.iloc[self.idx:self.window_size + self.idx, 7:-3].values).astype("float64")
        action = np.array([action])
        
        # Dictionary containing the next state
        next_state = {
                        "news_embeddings": news_embeddings,
                        "market_value": market_values,
                        "action": action
                    }
        self.insert_array_element(date, time, action, self.balance, current_price)

        # Check for end of contract to indicate the end the episode
        if eoc_indicator == True or self.balance < 0 or (self.idx + self.window_size) == len(self.nifty_df)-1:
            self.insert_to_df()
            eoc_indicator = True
            if self.idx + self.window_size == len(self.nifty_df)-1:
                eoc_indicator = True
                self.evaluate_model()

        return next_state, reward, eoc_indicator, {}

    def reset(self):
        """
            Reset the environment if the sequence
        """
        
        # Check for end of trading session and reset the balance and no. of lots held
        if self.idx + self.window_size == len(self.nifty_df)-1:
            self.idx = 0
            self.balance = self.max_num_lots * self.lot_size * self.nifty_df.loc[self.idx]["Close_1"]
            self.prev_close_price = self.nifty_df.loc[self.idx]["Close_1"]
        self.num_lots_held = 0
        self.episode += 1

        titles = self.news_df[self.idx + self.window_size]
        titles = titles[-self.num_news_articles:]

        # Add zero embeddings if no news articles are present
        # Or truncate the sequence if it exceeds the required sequence length
        if len(titles) == 0:
            news_embeddings = np.zeros((self.num_news_articles, 768))
            self.embeddings = news_embeddings
        else:
            titles = titles[-self.num_news_articles:]
            news_embeddings = self.get_news_embeddings(titles)
            self.embeddings = news_embeddings

        # Prepare the next state
        market_values = np.array(self.nifty_df.iloc[self.idx:self.window_size + self.idx, 7:-3].values)
        action = np.array([0])
        next_state = {
                        "news_embeddings": news_embeddings,
                        "market_value": market_values,
                        "action": action
                    }

        self.init_csv_file()
        return next_state

    def render(self, mode="human", close=False):
        return None

    def evaluate_model(self):
        """
            Evaluate the trading actions of the agent
        """
        print("Initial Balance: {:}".format(self.initial_balance))
        print("Final Balance: {:}".format(self.balance))
        trade_df = pd.read_csv(self.save_file)
        evaluate_trade = Evaluate(trade_df, self.initial_balance)
        sharpe_ratio = evaluate_trade.calc_sharpe_ratio()
        sortino_ratio = evaluate_trade.calc_sortino_ratio()
        profit = evaluate_trade.calc_profit()
        max_draw_down, draw_down_duration = evaluate_trade.calc_max_drawdown()
        annualized_return = evaluate_trade.calc_annualized_return()
        volatility = evaluate_trade.calc_annualized_volatility()

        print("Sharpe Ratio: {:}".format(sharpe_ratio))
        print("Sortino Ratio: {:}".format(sortino_ratio))
        print("Total Profit: {:}".format(profit))
        print("Max Draw Down (%): {:}".format(max_draw_down))
        print("Max Draw Down duration (days): {:}".format(draw_down_duration))
        print("Return (%): {:}".format(annualized_return))
        print("Volatility: {:}".format(volatility))
