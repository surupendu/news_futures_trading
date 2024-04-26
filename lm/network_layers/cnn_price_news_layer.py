from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
import torch.nn as nn
from gym import spaces
import torch

"""
    Configuration of feature extraction module
"""

class CNN_Price_News_Layer(BaseFeaturesExtractor):
    def __init__(self, observation_space: spaces.Dict, feature_dim):
        super(CNN_Price_News_Layer, self).__init__(observation_space, feature_dim)
        self.conv_1 = nn.Conv1d(in_channels=768, out_channels=200, kernel_size=3)
        self.conv_2 = nn.Conv1d(in_channels=14, out_channels=14, kernel_size=3)

        self.linear_1 = nn.Linear(200, 100)
        self.linear_2 = nn.Linear(100, 1)
        self.sigmoid = nn.Sigmoid()

        self.linear_3 = nn.Linear(14, 14)
        self.linear_4 = nn.Linear(16, feature_dim)

    def forward(self, observation):
        market_value = observation["market_value"]
        news_embeddings = observation["news_embeddings"]

        # Encode sequence of news embeddings
        news_embeddings = news_embeddings.transpose(1, 2)
        hidden_layer = self.conv_1(news_embeddings)
        hidden_layer = hidden_layer.transpose(1, 2)
        hidden_layer = torch.sum(hidden_layer, dim=1)
        news_context = self.linear_1(hidden_layer)
        news_context = self.linear_2(news_context)
        news_context = self.sigmoid(news_context)

        # Encode sequence of price vectors
        market_value = market_value.transpose(1, 2)
        hidden_layer = self.conv_2(market_value)
        hidden_layer = hidden_layer.transpose(1, 2)
        price_context = torch.sum(hidden_layer, dim=1)
        price_context = self.linear_3(price_context)

        # Combine the price representation and news representation
        context_vector = torch.cat([news_context, price_context], dim=1)
        action = observation["action"]
        obs_vector = torch.cat((context_vector, action), dim=1)
        obs_vector = self.linear_4(obs_vector)
        return obs_vector
