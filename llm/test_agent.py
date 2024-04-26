from trading_env.trading_env import Env_CNN_News_Price
from agent.ppo_agent import PPO_Agent
import warnings
import pandas as pd
import ast
import json
import argparse

warnings.filterwarnings(action="ignore")

"""
    Code to test the PPO-based RL agent
    use python test_agent.py [-h] to view the options available
"""

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Process some integers.')
    parser.add_argument('--text_representation', type=str,
                        help='text representation scheme',
                        choices=["llama2", "mistral"])
    args = parser.parse_args()

    # Select config file based on text representation scheme
    if args.text_representation == "llama2":
        config = json.load(open("configs/llama2.json", "r"))
        approach = "llama2"
        save_model_path = "trained_models/"
        model_file_name = "{:}_{:}.zip".format(config["model_name"], config["language_model"])
    elif args.text_representation == "mistral":
        config = json.load(open("configs/mistral.json", "r"))
        approach = "mistral"
        model_file_name = "{:}_{:}.zip".format(config["model_name"], config["language_model"])

    # Read the test data
    test_df = pd.read_csv(config["path"] + "test_df_news.csv")
    test_years = ["2017", "2018", "2019", "2020", "2021"]

    for test_year in test_years:
        save_file = config["save_file_path"] + "trade_data_{:}.csv".format(test_year)
        max_num_lots = 3
        if test_year in ["2018", "2019", "2020", "2021"]:
            lot_size = 75
        else:
            lot_size = 25
        test_df_1 = test_df[test_df["Date"].str.contains(test_year)]
        test_df_1.reset_index(drop=True, inplace=True)
        news_df = test_df_1["Titles"]
        news_df = news_df.apply(lambda x: ast.literal_eval(x))
        test_df_1 = test_df_1[test_df_1.columns[:-1]]

        # Initialize the parameters of the environment
        env_parameters = {
                            "nifty_df": test_df_1,
                            "news_df": news_df,
                            "save_file": save_file,
                            "action_dim": config["action_dim"],
                            "max_num_lots": config["max_num_lots"],
                            "lot_size": config["lot_size"],
                            "num_lots_held": config["num_lots_held"],
                            "margin_pct": config["margin_pct"],
                            "window_size": config["window_size"],
                            "num_news_articles": config["num_news_articles"],
                            "language_model": config["language_model"]
                        }

        # Intialize the environment
        observation_dim = len(test_df_1.columns[7:-2])
        env = Env_CNN_News_Price(**env_parameters)
        
        model_parameters = {
                            "policy": config["policy"],
                            "env": env,
                            "learning_rate": config["learning_rate"],
                            "n_steps": config["n_steps"],
                            "batch_size": config["batch_size"],
                            "ent_coef": config["ent_coef"],
                            "n_epochs": config["n_epochs"],
                            "seed": 42,
                            "verbose": 1,
                            "approach": approach
                        }

        # Load the PPO agent
        rl_agent = PPO_Agent(config["model_name"], **model_parameters)
        rl_agent.load_model(save_model_path, model_file_name)

        # Test the PPO agent
        print("--------------Year: {:}--------------".format(test_year))
        rl_agent.test_model(env)
        print("-------------------------------------")
