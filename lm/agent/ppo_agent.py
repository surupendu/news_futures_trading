from stable_baselines3.ppo import PPO
import tqdm as tq
import torch.nn as nn
from network_layers.cnn_price_news_layer import CNN_Price_News_Layer

"""
    PPO agent to be used for trading
"""

class PPO_Agent:
    def __init__(
                    self, model_name, policy, env, approach,
                    learning_rate=0.0003, n_steps=2048, batch_size=64,
                    n_epochs=10, gamma=0.99, gae_lambda=0.95,
                    activation_fn=nn.Tanh(),
                    clip_range=0.2, clip_range_vf=None,
                    normalize_advantage=True, ortho_init=False,
                    ent_coef=0.0, vf_coef=0.5,
                    max_grad_norm=0.5, use_sde=False,
                    sde_sample_freq=-1, target_kl=None, tensorboard_log=None,
                    policy_kwargs=None, verbose=1, seed=42,
                    device='auto', _init_setup_model=True
                ):
        super(PPO_Agent, self).__init__()
        
        self.model_name = model_name
        self.policy = policy
        self.env = env
        self.learning_rate = learning_rate
        self.n_steps = n_steps
        self.batch_size = batch_size
        self.n_epochs = n_epochs
        self.gamma = gamma
        self.gae_lambda = gae_lambda
        self.clip_range = clip_range
        self.clip_range_vf = clip_range_vf
        self.normalize_advantage = normalize_advantage
        self.ent_coef = ent_coef
        self.vf_coef = vf_coef
        self.max_grad_norm = max_grad_norm
        self.use_sde = use_sde
        self.sde_sample_freq = sde_sample_freq
        self.target_kl = target_kl
        self.tensorboard_log = tensorboard_log
        self.policy_kwargs = policy_kwargs
        self.verbose = verbose
        self.seed = seed
        self.device = device
        self._init_setup_model = _init_setup_model
        self.approach = approach

        # Define the configuration of the feature extraction module
        if self.policy == "MultiInputPolicy":
            if approach == "bert":
                features_extractor_class = CNN_Price_News_Layer
                net_arch = [16, 16]
                feature_dim = 16
            elif approach == "finbert":
                features_extractor_class = CNN_Price_News_Layer
                net_arch = [64, 16]
                feature_dim = 64
            
            self.policy_kwargs = dict(
                                        features_extractor_class=features_extractor_class,
                                        net_arch=net_arch,
                                        features_extractor_kwargs=dict(feature_dim=feature_dim)
                                    )

        # Initialize the PPO model
        self.model = PPO(
                            policy=self.policy, env=self.env,
                            learning_rate=self.learning_rate,
                            n_steps=self.n_steps, batch_size=self.batch_size,
                            n_epochs=self.n_epochs, gamma=self.gamma,
                            gae_lambda=self.gae_lambda,
                            clip_range=self.clip_range,
                            clip_range_vf=self.clip_range_vf,
                            normalize_advantage=self.normalize_advantage,
                            ent_coef=self.ent_coef, vf_coef=self.vf_coef,
                            max_grad_norm=self.max_grad_norm,
                            use_sde=self.use_sde,
                            sde_sample_freq=self.sde_sample_freq,
                            target_kl=self.target_kl,
                            tensorboard_log=self.tensorboard_log,
                            policy_kwargs=self.policy_kwargs,
                            verbose=self.verbose, seed=self.seed,
                            device=self.device,
                            _init_setup_model=self._init_setup_model
                        )

    def train_model(self, total_timesteps):
        """
            Train the model
        """
        self.model.learn(total_timesteps, progress_bar=True)

    def test_model(self, env):
        """
            Test the model
        """
        state = env.reset()
        for i in tq.tqdm(range(len(env.nifty_df))):
            action, _ = self.model.predict(state, deterministic=True)
            next_state, reward, eoc_indicator, _ = env.step(action)
            state = next_state
            if eoc_indicator:
                state = env.reset()

    def save_model(self, path, file_name):
        """
            Save model to directory
        """
        self.model.save(path + file_name)

    def load_model(self, path, file_name):
        """
            Load model from directory
        """
        self.model = self.model.load(path + file_name)
