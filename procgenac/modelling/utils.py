import os
import torch
from procgenac.modelling.encoder import Nature, Impala
from procgenac.modelling.ppo import PPO
from procgenac.modelling.a2c import A2C


def init_model(hyperparams, device, env):
    encoder_dict = {
        "nature": Nature(in_channels=3, feature_dim=hyperparams.feature_dim),
        "impala": Impala(in_channels=3, feature_dim=hyperparams.feature_dim),
    }
    if hyperparams.model_type == "A2C":
        model = A2C(
            encoder=encoder_dict[hyperparams.cnn_type],
            feature_dim=hyperparams.feature_dim,
            num_actions=env.action_space.n,
            c1=hyperparams.value_coef,
            c2=hyperparams.entropy_coef,
            grad_eps=hyperparams.grad_eps,
            device=device,
        )
    elif hyperparams.model_type == "PPO":
        model = PPO(
            encoder=encoder_dict[hyperparams.cnn_type],
            feature_dim=hyperparams.feature_dim,
            num_actions=env.action_space.n,
            c1=hyperparams.value_coef,
            c2=hyperparams.entropy_coef,
            eps=hyperparams.eps,
            grad_eps=hyperparams.grad_eps,
            device=device,
        )
    return model


def save_model(model, filepath):
    if not os.path.isdir(os.path.dirname(filepath)):
        os.makedirs(os.path.dirname(filepath))
    torch.save(model.state_dict, filepath)


class Hyperparameters:
    def __init__(self, model_type):
        assert model_type in ["A2C", "PPO"], "Model type has to be A2C or PPO"
        self.model_type = model_type
        self.cnn_type = "impala"
        self.env_name = "starpilot"
        self.num_envs = 32
        self.num_levels = 200
        self.feature_dim = 128
        self.value_coef = 0.5
        self.entropy_coef = 0.01
        self.eps = 0.2
        self.grad_eps = 0.5
        self.num_epochs = 3
        self.batch_size = 1024
        self.adam_lr = 2e-4 if model_type == "A2C" else 5e-4
        self.adam_eps = 1e-4 if model_type == "A2C" else 1e-5
        self.num_steps = 256
        self.total_steps = 10000
        self.get_test = True
        self.test_run = True
        self.model_id = 0
