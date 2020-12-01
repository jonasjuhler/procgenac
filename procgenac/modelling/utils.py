import os
import torch
from procgenac.modelling.encoder import Encoder
from procgenac.modelling.ppo import PPO
from procgenac.modelling.a2c import A2C


def init_model(hyperparams, device, env):
    if hyperparams.model_type == "A2C":
        model = A2C(
            encoder=Encoder(in_channels=3, feature_dim=hyperparams.feature_dim),
            feature_dim=hyperparams.feature_dim,
            num_actions=env.action_space.n,
            c1=hyperparams.value_coef,
            c2=hyperparams.entropy_coef,
            grad_eps=hyperparams.grad_eps,
            device=device,
        )
    elif hyperparams.model_type == "PPO":
        model = PPO(
            encoder=Encoder(in_channels=3, feature_dim=hyperparams.feature_dim),
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
    def __init__(self, argparser=None, model_type=None):
        if argparser:
            self.model_type = argparser.model_type
            self.env_name = argparser.env_name
            self.num_envs = int(argparser.num_envs)
            self.num_levels = int(argparser.num_levels)
            self.feature_dim = int(argparser.feature_dim)
            self.value_coef = float(argparser.value_coef)
            self.entropy_coef = float(argparser.entropy_coef)
            self.eps = float(argparser.eps)
            self.grad_eps = float(argparser.grad_eps)
            self.num_epochs = int(argparser.num_epochs)
            self.batch_size = int(argparser.batch_size)
            self.adam_lr = float(argparser.adam_lr)
            self.adam_eps = float(argparser.adam_eps)
            self.num_steps = int(argparser.num_steps)
            self.total_steps = int(argparser.total_steps)
            self.get_test = bool(argparser.get_test)
            self.test_run = bool(argparser.test_run)
        else:
            self.model_type = model_type
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
            self.get_test = 1
            self.test_run = 1
