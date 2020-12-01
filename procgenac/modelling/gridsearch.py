import argparse
from procgenac.modelling.utils import training_pipeline

parser = argparse.ArgumentParser()
parser.add_argument("--model_type", default="PPO")
parser.add_argument("--env_name", default="starpilot")
parser.add_argument("--num_envs", default="32")
parser.add_argument("--num_levels", default="200")
parser.add_argument("--feature_dim", default="128")
parser.add_argument("--value_coef", default="0.5")
parser.add_argument("--entropy_coef", default="0.01")
parser.add_argument("--eps", default="0.2")
parser.add_argument("--grad_eps", default="0.5")
parser.add_argument("--num_epochs", default="3")
parser.add_argument("--batch_size", default="1024")
parser.add_argument("--adam_lr", default="5e-4")  # 2e-4 for A2C
parser.add_argument("--adam_eps", default="1e-5")  # 1e-4 for A2C
parser.add_argument("--num_steps", default="256")
parser.add_argument("--total_steps", default="2_000_000")
parser.add_argument("--get_test", default="0")

param_args = parser.parse_args()

training_pipeline(param_args, path_to_base="../../", verbose=True)
