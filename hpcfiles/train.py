import argparse
from procgenac.modelling.training import training_pipeline

parser = argparse.ArgumentParser()
parser.add_argument("--model_type", type=str, default="PPO")
parser.add_argument("--env_name", type=str, default="starpilot")
parser.add_argument("--num_envs", type=int, default=64)
parser.add_argument("--num_levels", type=int, default=200)
parser.add_argument("--feature_dim", type=int, default=128)
parser.add_argument("--value_coef", type=float, default=0.5)
parser.add_argument("--entropy_coef", type=float, default=0.01)
parser.add_argument("--eps", type=float, default=0.2)
parser.add_argument("--grad_eps", type=float, default=0.5)
parser.add_argument("--num_epochs", type=int, default=3)
parser.add_argument("--batch_size", type=int, default=1024)
parser.add_argument("--adam_lr", type=float, default=5e-4)
parser.add_argument("--adam_eps", type=float, default=1e-5)
parser.add_argument("--num_steps", type=int, default=256)
parser.add_argument("--total_steps", type=int, default=10_000_000)
parser.add_argument("--no_test_err", dest="get_test", action="store_false")
parser.set_defaults(get_test=True)
parser.add_argument("--test_run", dest="test_run", action="store_true")
parser.set_defaults(test_run=False)
parser.add_argument("--cnn_type", type=str, default="nature")
parser.add_argument("--model_id", type=str, default="1")

hyperparams = parser.parse_args()

training_pipeline(hyperparams, path_to_base="../", verbose=True)
