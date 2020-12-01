from procgenac.modelling.training import training_pipeline
from procgenac.modelling.utils import Hyperparameters

path_to_base = "../"
hyperparams = Hyperparameters(model_type="PPO")

# Modify hyperparams
hyperparams.total_steps = 10_000
hyperparams.test_run = 1

training_pipeline(hyperparams, path_to_base, verbose=True)
