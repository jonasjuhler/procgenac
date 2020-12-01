from procgenac.modelling.utils import training_pipeline


class PPOParams:
    def __init__(self):
        self.model_type = "PPO"
        self.env_name = "starpilot"
        self.num_envs = "32"
        self.num_levels = "200"
        self.feature_dim = "128"
        self.value_coef = "0.5"
        self.entropy_coef = "0.01"
        self.eps = "0.2"
        self.grad_eps = "0.5"
        self.num_epochs = "3"
        self.batch_size = "1024"
        self.adam_lr = "5e-4"
        self.adam_eps = "1e-5"
        self.num_steps = "256"
        self.total_steps = "10000"
        self.get_test = "1"


path_to_base = "../"
param_args = PPOParams()

training_pipeline(param_args, path_to_base, verbose=True, prod=False)
