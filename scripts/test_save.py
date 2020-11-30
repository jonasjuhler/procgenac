from procgenac.utils import save_rewards
import torch

model_name = "PPO"
env_name = "starpilot"

a = torch.rand((10, 32)) * 10
b = torch.rand((10, 32)) * 10
rewards = (a, b)
steps = list(range(10))

filename = f"{model_name}_{env_name}.csv"
save_rewards(steps, rewards, filename)
