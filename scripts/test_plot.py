import matplotlib.pyplot as plt
from procgenac.utils import make_env, plot_results
import torch

a = torch.rand((10, 32)) * 10
b = torch.rand((10, 32)) * 10
rewards = (a, b)
steps = list(range(10))

fig, ax = plt.subplots(1, 1)
ax = plot_results(ax, steps, rewards)
plt.show()
