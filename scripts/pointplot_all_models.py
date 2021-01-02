import numpy as np
import os
import matplotlib
import matplotlib.pyplot as plt
import pandas as pd

# from procgenac.utils import plot_results, get_formatter

matplotlib.rcParams["text.usetex"] = True
SMALL_SIZE = 14
MEDIUM_SIZE = 16
BIGGER_SIZE = 18
plt.rc("font", size=SMALL_SIZE)  # controls default text sizes
plt.rc("axes", titlesize=SMALL_SIZE)  # fontsize of the axes title
plt.rc("axes", labelsize=MEDIUM_SIZE)  # fontsize of the x and y labels
plt.rc("xtick", labelsize=SMALL_SIZE)  # fontsize of the tick labels
plt.rc("ytick", labelsize=SMALL_SIZE)  # fontsize of the tick labels
plt.rc("legend", fontsize=SMALL_SIZE)  # legend fontsize
plt.rc("figure", titlesize=BIGGER_SIZE)  # fontsize of the figure title

rewards_dir = "../results/rewards"
a2c_train_rews = []
a2c_test_rews = []
ppo_train_rews = []
ppo_test_rews = []

for file in os.scandir(rewards_dir):
    df = pd.read_csv(file)
    if df.steps.values[-1] < 10_000_000:
        idx = df.test_rewards_mean.argmax()
        if "A2C" in file.name:
            a2c_train_rews.append(df.train_rewards_mean.values[idx])
            a2c_test_rews.append(df.test_rewards_mean.values[idx])
        else:
            ppo_train_rews.append(df.train_rewards_mean.values[idx])
            ppo_test_rews.append(df.test_rewards_mean.values[idx])

fig, ax = plt.subplots(1, 1, figsize=(7, 4))
plt.scatter(a2c_train_rews, a2c_test_rews, label="A2C", c="darkslategrey")
plt.scatter(ppo_train_rews, ppo_test_rews, label="PPO", c="darksalmon")
ax.set_xlabel("Train reward")
ax.set_ylabel("Test reward")
plt.legend()
plt.tight_layout()
figpath = "../results/figures/pointplot_stage2.pdf"
plt.savefig(figpath)
