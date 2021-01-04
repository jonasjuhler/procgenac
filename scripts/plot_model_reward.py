import matplotlib
import matplotlib.pyplot as plt
import pandas as pd
from procgenac.utils import plot_results, get_formatter

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

idxs = [12, 15]
k = "grad_eps"
v = ["0.7", "0.5"]

mt = "PPO"
fig, ax = plt.subplots(1, 1, figsize=(7, 4))
cols = [["darkslategrey", "lightskyblue"], ["darksalmon", "maroon"]]

for i, hp in enumerate(v):
    idx = idxs[i]
    rew_df = pd.read_csv(f"../results/rewards/{mt}_id{idx}_starpilot.csv")
    k_lab = k.replace("_", "\\_")
    ax = plot_results(
        ax,
        rew_df,
        colors=cols[i],
        include_train=False,
        include_std=False,
        model_label=f"{k_lab}={hp}",
    )

formatter = get_formatter()
ax.xaxis.set_major_formatter(formatter)
ax.set_xlabel("Steps")
ax.set_ylabel("Reward")
plt.tight_layout()
plt.legend()
figpath = f"../results/figures/{mt}_{k}_compare.pdf"
plt.savefig(figpath)
