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

g_ite = 3
from_id = 0
idxs = [12, 10]
v = ["200", "30"]

mt = "PPO"
fig, ax = plt.subplots(1, 1, figsize=(7, 4))
cols = [["darkslategrey", "lightskyblue"], ["maroon", "darksalmon"]]
k = "num_levels"
for i, hp in enumerate(v):
    best_m_idx = idxs[i]
    rew_df = pd.read_csv(f"../results/rewards/{mt}_id{best_m_idx}_starpilot.csv")
    inc_train = True if k == "num_levels" else False
    k_lab = "training\\_levels"
    ax = plot_results(
        ax,
        rew_df,
        colors=cols[i],
        include_train=inc_train,
        include_std=False,
        model_label=f"{k_lab}={hp}",
    )

formatter = get_formatter()
ax.xaxis.set_major_formatter(formatter)
ax.set_xlabel("Steps")
ax.set_ylabel("Reward")
plt.legend()
figpath = "../results/figures/num_levels_generalization.png"
plt.savefig(figpath, dpi=600)
plt.show()
