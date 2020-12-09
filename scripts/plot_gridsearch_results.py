import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
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
g_ite = 4

models_df = pd.read_csv("../results/model_configs.csv")
hp_cols = models_df.columns[models_df.columns != "model_id"]
max_test_rews = []
step_max_test = []

for idx in models_df.index:
    row = models_df.loc[idx]
    m_id = row.model_id
    m_type = row.model_type
    env_name = row.env_name
    try:
        rew_df = pd.read_csv(f"../results/rewards/{m_type}_id{m_id}_{env_name}.csv")
    except FileNotFoundError:
        max_test_rews.append(0)
        step_max_test.append(0)
        continue
    max_test_rews.append(rew_df.test_rewards_mean.max())
    step_max_test.append(int(rew_df.loc[rew_df.test_rewards_mean.idxmax()].steps))

models_df["test_reward"] = max_test_rews
models_df["step_max_test"] = step_max_test

fig, ax = plt.subplots(1, 1, figsize=(7, 4))
cols = [["darkslategrey", "lightskyblue"], ["maroon", "darksalmon"]]
for i, mt in enumerate(["A2C", "PPO"]):
    best_m_idx = models_df[
        (models_df.model_type == mt) & (models_df.step_max_test > 5_000_000)
    ].test_reward.idxmax()
    row = models_df.loc[best_m_idx]
    print(f"\nBEST MODEL STATS ({mt}):")
    print(row)
    m_id = row.model_id
    m_type = row.model_type
    rew_df = pd.read_csv(f"../results/rewards/{m_type}_id{m_id}_{env_name}.csv")
    ax = plot_results(ax, rew_df, colors=cols[i], include_train=False, model_label=mt)
formatter = get_formatter()
ax.xaxis.set_major_formatter(formatter)
ax.set_xlabel("Steps")
ax.set_ylabel("Reward")
plt.legend()
figpath = f"../results/figures/A2CvsPPO_{g_ite}.pdf"
plt.savefig(figpath, format="pdf")
# plt.show()

ppo_df = models_df[models_df.model_type == "PPO"].copy()
a2c_df = models_df[models_df.model_type == "A2C"].copy()
comparison_dict = {
    "PPO": {c: ppo_df[c].unique() for c in hp_cols if len(ppo_df[c].unique()) > 1},
    "A2C": {c: a2c_df[c].unique() for c in hp_cols if len(a2c_df[c].unique()) > 1},
}

for mt in ["A2C", "PPO"]:
    comp_dict = comparison_dict[mt]
    for k, v in comp_dict.items():
        fig, ax = plt.subplots(1, 1, figsize=(7, 4))
        cols = [["darkslategrey", "lightskyblue"], ["maroon", "darksalmon"]]
        for i, hp in enumerate(v):
            hp_df = models_df[
                (models_df.model_type == mt)
                & (models_df[k] == hp)
                & (models_df.step_max_test > 5_000_000)
            ]
            if hp_df.empty:
                continue
            best_m_idx = hp_df.test_reward.idxmax()
            row = models_df.loc[best_m_idx]
            m_id = row.model_id
            m_type = row.model_type
            rew_df = pd.read_csv(f"../results/rewards/{m_type}_id{m_id}_{env_name}.csv")
            inc_train = True if k == "num_levels" else False
            k_lab = k.replace("_", "\\_")
            ax = plot_results(
                ax,
                rew_df,
                colors=cols[i],
                include_train=inc_train,
                model_label=f"{k_lab}={hp}",
            )

        formatter = get_formatter()
        ax.xaxis.set_major_formatter(formatter)
        ax.set_xlabel("Steps")
        ax.set_ylabel("Reward")
        plt.legend()
        figpath = f"../results/figures/{mt}_{k}_{g_ite}.pdf"
        plt.savefig(figpath, format="pdf")
        # plt.show()
