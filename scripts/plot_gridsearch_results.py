import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
from procgenac.utils import plot_results, get_formatter

matplotlib.rcParams["text.usetex"] = True
g_ite = 1

models_df = pd.read_csv("../results/model_configs.csv")
max_test_rews = []
step_max_test = []

for idx in models_df.index:
    row = models_df.loc[idx]
    m_id = row.model_id
    m_type = row.model_type
    env_name = row.env_name

    rew_df = pd.read_csv(f"../results/rewards/{m_type}_id{m_id}_{env_name}.csv")
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
    m_id = row.model_id
    m_type = row.model_type
    rew_df = pd.read_csv(f"../results/rewards/{m_type}_id{m_id}_{env_name}.csv")
    ax = plot_results(ax, rew_df, colors=cols[i], include_train=False, model_label=mt)
formatter = get_formatter()
ax.xaxis.set_major_formatter(formatter)
# ax.set_title("A2C vs. PPO")
ax.set_xlabel("Steps")
ax.set_ylabel("Reward")
plt.legend()
figpath = f"../results/figures/A2CvsPPO_{g_ite}.pdf"
plt.savefig(figpath, format="pdf")

ppo_df = models_df[models_df.model_type == "PPO"].copy()
a2c_df = models_df[models_df.model_type == "A2C"].copy()

comparison_dict = {
    "PPO": {
        "num_levels": [50, 200],
        "adam_lrs": [0.0005, 0.002],
        "adam_eps": [1e-4, 1e-5],
        "grad_eps": [0.5, 0.7],
    },
    "A2C": {
        "num_levels": [50, 200],
        "adam_lrs": [2e-4, 2e-5],
        "adam_eps": [1e-4, 1e-5],
        "grad_eps": [0.3, 0.5],
    },
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
