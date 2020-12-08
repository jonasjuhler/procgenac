import matplotlib
import matplotlib.pyplot as plt
import pandas as pd
from procgenac.utils import plot_results, get_formatter

matplotlib.rcParams["text.usetex"] = True
g_ite = 3
from_id = 0

models_df = pd.read_csv("../results/model_configs.csv")
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

mt = "PPO"
fig, ax = plt.subplots(1, 1, figsize=(7, 4))
cols = [["darkslategrey", "lightskyblue"], ["maroon", "darksalmon"]]
v = ["30", "200"]
best_m_idxs = [10, 12]
k = "num_levels"
for i, hp in enumerate(v):
    best_m_idx = best_m_idxs[i]
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
# figpath = f"../results/figures/{mt}_{k}_{g_ite}.pdf"
# plt.savefig(figpath, format="pdf")
plt.show()
