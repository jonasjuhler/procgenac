import pandas as pd
import matplotlib.pyplot as plt
from procgenac.utils import plot_results

filename = "../results/rewards/PPO_starpilot.csv"
df = pd.read_csv(filename)

fig, ax = plt.subplots(1, 1)
ax = plot_results(ax, df)
plt.show()
