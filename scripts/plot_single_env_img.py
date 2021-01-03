from procgenac.utils import make_env
import matplotlib.pyplot as plt
import numpy as np

use_bg = False
num_steps = 610
action = np.array([0])

env = make_env(n_envs=1, use_backgrounds=use_bg)

# Evaluate policy
for _ in range(num_steps):

    # Take step in environment
    obs, reward, done, info = env.step(action)

frame = env.render(mode="rgb_array")
fig, ax = plt.subplots(1, 1)
ax.imshow(frame)
plt.axis("off")
plt.savefig(
    "../results/figures/starpilot_no_bg.pdf", format="pdf", pad_inches=0, bbox_inches="tight"
)
