import torch
from procgenac.utils import make_env, save_rewards
from procgenac.modelling.utils import train_model, save_model, evaluate_model
from procgenac.modelling.ppo import PPO
from procgenac.modelling.encoder import Encoder
import time

before = time.time()
# Model name
model_name = "PolicyPPO"

# Hyperparameters
num_epochs = 3
batch_size = 1024
adam_lr = 5e-4
adam_eps = 1e-5
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Running on {device}")

# Environment hyperparameters
total_steps = 2_000_000
num_envs = 32
num_steps = 256
num_levels = 200
env_name = "starpilot"

# Model hyperparameters
feature_dim = 128  # Length of output feature vector from Encoder
grad_eps = 0.5  # Clip value for norm of gradients
eps = 0.2  # needed for clip values in loss function
value_coef = 0.5  # coefficient in loss
entropy_coef = 0.01  # coefficient in loss

# Define training and test environment (unseen levels)
env = make_env(n_envs=num_envs, env_name=env_name, num_levels=num_levels)
eval_env = make_env(
    n_envs=num_envs,
    env_name=env_name,
    start_level=num_levels,
    num_levels=num_levels,
    normalize_reward=False,
)

# Define network
policy_ppo = PPO(
    encoder=Encoder(in_channels=3, feature_dim=feature_dim),
    feature_dim=feature_dim,
    num_actions=env.action_space.n,
    c1=value_coef,
    c2=entropy_coef,
    eps=eps,
    grad_eps=grad_eps,
    device=device,
)

# Train model
policy_ppo, (steps, rewards) = train_model(
    model=policy_ppo,
    env=env,
    device=device,
    num_epochs=num_epochs,
    batch_size=batch_size,
    adam_lr=adam_lr,
    adam_eps=adam_eps,
    num_steps=num_steps,
    total_steps=total_steps,
    get_test_error=False,
    eval_env=eval_env,
    verbose=True,
)
print("Completed training!")

# Store training results
filename = f"{model_name}_{env_name}.csv"
save_rewards(steps, rewards, filename)

# Save snapshot of current policy
filename = f"{model_name}_{env_name}.pt"
save_model(policy_ppo, filename)

# Make env for video
video_env = make_env(
    n_envs=1,
    env_name=env_name,
    start_level=num_levels,
    num_levels=num_levels,
    normalize_reward=False,
)
obs = video_env.reset()
total_reward, _ = evaluate_model(
    model=policy_ppo,
    eval_env=video_env,
    obs=obs,
    num_steps=512,
    video=True,
    video_filename=f"{model_name}_{env_name}.mp4",
)
print("Average return:", total_reward.mean(0).item())

print(f"Time taken: {(time.time() - before)/60:.1f} minutes")
