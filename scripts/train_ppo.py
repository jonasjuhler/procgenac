import os
import torch
import imageio
from procgenac.utils import make_env, Storage
from procgenac.modelling.ppo import Encoder, Policy

# Hyperparameters
num_epochs = 3
batch_size = 512
device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
print(f"Running on {device}")

# Environment hyperparameters
total_steps = 10
num_envs = 32
num_steps = 256
num_levels = 10
env_name = "starpilot"

# Model hyperparameters
feature_dim = 64  # Length of output feature vector from Encoder
grad_eps = 0.5  # Clip value for norm of gradients
eps = 0.2  # needed for clip values in loss function
value_coef = 0.5  # coefficient in loss
entropy_coef = 0.01  # coefficient in loss

# Define environment
env = make_env(n_envs=num_envs, env_name=env_name, num_levels=num_levels)

# Define network
in_channels = 3  # RGB
n_actions = env.action_space.n
encoder = Encoder(in_channels=in_channels, feature_dim=feature_dim)
policy = Policy(
    encoder=encoder,
    feature_dim=feature_dim,
    num_actions=n_actions,
    c1=value_coef,
    c2=entropy_coef,
    eps=eps,
    device=device,
)
policy.to(device=device)

# Define optimizer
# these are reasonable values but probably not optimal
optimizer = torch.optim.Adam(policy.parameters(), lr=5e-3, eps=1e-5)

# Define temporary storage
# we use this to collect transitions during each iteration
storage = Storage(env.observation_space.shape, num_steps, num_envs, device)

# Run training
obs = env.reset()
step = 0
while step < total_steps:

    # Use policy to collect data for num_steps steps
    policy.eval()
    for _ in range(num_steps):
        # Use policy
        action, log_prob, value = policy.act(obs)

        # Take step in environment
        next_obs, reward, done, info = env.step(action)

        # Store data
        storage.store(obs, action, reward, done, info, log_prob, value)

        # Update current observation
        obs = next_obs

    # Add the last observation to collected data
    _, _, value = policy.act(obs)
    storage.store_last(obs, value)

    # Compute return and advantage
    storage.compute_return_advantage()

    # Optimize policy
    policy.train()
    for epoch in range(num_epochs):

        # Iterate over batches of transitions
        generator = storage.get_generator(batch_size)
        for batch in generator:
            b_obs, b_action, b_log_pi, b_value, b_returns, b_advantage = batch

            # Get current policy outputs
            dist, value = policy(b_obs)
            log_pi = dist.log_prob(b_action)

            # Clipped policy objective
            pi_loss = policy.pi_loss(log_pi, b_log_pi, b_advantage)

            # Clipped value function objective
            vf_loss = policy.value_loss(value, b_returns)

            # Entropy loss
            entropy = policy.entropy_loss(dist)

            # Backpropagate losses
            loss = torch.mean(-1 * (pi_loss - vf_loss + entropy))
            loss.backward()

            # Clip gradients
            torch.nn.utils.clip_grad_norm_(policy.parameters(), grad_eps)

            # Update policy
            optimizer.step()
            optimizer.zero_grad()

    # Update stats
    step += num_envs * num_steps
    print(f"Step: {step}\tMean reward: {storage.get_reward()}")

print("Completed training!")
# Save snapshot of current policy
model_path = f"../models/policy_{env_name}.pt"
if not os.path.isdir(os.path.dirname(model_path)):
    os.makedirs(os.path.dirname(model_path))
torch.save(policy.state_dict, model_path)

# Make evaluation environment (unseen levels)
eval_env = make_env(num_envs, env_name=env_name, start_level=num_levels, num_levels=num_levels)
obs = eval_env.reset()

frames = []
total_reward = []

# Evaluate policy
policy.eval()
for _ in range(512):

    # Use policy
    action, log_prob, value = policy.act(obs)

    # Take step in environment
    obs, reward, done, info = eval_env.step(action)
    total_reward.append(torch.Tensor(reward))

    # Render environment and store
    frame = (torch.Tensor(eval_env.render(mode="rgb_array")) * 255.0).byte()
    frames.append(frame)

# Calculate average return
total_reward = torch.stack(total_reward).sum(0).mean(0)
print("Average return:", total_reward)

# Save frames as video
video_path = f"../videos/vid_{env_name}.mp4"
if not os.path.isdir(os.path.dirname(video_path)):
    os.makedirs(os.path.dirname(video_path))
frames = torch.stack(frames)
imageio.mimsave(video_path, frames, fps=25)
