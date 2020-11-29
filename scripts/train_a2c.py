import torch
from procgenac.utils import make_env, Storage, save_model, save_video
from procgenac.modelling.a2c import A2C
from procgenac.modelling.encoder import Encoder

# Model name
model_name = "A2C"

# Hyperparameters
num_epochs = 3
batch_size = 512
device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
print(f"Running on {device}")

# Environment hyperparameters
total_steps = 2000000
num_envs = 32
num_steps = 256
num_levels = 500
env_name = "starpilot"

# Model hyperparameters
feature_dim = 64  # Length of output feature vector from Encoder
grad_eps = 0.5  # Clip value for norm of gradients
value_coef = 0.5  # coefficient in loss
entropy_coef = 0.01  # coefficient in loss

# Define environment
env = make_env(n_envs=num_envs, env_name=env_name, num_levels=num_levels)

# Define network
in_channels = 3  # RGB
n_actions = env.action_space.n
encoder = Encoder(in_channels=in_channels, feature_dim=feature_dim)
a2c_model = A2C(
    encoder=encoder,
    feature_dim=feature_dim,
    num_actions=n_actions,
    c1=value_coef,
    c2=entropy_coef,
    device=device,
)
a2c_model.to(device=device)

# Define optimizer
# these are reasonable values but probably not optimal
optimizer = torch.optim.Adam(a2c_model.parameters(), lr=5e-3, eps=1e-5)

# Define temporary storage
# we use this to collect transitions during each iteration
storage = Storage(env.observation_space.shape, num_steps, num_envs, device)

# Run training
obs = env.reset()
step = 0
while step < total_steps:

    # Use policy to collect data for num_steps steps
    a2c_model.eval()
    for _ in range(num_steps):
        # Use policy
        action, log_prob, value = a2c_model.act(obs)

        # Take step in environment
        next_obs, reward, done, info = env.step(action)

        # Store data
        storage.store(obs, action, reward, done, info, log_prob, value)

        # Update current observation
        obs = next_obs

    # Add the last observation to collected data
    _, _, value = a2c_model.act(obs)
    storage.store_last(obs, value)

    # Compute return and advantage
    storage.compute_return_advantage()

    # Optimize policy
    a2c_model.train()
    for epoch in range(num_epochs):

        # Iterate over batches of transitions
        generator = storage.get_generator(batch_size)

        for batch in generator:
            b_obs, b_action, b_log_pi, b_value, b_returns, b_delta, b_advantage = batch

            # Get current policy outputs
            dist, value = a2c_model(b_obs)
            log_pi = dist.log_prob(b_action)

            # Clipped policy objective
            pi_loss = a2c_model.actor_objective(log_pi, b_advantage)

            # Clipped value function objective
            vf_loss = a2c_model.value_loss(value, b_returns)

            # Entropy loss
            entropy = a2c_model.entropy_objective(dist)

            # Backpropagate losses
            loss = a2c_model.criterion(pi_loss, vf_loss, entropy)
            loss.backward()

            # Clip gradients
            torch.nn.utils.clip_grad_norm_(a2c_model.parameters(), grad_eps)

            # Update policy
            optimizer.step()
            optimizer.zero_grad()

    # Update stats
    step += num_envs * num_steps
    print(f"Step: {step}\tMean reward: {storage.get_reward()}")

print("Completed training!")
# Save snapshot of current policy
save_model(a2c_model, model_name, env_name)

# Make evaluation environment (unseen levels)
eval_env = make_env(num_envs, env_name=env_name, start_level=num_levels, num_levels=num_levels)
obs = eval_env.reset()

frames = []
total_reward = []

# Evaluate policy
a2c_model.eval()
for _ in range(512):

    # Use policy
    action, log_prob, value = a2c_model.act(obs)

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
save_video(frames, model_name, env_name)
