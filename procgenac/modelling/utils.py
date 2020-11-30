import os
import torch
from procgenac.utils import Storage, save_video


def train_model(
    model,
    env,
    device,
    num_epochs,
    batch_size,
    adam_lr,
    adam_eps,
    num_steps,
    total_steps,
    get_test_error=False,
    eval_env=None,
    verbose=False,
):
    model.to(device=device)

    # Define optimizer
    # these are reasonable values but probably not optimal
    optimizer = torch.optim.Adam(model.parameters(), lr=adam_lr, eps=adam_eps)

    # Define temporary storage
    # we use this to collect transitions during each iteration
    storage = Storage(env.observation_space.shape, num_steps, env.num_envs, device)

    # Run training
    steps = []
    train_rewards = []
    test_rewards = []
    obs = env.reset()
    eval_obs = eval_env.reset()
    step = 0
    n_updates = 0
    update_ite = total_steps // (env.num_envs * num_steps * 100) + 1
    while step < total_steps:

        # Use policy to collect data for num_steps steps
        model.eval()
        for _ in range(num_steps):
            # Use policy
            action, log_prob, value = model.act(obs)

            # Take step in environment
            next_obs, reward, done, info = env.step(action)

            # Store data
            storage.store(obs, action, reward, done, info, log_prob, value)

            # Update current observation
            obs = next_obs

        # Update stats
        if n_updates % update_ite == 0:
            steps.append(step)
            train_rewards.append(storage.get_reward())
            if verbose:
                print(f"Step: {step}\tMean reward: {storage.get_reward().mean()}")
            if get_test_error:
                test_rew, eval_obs = evaluate_model(model, eval_env, eval_obs)
                test_rewards.append(test_rew)
        step += env.num_envs * num_steps

        # Add the last observation to collected data
        _, _, value = model.act(obs)
        storage.store_last(obs, value)

        # Compute return and advantage
        storage.compute_return_advantage()

        # Optimize policy
        model.train()
        for epoch in range(num_epochs):

            # Iterate over batches of transitions
            generator = storage.get_generator(batch_size)

            for batch in generator:
                b_obs, b_action, b_log_pi, b_value, b_returns, b_delta, b_advantage = batch

                # Get current policy outputs
                policy, value = model(b_obs)

                # Calculate and backpropagate loss
                loss = model.criterion(batch, policy, value)
                loss.backward()

                # Clip gradients
                torch.nn.utils.clip_grad_norm_(model.parameters(), model.grad_eps)

                # Update policy
                optimizer.step()
                optimizer.zero_grad()

        # Number of update iterations performed
        n_updates += 1

    if get_test_error:
        rewards = (torch.stack(train_rewards), torch.stack(test_rewards))
    else:
        rewards = torch.stack(train_rewards)

    return model, (steps, rewards)


def save_model(model, filename):
    model_path = f"../models/{filename}"
    if not os.path.isdir(os.path.dirname(model_path)):
        os.makedirs(os.path.dirname(model_path))
    torch.save(model.state_dict, model_path)


def evaluate_model(model, eval_env, obs, num_steps=512, video=False, video_filename=None):
    frames = []
    total_reward = []

    # Evaluate policy
    model.eval()
    for _ in range(num_steps):

        # Use policy
        action, log_prob, value = model.act(obs)

        # Take step in environment
        obs, reward, done, info = eval_env.step(action)
        total_reward.append(torch.Tensor(reward))

        # Render environment and store
        if video:
            frame = (torch.Tensor(eval_env.render(mode="rgb_array")) * 255.0).byte()
            frames.append(frame)

    if video:
        # Save frames as video
        save_video(frames, video_filename)

    # Calculate total reward
    return torch.stack(total_reward).sum(0), obs
