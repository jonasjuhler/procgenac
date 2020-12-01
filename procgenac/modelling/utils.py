import os
import time
import torch
from procgenac.utils import Storage, save_video, make_env, save_rewards
from procgenac.modelling.encoder import Encoder
from procgenac.modelling.ppo import PPO
from procgenac.modelling.a2c import A2C


def training_pipeline(param_args, path_to_base, verbose=False, prod=True):

    # Define device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if verbose:
        before = time.time()
        print(f"Running on {device}")

    # Training env initialization
    env_name = param_args.env_name
    num_envs = int(param_args.num_envs)
    num_levels = int(param_args.num_levels)
    env = make_env(n_envs=num_envs, env_name=env_name, num_levels=num_levels)

    # Model hyperparameters
    model_type = param_args.model_type
    feature_dim = int(param_args.feature_dim)
    encoder = Encoder(in_channels=3, feature_dim=feature_dim)

    # Define network
    if model_type == "A2C":
        model = A2C(
            encoder=encoder,
            feature_dim=feature_dim,
            num_actions=env.action_space.n,
            c1=float(param_args.value_coef),
            c2=float(param_args.entropy_coef),
            grad_eps=float(param_args.grad_eps),
            device=device,
        )
    elif model_type == "PPO":
        model = PPO(
            encoder=encoder,
            feature_dim=feature_dim,
            num_actions=env.action_space.n,
            c1=float(param_args.value_coef),
            c2=float(param_args.entropy_coef),
            eps=float(param_args.eps),
            grad_eps=float(param_args.grad_eps),
            device=device,
        )
    if not prod:
        model.name = "test_" + model.name

    # Evaluation env (unseen levels)
    eval_env = make_env(
        num_envs,
        env_name=env_name,
        start_level=num_levels,
        num_levels=num_levels,
        normalize_reward=False,
    )

    # Train model
    model, (steps, rewards) = train_model(
        model=model,
        env=env,
        device=device,
        num_epochs=int(param_args.num_epochs),
        batch_size=int(param_args.batch_size),
        adam_lr=float(param_args.adam_lr),
        adam_eps=float(param_args.adam_eps),
        num_steps=int(param_args.num_steps),
        total_steps=int(param_args.total_steps),
        get_test_error=bool(int(param_args.get_test)),
        eval_env=eval_env,
        verbose=True,
    )

    # Store training results
    filepath = os.path.join(path_to_base, "results", "rewards", f"{model.name}_{env_name}.csv")
    save_rewards(steps, rewards, filepath=filepath)

    # Save snapshot of current policy
    filepath = os.path.join(path_to_base, "models", f"{model.name}_{env_name}.pt")
    save_model(model, filepath=filepath)

    # Make env for generating a video
    video_env = make_env(
        n_envs=1,
        env_name=env_name,
        start_level=num_levels,
        num_levels=num_levels,
        normalize_reward=False,
    )
    obs = video_env.reset()
    filepath = os.path.join(path_to_base, "results", "videos", f"{model.name}_{env_name}.mp4")
    total_reward, _ = evaluate_model(
        model=model,
        eval_env=video_env,
        obs=obs,
        num_steps=int(param_args.num_steps),
        video=True,
        video_filepath=filepath,
    )

    if verbose:
        print("Video return:", total_reward.mean(0).item())
        print(f"Time taken: {(time.time() - before)/60:.1f} minutes")


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
                print(
                    f"Step: {step} \tMean train reward: {storage.get_reward().mean():.4f}",
                    end="" if get_test_error else "\n",
                )
            if get_test_error:
                test_rew, eval_obs = evaluate_model(model, eval_env, eval_obs, num_steps=num_steps)
                test_rewards.append(test_rew)
                if verbose:
                    print(f" \tMean test reward: {test_rew.mean().item():.4f}")

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


def save_model(model, filepath):
    if not os.path.isdir(os.path.dirname(filepath)):
        os.makedirs(os.path.dirname(filepath))
    torch.save(model.state_dict, filepath)


def evaluate_model(model, eval_env, obs, num_steps=256, video=False, video_filepath=None):
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
        save_video(frames, video_filepath)

    # Calculate total reward
    return torch.stack(total_reward).sum(0), obs
