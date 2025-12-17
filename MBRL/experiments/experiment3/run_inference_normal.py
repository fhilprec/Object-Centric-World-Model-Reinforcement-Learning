#!/usr/bin/env python3
"""
Normal inference script - baseline evaluation.
Runs the trained Dreamer agent in the Pong environment and collects episode rewards.
"""
import os
import pathlib
import pickle
import sys
import numpy as np
from tqdm import tqdm

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['MUJOCO_GL'] = 'egl'

import tensorflow as tf
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)

# Add DreamerV2 to path
sys.path.insert(0, '/workspace/dreamerv2')
sys.path.insert(0, '/workspace/dreamerv2/dreamerv2')

import common
import agent as agent_module


def main():
    print("=" * 80)
    print("DreamerV2 Normal Inference - Baseline Evaluation")
    print("=" * 80)
    print()

    # Configuration
    logdir = pathlib.Path('/logdir/atari_pong/dreamerv2/1')
    checkpoint_path = logdir / 'variables.pkl'
    num_episodes = 20

    if not checkpoint_path.exists():
        print(f"❌ No checkpoint found at {checkpoint_path}")
        print("Please train DreamerV2 first")
        return

    # Load config
    print("Loading configuration...")
    config_path = logdir / 'config.yaml'
    configs = common.Config.load(config_path)
    print("✓ Config loaded")

    # Configure TensorFlow
    print("Configuring TensorFlow...")
    tf.config.experimental_run_functions_eagerly(not configs.jit)
    for gpu in tf.config.experimental.list_physical_devices('GPU'):
        tf.config.experimental.set_memory_growth(gpu, True)
    if configs.precision == 16:
        from tensorflow.keras import mixed_precision as prec
        prec.set_global_policy('mixed_float16')
    print("✓ TensorFlow configured")

    # Create environment
    print("Creating environment...")
    suite, task = configs.task.split('_', 1)
    env = common.Atari(task, configs.action_repeat, configs.render_size, configs.atari_grayscale)
    env = common.OneHotAction(env)
    env = common.TimeLimit(env, configs.time_limit)
    print("✓ Environment created")

    # Load replay buffer for initialization
    print("Loading replay buffer for initialization...")
    replay_dir = logdir / 'eval_episodes'
    if not replay_dir.exists():
        replay_dir = logdir / 'train_episodes'
    replay = common.Replay(replay_dir, **configs.replay)
    dataset = iter(replay.dataset(**configs.dataset))
    print("✓ Replay buffer loaded")

    # Create agent
    print("Creating agent...")
    step = common.Counter(0)
    agent = agent_module.Agent(configs, env.obs_space, env.act_space, step)

    # Initialize agent variables with forward pass
    print("Initializing agent variables...")
    train_agent = common.CarryOverState(agent.train)
    init_data = next(dataset)
    train_agent(init_data)

    # Load checkpoint
    print(f"Loading checkpoint from {checkpoint_path}...")
    agent.load(checkpoint_path)
    print("✓ Agent loaded and ready")
    print()

    # Run episodes and collect rewards
    print(f"Running {num_episodes} episodes...")
    print("-" * 80)

    all_rewards = []
    episode_lengths = []

    for episode_idx in tqdm(range(num_episodes), desc="Episodes"):
        # Reset environment
        obs = env.reset()
        state = None
        done = False
        episode_reward = 0.0
        episode_length = 0

        while not done:
            # Get action from agent
            # Convert observation to expected format (add batch dimension)
            obs_dict = {}
            for k, v in obs.items():
                if isinstance(v, (int, float, bool)):
                    obs_dict[k] = np.array([v])
                else:
                    obs_dict[k] = v[None]
            action_dict, state = agent.policy(obs_dict, state, mode='eval')

            # Extract action and remove batch dimension
            # Keep as dictionary for OneHotAction wrapper
            action = {k: v[0] for k, v in action_dict.items()}

            # Step environment
            obs = env.step(action)
            episode_reward += obs['reward']
            episode_length += 1
            done = obs['is_last']

        all_rewards.append(episode_reward)
        episode_lengths.append(episode_length)

        if (episode_idx + 1) % 5 == 0:
            recent_mean = np.mean(all_rewards[-5:])
            print(f"  Episode {episode_idx + 1:2d}: Reward = {episode_reward:6.1f} (Recent avg: {recent_mean:6.1f})")

    print("-" * 80)
    print()

    # Compute statistics
    mean_reward = np.mean(all_rewards)
    std_reward = np.std(all_rewards)
    min_reward = np.min(all_rewards)
    max_reward = np.max(all_rewards)
    mean_length = np.mean(episode_lengths)

    print("=" * 80)
    print("NORMAL INFERENCE RESULTS")
    print("=" * 80)
    print(f"Episodes: {num_episodes}")
    print(f"Mean reward: {mean_reward:.2f} ± {std_reward:.2f}")
    print(f"Reward range: [{min_reward:.1f}, {max_reward:.1f}]")
    print(f"Mean episode length: {mean_length:.1f}")
    print("=" * 80)
    print()

    # Save results
    results = {
        'rewards': all_rewards,
        'episode_lengths': episode_lengths,
        'mean_reward': mean_reward,
        'std_reward': std_reward,
        'min_reward': min_reward,
        'max_reward': max_reward,
        'num_episodes': num_episodes,
    }

    output_path = "experiment3_normal_results.pkl"
    with open(output_path, "wb") as f:
        pickle.dump(results, f)
    print(f"✓ Saved results to {output_path}")
    print()


if __name__ == "__main__":
    main()
