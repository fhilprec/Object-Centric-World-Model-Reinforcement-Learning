#!/usr/bin/env python3
"""
Evaluation script for DreamerV2's imagine function.
Place this inside dreamerv2/dreamerv2/ to run with the same Docker setup as training.
"""

import pathlib
import sys
import numpy as np

import common
import agent as agent_module


def main():
    # Configuration
    logdir = pathlib.Path('/logdir/atari_pong/dreamerv2/1')
    max_horizon = 50
    num_sequences = 50

    print("="*80)
    print("DreamerV2 Imagine Function Evaluation")
    print("="*80)
    print(f"Logdir: {logdir}")
    print(f"Max horizon: {max_horizon}")
    print()

    # Load config
    config_path = logdir / 'config.yaml'
    configs = common.Config.load(config_path)
    print("Loaded config")

    # Configure TensorFlow exactly like train.py does
    import tensorflow as tf
    tf.config.experimental_run_functions_eagerly(not configs.jit)
    for gpu in tf.config.experimental.list_physical_devices('GPU'):
        tf.config.experimental.set_memory_growth(gpu, True)
    if configs.precision == 16:
        from tensorflow.keras import mixed_precision as prec
        prec.set_global_policy('mixed_float16')
    print("TensorFlow configured")

    # Create environment
    print("Creating environment...")
    suite, task = configs.task.split('_', 1)
    env = common.Atari(task, configs.action_repeat, configs.render_size, configs.atari_grayscale)
    env = common.OneHotAction(env)
    env = common.TimeLimit(env, configs.time_limit)

    # Load replay buffer first (needed for initialization)
    print("Loading replay buffer...")
    replay_dir = logdir / 'eval_episodes'
    if not replay_dir.exists():
        replay_dir = logdir / 'train_episodes'
    replay = common.Replay(replay_dir, **configs.replay)
    dataset = iter(replay.dataset(**configs.dataset))

    # Create agent
    print("Creating agent...")
    step = common.Counter(0)
    agent = agent_module.Agent(configs, env.obs_space, env.act_space, step)

    # CRITICAL: Do a forward pass to initialize all variables (like train.py line 161)
    print("Initializing agent variables with forward pass...")
    train_agent = common.CarryOverState(agent.train)
    init_data = next(dataset)
    train_agent(init_data)

    # NOW load checkpoint (all 85 variables exist now)
    checkpoint_path = logdir / 'variables.pkl'
    print(f"Loading checkpoint...")
    agent.load(checkpoint_path)
    print("Ready!")
    print()

    # Reset dataset iterator
    dataset = iter(replay.dataset(**configs.dataset))

    # Evaluate
    print("Evaluating imagination...")
    horizons = range(1, max_horizon + 1)
    errors = []

    for horizon in horizons:
        horizon_errors = []

        for seq_idx in range(num_sequences):
            try:
                data = next(dataset)
            except StopIteration:
                dataset = iter(replay.dataset(**configs.dataset))
                data = next(dataset)

            data = agent.wm.preprocess(data)

            # Check sequence length
            seq_len = data['action'].shape[1]
            context_len = 5
            if seq_len < context_len + horizon:
                continue

            # Encode and observe context
            embed = agent.wm.encoder(data)
            states, _ = agent.wm.rssm.observe(
                embed[:, :context_len],
                data['action'][:, :context_len],
                data['is_first'][:, :context_len])

            # Get initial state
            init_state = {k: v[:, -1] for k, v in states.items()}

            # Imagine with ground truth actions
            imagined_states = agent.wm.rssm.imagine(
                data['action'][:, context_len:context_len+horizon],
                init_state)

            # Decode imagined frames
            imagined_feat = agent.wm.rssm.get_feat(imagined_states)
            imagined_frames = agent.wm.heads['decoder'](imagined_feat)['image'].mode()

            # Get ground truth
            gt_embed = embed[:, context_len:context_len+horizon]
            gt_states, _ = agent.wm.rssm.observe(
                gt_embed,
                data['action'][:, context_len:context_len+horizon],
                data['is_first'][:, context_len:context_len+horizon],
                init_state)
            gt_feat = agent.wm.rssm.get_feat(gt_states)
            gt_frames = agent.wm.heads['decoder'](gt_feat)['image'].mode()

            # Compute MSE
            mse = np.mean((imagined_frames.numpy() - gt_frames.numpy()) ** 2)
            horizon_errors.append(mse)

        if horizon_errors:
            avg_error = np.mean(horizon_errors)
            errors.append(avg_error)
            if horizon % 10 == 0 or horizon == 1:
                print(f"  Horizon {horizon:2d}: MSE = {avg_error:.6f}")
        else:
            errors.append(np.nan)

    print()
    print("Evaluation complete!")
    print()

    # Save results
    results_data = {
        'horizons': np.array(list(horizons)),
        'errors': np.array(errors),
        'config': {
            'num_sequences': num_sequences,
            'max_horizon': max_horizon,
            'task': configs.task,
        }
    }

    output_path = logdir / 'horizon_errors.npy'
    np.save(output_path, results_data)
    print(f"Saved: {output_path}")

    txt_path = logdir / 'horizon_errors.txt'
    with open(txt_path, 'w') as f:
        f.write("Horizon\tMSE\n")
        for h, e in zip(horizons, errors):
            f.write(f"{h}\t{e:.8f}\n")
    print(f"Saved: {txt_path}")

    print()
    print("="*80)
    print("Summary")
    print("="*80)
    print(f"Horizon  1: {errors[0]:.6f}")
    print(f"Horizon 10: {errors[9]:.6f}")
    print(f"Horizon 25: {errors[24]:.6f}")
    print(f"Horizon 50: {errors[49]:.6f}")
    print("="*80)


if __name__ == '__main__':
    main()
