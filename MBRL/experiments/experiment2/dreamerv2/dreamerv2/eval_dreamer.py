#!/usr/bin/env python3
"""
Consolidated DreamerV2 evaluation script for world model imagination accuracy.
Tests prediction error over different horizon lengths (1-50 steps).

This script should be run inside the DreamerV2 Docker container.
Usage: docker run ... dreamerv2 python3 dreamerv2/dreamerv2/eval_dreamer.py
"""

import pathlib
import sys
import numpy as np
import tensorflow as tf

# Add dreamerv2 to path when run from workspace
workspace_path = pathlib.Path(__file__).parent / 'dreamerv2' / 'dreamerv2'
if workspace_path.exists():
    sys.path.insert(0, str(workspace_path))

import common
import agent as agent_module


def main():
    # Configuration
    logdir = pathlib.Path('/logdir/atari_pong/dreamerv2/1')
    max_horizon = 50
    num_sequences = 50
    context_len = 5  # Number of steps to observe before imagination

    print("="*80)
    print("DreamerV2 World Model Evaluation")
    print("="*80)
    print(f"Logdir: {logdir}")
    print(f"Max horizon: {max_horizon}")
    print(f"Sequences per horizon: {num_sequences}")
    print(f"Context length: {context_len}")
    print()

    # Load config
    config_path = logdir / 'config.yaml'
    if not config_path.exists():
        raise FileNotFoundError(f"Config not found: {config_path}")

    configs = common.Config.load(config_path)
    print("✓ Loaded config")

    # Configure TensorFlow exactly like train.py does
    tf.config.experimental_run_functions_eagerly(not configs.jit)
    for gpu in tf.config.experimental.list_physical_devices('GPU'):
        tf.config.experimental.set_memory_growth(gpu, True)
    if configs.precision == 16:
        from tensorflow.keras import mixed_precision as prec
        prec.set_global_policy('mixed_float16')
    print("✓ TensorFlow configured")

    # Create environment (needed for observation/action spaces)
    print("Creating environment...")
    suite, task = configs.task.split('_', 1)
    env = common.Atari(task, configs.action_repeat, configs.render_size, configs.atari_grayscale)
    env = common.OneHotAction(env)
    env = common.TimeLimit(env, configs.time_limit)
    print("✓ Environment created")

    # Load replay buffer (needed for initialization and data)
    print("Loading replay buffer...")
    replay_dir = logdir / 'eval_episodes'
    if not replay_dir.exists():
        replay_dir = logdir / 'train_episodes'
    replay = common.Replay(replay_dir, **configs.replay)
    dataset = iter(replay.dataset(**configs.dataset))
    print(f"✓ Replay buffer loaded ({replay.stats['total_steps']} steps)")

    # Create agent
    print("Creating agent...")
    step = common.Counter(0)
    agent = agent_module.Agent(configs, env.obs_space, env.act_space, step)
    print("✓ Agent created")

    # CRITICAL: Do a forward pass to initialize all variables (like train.py line 161)
    print("Initializing agent variables with forward pass...")
    train_agent = common.CarryOverState(agent.train)
    init_data = next(dataset)
    train_agent(init_data)
    print("✓ Agent variables initialized")

    # NOW load checkpoint (all 85 variables exist now)
    checkpoint_path = logdir / 'variables.pkl'
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
    print(f"Loading checkpoint from: {checkpoint_path.name}")
    agent.load(checkpoint_path)
    print("✓ Checkpoint loaded")
    print()

    # Reset dataset iterator for evaluation
    dataset = iter(replay.dataset(**configs.dataset))

    # Evaluate imagination at different horizons
    print("Evaluating imagination accuracy...")
    print("-" * 80)
    horizons = range(1, max_horizon + 1)
    errors = []

    # Track some detailed stats for debugging
    frame_errors_by_horizon = {}  # horizon -> list of per-frame errors

    for horizon in horizons:
        horizon_errors = []
        horizon_frame_errors = []

        for seq_idx in range(num_sequences):
            try:
                data = next(dataset)
            except StopIteration:
                dataset = iter(replay.dataset(**configs.dataset))
                data = next(dataset)

            # Preprocess data
            data = agent.wm.preprocess(data)

            # Check sequence length
            seq_len = data['action'].shape[1]
            if seq_len < context_len + horizon:
                continue

            # Encode observations
            embed = agent.wm.encoder(data)

            # Observe context to get initial state
            states, _ = agent.wm.rssm.observe(
                embed[:, :context_len],
                data['action'][:, :context_len],
                data['is_first'][:, :context_len])

            # Get initial state (last context state)
            init_state = {k: v[:, -1] for k, v in states.items()}

            # Imagine forward using ground truth actions
            imagined_states = agent.wm.rssm.imagine(
                data['action'][:, context_len:context_len+horizon],
                init_state)

            # Decode imagined frames
            imagined_feat = agent.wm.rssm.get_feat(imagined_states)
            imagined_frames = agent.wm.heads['decoder'](imagined_feat)['image'].mode()

            # Get ground truth frames
            gt_embed = embed[:, context_len:context_len+horizon]
            gt_states, _ = agent.wm.rssm.observe(
                gt_embed,
                data['action'][:, context_len:context_len+horizon],
                data['is_first'][:, context_len:context_len+horizon],
                init_state)
            gt_feat = agent.wm.rssm.get_feat(gt_states)
            gt_frames = agent.wm.heads['decoder'](gt_feat)['image'].mode()

            # Compute MSE
            pred = imagined_frames.numpy()
            gt = gt_frames.numpy()
            mse = np.mean((pred - gt) ** 2)
            horizon_errors.append(mse)

            # Also track per-frame errors for debugging
            frame_errors = np.mean((pred - gt) ** 2, axis=(0, 2, 3, 4))  # Average over batch and spatial dims
            horizon_frame_errors.extend(frame_errors)

        if horizon_errors:
            avg_error = np.mean(horizon_errors)
            errors.append(avg_error)
            frame_errors_by_horizon[horizon] = horizon_frame_errors

            if horizon % 10 == 0 or horizon == 1:
                std_error = np.std(horizon_errors)
                min_error = np.min(horizon_errors)
                max_error = np.max(horizon_errors)
                print(f"Horizon {horizon:2d}: MSE = {avg_error:.6f} ± {std_error:.6f} "
                      f"(min={min_error:.6f}, max={max_error:.6f})")
        else:
            errors.append(np.nan)
            print(f"Horizon {horizon:2d}: No valid sequences (skipped)")

    print("-" * 80)
    print()

    # Save results
    results_data = {
        'horizons': np.array(list(horizons)),
        'errors': np.array(errors),
        'frame_errors_by_horizon': frame_errors_by_horizon,
        'config': {
            'num_sequences': num_sequences,
            'max_horizon': max_horizon,
            'context_len': context_len,
            'task': configs.task,
        }
    }

    output_path = logdir / 'horizon_errors.npy'
    np.save(output_path, results_data)
    print(f"✓ Saved binary results: {output_path}")

    txt_path = logdir / 'horizon_errors.txt'
    with open(txt_path, 'w') as f:
        f.write("# DreamerV2 World Model Evaluation Results\n")
        f.write(f"# Task: {configs.task}\n")
        f.write(f"# Sequences: {num_sequences}, Context: {context_len}\n")
        f.write("#\n")
        f.write("Horizon\tMSE\n")
        for h, e in zip(horizons, errors):
            f.write(f"{h}\t{e:.8f}\n")
    print(f"✓ Saved text results: {txt_path}")

    print()
    print("="*80)
    print("DREAMERV2 EVALUATION SUMMARY")
    print("="*80)
    valid_errors = [e for e in errors if not np.isnan(e)]
    if valid_errors:
        print(f"Horizon  1: MSE = {errors[0]:.6f}")
        print(f"Horizon 10: MSE = {errors[9]:.6f}")
        print(f"Horizon 25: MSE = {errors[24]:.6f}")
        print(f"Horizon 50: MSE = {errors[49]:.6f}")
        print()
        print(f"Mean error across all horizons: {np.mean(valid_errors):.6f}")
        print(f"Error range: [{np.min(valid_errors):.6f}, {np.max(valid_errors):.6f}]")
    else:
        print("No valid results!")
    print("="*80)


if __name__ == '__main__':
    main()
