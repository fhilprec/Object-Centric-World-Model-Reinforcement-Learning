#!/usr/bin/env python3
"""
DreamerV2 World Model evaluation with image-based MSE.
Mirrors generate_render_sequences.py from the JAX implementation.
Samples one episode, picks 100 random starting points, evaluates 50-step rollouts.
"""
import os
import pathlib
import pickle
import sys
import time
import numpy as np
from tqdm import tqdm
from PIL import Image, ImageDraw, ImageFont

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['MUJOCO_GL'] = 'egl'

import tensorflow as tf
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)

# Add DreamerV2 to path
sys.path.insert(0, '/workspace/dreamerv2')

import common
import agent as agent_module


def main():
    print("=" * 80)
    print("DreamerV2 World Model - Sequence Generation")
    print("=" * 80)
    print()

    # Configuration
    logdir = pathlib.Path('/logdir/atari_pong/dreamerv2/1')
    checkpoint_path = logdir / 'variables.pkl'

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
    print("Loading replay buffer...")
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

    # Use existing replay buffer data instead of collecting new episode
    print("Loading test data from replay buffer...")

    # Collect one long sequence from the dataset
    test_data = next(dataset)
    test_data = agent.wm.preprocess(test_data)

    # Extract observations and actions
    obs_tensor = test_data['image']  # Shape: (batch, time, H, W, C)
    action_tensor = test_data['action']  # Shape: (batch, time, action_dim)

    # Take the first batch element and convert to numpy
    obs_array = obs_tensor[0].numpy()  # (time, H, W, C)
    action_array = action_tensor[0].numpy()  # (time, action_dim)

    print(f"✓ Loaded test data with {obs_array.shape[0]} steps")
    print(f"   Observation shape: {obs_array.shape}")
    print(f"   Action shape: {action_array.shape}")
    print()

    # Convert observations to grayscale 84x84 (like JAX implementation)
    def obs_to_grayscale_84x84(obs_hwc):
        """Convert observation to 84x84 grayscale."""
        img = Image.fromarray((obs_hwc * 255).astype(np.uint8))
        img_gray = img.convert('L')  # Convert to grayscale
        img_resized = img_gray.resize((84, 84), Image.BILINEAR)
        return np.array(img_resized, dtype=np.float32)

    # Pick 100 random starting points
    num_sequences = 100
    sequence_length = 50
    max_start_idx = obs_array.shape[0] - sequence_length - 1

    if max_start_idx < num_sequences:
        print(f"⚠ Not enough steps for {num_sequences} sequences. Using {max_start_idx} sequences.")
        num_sequences = max_start_idx

    np.random.seed(int(time.time()))
    start_indices = np.random.choice(max_start_idx, size=num_sequences, replace=False)

    print(f"Selected {num_sequences} random starting points")
    print(f"Generating sequences of length {sequence_length}")
    print()

    # Storage for predictions and ground truth
    all_real_frames = []  # 84x84 grayscale frames
    all_predicted_frames = []  # 84x84 grayscale frames

    print("Processing sequences...")
    # Generate sequences from each starting point
    for start_idx in tqdm(start_indices, desc="Generating sequences"):
        start_idx = int(start_idx)

        # Get the real sequence of observations
        real_sequence = obs_array[start_idx:start_idx + sequence_length + 1]  # +1 for initial obs

        # Convert real frames to 84x84 grayscale
        real_frames = []
        for step in range(sequence_length + 1):
            frame = obs_to_grayscale_84x84(real_sequence[step])
            real_frames.append(frame)
        all_real_frames.append(np.stack(real_frames))

        # Generate predicted sequence using DreamerV2's world model
        predicted_frames = []

        # Get initial observation
        initial_obs = obs_array[start_idx]

        # Convert initial frame to 84x84 grayscale
        initial_frame = obs_to_grayscale_84x84(initial_obs)
        predicted_frames.append(initial_frame)

        # Prepare data for encoding (need batch dimension and time dimension)
        context_data = {
            'image': tf.convert_to_tensor(obs_array[start_idx:start_idx+1][None], dtype=tf.float32),  # (1, 1, H, W, C)
            'action': tf.convert_to_tensor(action_array[start_idx:start_idx+1][None], dtype=tf.float32),  # (1, 1, action_dim)
        }

        # Encode initial observation to latent state
        embed = agent.wm.encoder(context_data)
        post, prior = agent.wm.dynamics.observe(embed, context_data['action'])

        # Start from the last latent state
        latent = {k: v[:, -1] for k, v in post.items()}  # (1, latent_dim)

        # Rollout the world model
        for step in range(sequence_length):
            # Get the action that was taken in reality
            real_action = action_array[start_idx + step:start_idx + step + 1]  # (action_dim,)
            action_tensor = tf.convert_to_tensor(real_action[None], dtype=tf.float32)  # (1, action_dim)

            # Imagine next latent state
            latent = agent.wm.dynamics.img_step(latent, action_tensor)

            # Decode latent to image
            pred_image = agent.wm.heads['decoder'](latent).mode()
            pred_image = pred_image.numpy()[0]  # Remove batch dimension

            # Convert predicted frame to 84x84 grayscale
            pred_frame = obs_to_grayscale_84x84(pred_image)
            predicted_frames.append(pred_frame)

        all_predicted_frames.append(np.stack(predicted_frames))

    # Convert to arrays for analysis
    all_real_frames = np.stack(all_real_frames)  # Shape: (num_sequences, sequence_length+1, 84, 84)
    all_predicted_frames = np.stack(all_predicted_frames)  # Shape: (num_sequences, sequence_length+1, 84, 84)

    print()
    print(f"Real frames shape (84x84 grayscale): {all_real_frames.shape}")
    print(f"Predicted frames shape (84x84 grayscale): {all_predicted_frames.shape}")
    print()

    # Compute MSE for each horizon length in PIXEL SPACE (84x84 grayscale)
    mse_per_horizon_pixels = []
    for horizon in range(1, sequence_length + 1):
        # Compare predictions at this horizon with ground truth in pixel space
        squared_errors = (all_predicted_frames[:, horizon] - all_real_frames[:, horizon]) ** 2
        mse = np.mean(squared_errors)
        mse_per_horizon_pixels.append(float(mse))

    print("=" * 80)
    print("Mean MSE per Horizon Length (84x84 Grayscale Pixel Space)")
    print("=" * 80)
    for horizon, mse in enumerate(mse_per_horizon_pixels, start=1):
        if horizon % 5 == 0 or horizon <= 5:  # Print every 5th horizon and first 5
            print(f"Horizon {horizon:2d}: MSE = {mse:.4f}")
    print()

    # Save results
    results = {
        "start_indices": start_indices,
        "all_real_frames": all_real_frames,  # 84x84 grayscale frames
        "all_predicted_frames": all_predicted_frames,  # 84x84 grayscale frames
        "mse_per_horizon_pixels": mse_per_horizon_pixels,  # Primary metric (pixel space)
        "sequence_length": sequence_length,
        "num_sequences": num_sequences,
    }

    results_path = "/logdir/sequence_predictions_dreamer.pkl"
    with open(results_path, "wb") as f:
        pickle.dump(results, f)
    print(f"Saved results to {results_path}")
    print()

    # Save a single sequence as PNG images
    print("=" * 80)
    print("Saving Sequence as PNG Images")
    print("=" * 80)

    # Pick a random sequence to visualize
    example_idx = np.random.choice(num_sequences)
    real_frames_seq = all_real_frames[example_idx]  # 84x84 grayscale frames
    pred_frames_seq = all_predicted_frames[example_idx]  # 84x84 grayscale frames
    start_idx_val = start_indices[example_idx]

    print(f"Rendering sequence starting at index {start_idx_val}")

    # Create output directory
    output_dir = "/logdir/sequence_pngs_dreamer"
    os.makedirs(output_dir, exist_ok=True)

    # Use 84x84 grayscale images scaled up for visibility
    FRAME_SIZE = 84
    RENDER_SCALE = 6  # Scale up 84x84 to 504x504
    GAP = 20

    print(f"Saving {sequence_length} PNG images to {output_dir}/")

    for step in tqdm(range(sequence_length), desc="Saving PNGs"):
        # Get the 84x84 grayscale frames
        real_frame = real_frames_seq[step].astype(np.uint8)  # Shape: (84, 84)
        pred_frame = pred_frames_seq[step].astype(np.uint8)  # Shape: (84, 84)

        # Convert to PIL Images and scale up
        real_img = Image.fromarray(real_frame, mode='L')  # 'L' mode for grayscale
        real_img = real_img.resize((FRAME_SIZE * RENDER_SCALE, FRAME_SIZE * RENDER_SCALE), Image.NEAREST)

        pred_img = Image.fromarray(pred_frame, mode='L')
        pred_img = pred_img.resize((FRAME_SIZE * RENDER_SCALE, FRAME_SIZE * RENDER_SCALE), Image.NEAREST)

        # Create combined image with labels
        combined_width = FRAME_SIZE * RENDER_SCALE * 2 + GAP
        combined_height = FRAME_SIZE * RENDER_SCALE + 60  # Extra space for labels
        combined = Image.new('RGB', (combined_width, combined_height), color='black')

        # Convert grayscale images to RGB for pasting
        real_img_rgb = real_img.convert('RGB')
        pred_img_rgb = pred_img.convert('RGB')

        # Paste the images
        combined.paste(real_img_rgb, (0, 30))
        combined.paste(pred_img_rgb, (FRAME_SIZE * RENDER_SCALE + GAP, 30))

        # Add labels
        draw = ImageDraw.Draw(combined)
        try:
            # Try to use a nice font
            font_large = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 24)
            font_small = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 18)
        except:
            # Fall back to default font
            font_large = ImageFont.load_default()
            font_small = ImageFont.load_default()

        # Draw text labels
        draw.text((10, 5), "Real (84x84 Gray)", fill='white', font=font_large)
        draw.text((FRAME_SIZE * RENDER_SCALE + GAP + 10, 5), f"Predicted (Step {step+1}/{sequence_length})", fill='white', font=font_large)

        # Draw pixel-space MSE at the bottom
        mse_text = f"Pixel MSE (84x84 gray): {mse_per_horizon_pixels[step]:.2f}"
        draw.text((combined_width // 2 - 150, combined_height - 30), mse_text, fill='white', font=font_small)

        # Save the image
        output_path = os.path.join(output_dir, f"frame_{step:04d}.png")
        combined.save(output_path)

    print()
    print("=" * 80)
    print(f"Saved {sequence_length} PNG images to {output_dir}/")
    print("=" * 80)
    print(f"Images are 84x84 grayscale frames scaled up {RENDER_SCALE}x for visibility")
    print(f"To create a video from these images, you can use:")
    print(f"  ffmpeg -framerate 10 -i {output_dir}/frame_%04d.png -c:v libx264 -pix_fmt yuv420p sequence_video_dreamer.mp4")
    print()


if __name__ == "__main__":
    main()
