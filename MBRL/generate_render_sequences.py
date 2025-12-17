import time
import jax
import jax.numpy as jnp
import optax
import pickle
import os
import sys
from tqdm import tqdm
from typing import Tuple, Any
import numpy as np

from worldmodel_mlp import collect_experience, create_env, compare_real_vs_model


def main():

    frame_stack_size = 4
    frame_skip = 4

    env = create_env(frame_stack_size=frame_stack_size, frame_skip=frame_skip)

    test_data = collect_experience(
                env,
                num_episodes=1,
                max_steps_per_episode=1000,
                frame_stack_size=frame_stack_size,
                exploration_rate=0.5,
                seed=int(time.time()),  # Different seed for fresh data
            )


    obs = test_data["obs"]
    actions = test_data["actions"]

    # Load the trained world model
    checkpoint_path = "worldmodel_mlp.pkl"
    if not os.path.exists(checkpoint_path):
        print(f"No model found at {checkpoint_path}")
        print("Run 'python worldmodel_mlp.py train' first")
        return

    with open(checkpoint_path, "rb") as f:
        model_data = pickle.load(f)
        dynamics_params = model_data.get("dynamics_params", model_data["params"])
        normalization_stats = model_data.get("normalization_stats", None)

    state_mean = normalization_stats["mean"]
    state_std = normalization_stats["std"]

    # Initialize the world model
    from model_architectures import PongMLPDeep
    world_model = PongMLPDeep(1)  # MODEL_SCALE_FACTOR = 1

    # Set up rendering for pixel-space MSE calculation
    from obs_state_converter import pong_flat_observation_to_state
    from jaxatari.games.jax_pong import PongRenderer, JaxPong
    from jaxatari.wrappers import AtariWrapper
    from worldmodel_mlp import flatten_obs
    from PIL import Image

    renderer = PongRenderer()

    # Create unflattener for state conversion
    game = JaxPong()
    temp_env = AtariWrapper(
        game,
        sticky_actions=False,
        episodic_life=False,
        frame_stack_size=frame_stack_size,
        frame_skip=frame_skip,
    )
    dummy_obs, _ = temp_env.reset(jax.random.PRNGKey(0))
    _, unflattener = flatten_obs(dummy_obs, single_state=True)

    # Pick 10 random starting points (ensure we have room for 20-step sequences)
    num_sequences = 10
    sequence_length = 20
    max_start_idx = len(obs) - sequence_length - 1

    rng = jax.random.PRNGKey(int(time.time()))
    rng, sample_key = jax.random.split(rng)
    start_indices = jax.random.choice(sample_key, max_start_idx, shape=(num_sequences,), replace=False)
    start_indices = np.array(start_indices)

    print(f"Selected {num_sequences} random starting points")
    print(f"Generating sequences of length {sequence_length}")

    # Storage for predictions and ground truth (both observation and pixel space)
    all_real_sequences = []
    all_predicted_sequences = []
    all_real_frames = []  # 84x74 cropped grayscale frames
    all_predicted_frames = []  # 84x74 cropped grayscale frames

    def obs_to_grayscale_84x74(obs_vec, unflattener, frame_stack_size):
        """Convert observation vector to 84x74 grayscale image (cropped to remove score)."""
        # Add zeros for score if obs dim is 48
        if obs_vec.shape[0] == 48:
            obs_for_state = jnp.concatenate([obs_vec, jnp.zeros(8)])
        else:
            obs_for_state = obs_vec

        # Convert to state and render
        state = pong_flat_observation_to_state(
            obs_for_state, unflattener, frame_stack_size=frame_stack_size
        )
        raster = renderer.render(state)
        img_array = np.array(raster * 255, dtype=np.uint8)

        # Convert to PIL Image, convert to grayscale, resize to 84x84
        img = Image.fromarray(img_array)
        img_gray = img.convert('L')  # Convert to grayscale

        # Rotate 90 degrees clockwise to match DreamerV2 orientation
        img_gray = img_gray.rotate(-90, expand=True)

        img_resized = img_gray.resize((84, 84), Image.BILINEAR)

        # Crop top 10 pixels to remove score area
        img_array = np.array(img_resized, dtype=np.float32)
        img_cropped = img_array[10:, :]  # Shape: (74, 84)

        return img_cropped

    # Generate sequences from each starting point
    for start_idx in tqdm(start_indices, desc="Generating sequences"):
        start_idx = int(start_idx)

        # Get the real sequence
        real_sequence = obs[start_idx:start_idx + sequence_length + 1]  # +1 for initial obs
        all_real_sequences.append(real_sequence)

        # Render real frames to 84x74 grayscale (cropped)
        real_frames = []
        for step in range(sequence_length + 1):
            frame = obs_to_grayscale_84x74(real_sequence[step], unflattener, frame_stack_size)
            real_frames.append(frame)
        all_real_frames.append(np.stack(real_frames))

        # Generate predicted sequence using the world model
        predicted_sequence = []
        predicted_frames = []
        current_obs = obs[start_idx]
        predicted_sequence.append(current_obs)

        # Render initial frame
        initial_frame = obs_to_grayscale_84x74(current_obs, unflattener, frame_stack_size)
        predicted_frames.append(initial_frame)

        lstm_state = None  # Initialize LSTM state for the model

        for step in range(sequence_length):
            # Get the action that was taken in reality
            action = actions[start_idx + step]

            # Normalize observation
            normalized_obs = (current_obs - state_mean) / state_std

            # Predict next observation
            normalized_pred, lstm_state = world_model.apply(
                dynamics_params,
                rng,
                normalized_obs,
                jnp.array([action]),
                lstm_state
            )

            # Denormalize
            pred_obs = jnp.round(normalized_pred.squeeze() * state_std + state_mean)
            predicted_sequence.append(pred_obs)

            # Render predicted frame to 84x74 grayscale (cropped)
            pred_frame = obs_to_grayscale_84x74(pred_obs, unflattener, frame_stack_size)
            predicted_frames.append(pred_frame)

            # Update current observation for next prediction
            current_obs = pred_obs

        all_predicted_sequences.append(jnp.stack(predicted_sequence))
        all_predicted_frames.append(np.stack(predicted_frames))

    # Convert to arrays for analysis
    all_real_sequences = jnp.stack(all_real_sequences)  # Shape: (num_sequences, sequence_length+1, obs_dim)
    all_predicted_sequences = jnp.stack(all_predicted_sequences)  # Shape: (num_sequences, sequence_length+1, obs_dim)
    all_real_frames = np.stack(all_real_frames)  # Shape: (num_sequences, sequence_length+1, 74, 84)
    all_predicted_frames = np.stack(all_predicted_frames)  # Shape: (num_sequences, sequence_length+1, 74, 84)

    print(f"\nReal sequences shape: {all_real_sequences.shape}")
    print(f"Predicted sequences shape: {all_predicted_sequences.shape}")
    print(f"Real frames shape (84x74 cropped grayscale): {all_real_frames.shape}")
    print(f"Predicted frames shape (84x74 cropped grayscale): {all_predicted_frames.shape}")

    # Compute MSE for each horizon length in PIXEL SPACE (84x74 cropped grayscale)
    mse_per_horizon_pixels = []
    for horizon in range(1, sequence_length + 1):
        # Compare predictions at this horizon with ground truth in pixel space
        # all_predicted_frames[:, horizon] vs all_real_frames[:, horizon]
        # Shape: (num_sequences, 74, 84)
        squared_errors = (all_predicted_frames[:, horizon] - all_real_frames[:, horizon]) ** 2
        mse = np.mean(squared_errors)
        mse_per_horizon_pixels.append(float(mse))

    print("\n=== Mean MSE per Horizon Length (84x74 Cropped Grayscale Pixel Space) ===")
    for horizon, mse in enumerate(mse_per_horizon_pixels, start=1):
        if horizon % 5 == 0 or horizon <= 5:  # Print every 5th horizon and first 5
            print(f"Horizon {horizon:2d}: MSE = {mse:.4f}")

    # Also compute observation-space MSE for comparison
    mse_per_horizon_obs = []
    for horizon in range(1, sequence_length + 1):
        squared_errors = (all_predicted_sequences[:, horizon] - all_real_sequences[:, horizon]) ** 2
        mse = jnp.mean(squared_errors)
        mse_per_horizon_obs.append(float(mse))

    print("\n=== Mean MSE per Horizon Length (Observation Space - for reference) ===")
    for horizon, mse in enumerate(mse_per_horizon_obs, start=1):
        if horizon % 5 == 0 or horizon <= 5:
            print(f"Horizon {horizon:2d}: MSE = {mse:.4f}")

    # Save results
    results = {
        "start_indices": start_indices,
        "all_real_sequences": all_real_sequences,
        "all_predicted_sequences": all_predicted_sequences,
        "all_real_frames": all_real_frames,  # 84x84 grayscale frames
        "all_predicted_frames": all_predicted_frames,  # 84x84 grayscale frames
        "mse_per_horizon_pixels": mse_per_horizon_pixels,  # Primary metric (pixel space)
        "mse_per_horizon_obs": mse_per_horizon_obs,  # For reference (observation space)
        "sequence_length": sequence_length,
        "num_sequences": num_sequences,
    }

    results_path = "sequence_predictions.pkl"
    with open(results_path, "wb") as f:
        pickle.dump(results, f)
    print(f"\nSaved results to {results_path}")

    # Save a single sequence as PNG images
    print("\n=== Saving Sequence as PNG Images ===")

    # Pick a random sequence to visualize
    example_idx = np.random.choice(num_sequences)
    real_frames_seq = all_real_frames[example_idx]  # 84x84 grayscale frames
    pred_frames_seq = all_predicted_frames[example_idx]  # 84x84 grayscale frames
    start_idx_val = start_indices[example_idx]

    print(f"Rendering sequence starting at index {start_idx_val}")

    from PIL import Image, ImageDraw, ImageFont

    # Create output directory
    output_dir = "sequence_pngs"
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

    print(f"\n=== Saved {sequence_length} PNG images to {output_dir}/ ===")
    print(f"Images are 84x84 grayscale frames scaled up {RENDER_SCALE}x for visibility")
    print(f"To create a video from these images, you can use:")
    print(f"  ffmpeg -framerate 10 -i {output_dir}/frame_%04d.png -c:v libx264 -pix_fmt yuv420p sequence_video.mp4")


if __name__ == "__main__":
    main()
