#!/usr/bin/env python3
"""
Compare DreamerV2 and JAX world model predictions.
Loads both pickle files and creates comparison plots.
"""
import pickle
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path


def main():
    
    
    
    

    # Load DreamerV2 results
    dreamer_path = Path("sequence_predictions_dreamer.pkl")
    if not dreamer_path.exists():
        
        
        return

    with open(dreamer_path, "rb") as f:
        dreamer_results = pickle.load(f)

    
    
    

    # Load JAX results
    jax_path = Path("sequence_predictions_jax.pkl")
    if not jax_path.exists():
        # Try old name for backwards compatibility
        jax_path = Path("sequence_predictions.pkl")
    if not jax_path.exists():
        
        
        return

    with open(jax_path, "rb") as f:
        jax_results = pickle.load(f)

    
    
    
    

    # Extract MSE data
    dreamer_mse = dreamer_results['mse_per_horizon_pixels']
    jax_mse = jax_results['mse_per_horizon_pixels']

    # Calculate standard deviation for JAX model from raw frames
    jax_std_per_horizon = None
    if 'all_real_frames' in jax_results and 'all_predicted_frames' in jax_results:
        real_frames = jax_results['all_real_frames']  # Shape: (num_sequences, seq_len, H, W)
        pred_frames = jax_results['all_predicted_frames']

        # Calculate MSE per sequence per horizon
        # We need to skip the first frame since predictions start from step 1
        mse_per_seq_per_horizon = []
        for seq_idx in range(real_frames.shape[0]):
            mse_per_horizon_this_seq = []
            for horizon_idx in range(1, real_frames.shape[1]):
                real_frame = real_frames[seq_idx, horizon_idx]
                pred_frame = pred_frames[seq_idx, horizon_idx]
                mse = np.mean((real_frame - pred_frame) ** 2)
                mse_per_horizon_this_seq.append(mse)
            mse_per_seq_per_horizon.append(mse_per_horizon_this_seq)

        mse_per_seq_per_horizon = np.array(mse_per_seq_per_horizon)  # Shape: (num_sequences, horizons)
        jax_std_per_horizon = np.std(mse_per_seq_per_horizon, axis=0)

    # Ensure same length
    min_length = min(len(dreamer_mse), len(jax_mse))
    dreamer_mse = dreamer_mse[:min_length]
    jax_mse = jax_mse[:min_length]
    if jax_std_per_horizon is not None:
        jax_std_per_horizon = jax_std_per_horizon[:min_length]

    horizons = np.arange(1, min_length + 1)

    # 
    
    
    
    
    

    for i, horizon in enumerate(horizons):
        if horizon % 5 == 0 or horizon <= 5:
            diff = dreamer_mse[i] - jax_mse[i]
            ratio = dreamer_mse[i] / jax_mse[i] if jax_mse[i] > 0 else float('inf')
            

    

    # Calculate overall statistics
    dreamer_mean = np.mean(dreamer_mse)
    jax_mean = np.mean(jax_mse)
    dreamer_std = np.std(dreamer_mse)
    jax_std = np.std(jax_mse)

    
    
    
    
    
    
    

    # Create comparison plot
    fig, ax = plt.subplots(1, 1, figsize=(14, 8))

    ax.plot(horizons, dreamer_mse, 'o-', label='DreamerV2', linewidth=2, markersize=6, color='blue')
    ax.plot(horizons, jax_mse, 's-', label='JAX World Model', linewidth=2, markersize=6, color='orange')

    # Add standard deviation shading for JAX model if available
    if jax_std_per_horizon is not None:
        ax.fill_between(horizons,
                        jax_mse - jax_std_per_horizon,
                        jax_mse + jax_std_per_horizon,
                        color='orange', alpha=0.2)

    ax.set_xlabel('Prediction Horizon (steps)', fontsize=20)
    ax.set_ylabel('MSE (84x74 cropped grayscale pixels)', fontsize=20)
    ax.set_title('World Model Prediction Error Comparison\n(Lower is Better)', fontsize=20, fontweight='bold')
    ax.legend(fontsize=16)
    ax.grid(True, alpha=0.3)
    ax.set_xlim(0, min_length + 1)

    plt.tight_layout()
    output_path = "model_comparison_plot.png"
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    
    

    # Determine winner
    
    
    
    if dreamer_mean < jax_mean:
        improvement = (1 - dreamer_mean / jax_mean) * 100
        
    elif jax_mean < dreamer_mean:
        improvement = (1 - jax_mean / dreamer_mean) * 100
        
        
    

    
    
    


if __name__ == "__main__":
    main()
