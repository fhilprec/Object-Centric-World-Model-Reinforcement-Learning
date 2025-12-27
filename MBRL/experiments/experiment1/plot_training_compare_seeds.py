import re
import matplotlib.pyplot as plt
import argparse
import sys
import os

parser = argparse.ArgumentParser(description='Plot training progress comparing two seeds')
parser.add_argument('--seed2', required=True, help='Path to seed 2 training log file')
parser.add_argument('--seed4', required=True, help='Path to seed 4 training log file')
parser.add_argument('--output', '-o', default='training_curve_seed_comparison.png', help='Output path for plot')
args = parser.parse_args()

MAX_SAMPLES = 10_000_000

# Verify files exist
if not os.path.exists(args.seed2):
    print(f"Error: Seed 2 training log not found: {args.seed2}", file=sys.stderr)
    sys.exit(1)
if not os.path.exists(args.seed4):
    print(f"Error: Seed 4 training log not found: {args.seed4}", file=sys.stderr)
    sys.exit(1)

import numpy as np
from scipy.interpolate import make_interp_spline
from scipy.signal import savgol_filter

def parse_training_log(log_file):
    """Parse a single training log file following imagined_training_log format"""
    with open(log_file, 'r') as f:
        content = f.read()

    retrains = content.split('Retrained Model')[1:]
    samples, means, stds = [], [], []

    for i, section in enumerate(retrains):
        match = re.search(r'eval_mean_reward=([-\d.]+), eval_std_reward=([-\d.]+)', section)
        resample_match = re.search(r'Resampled=(\d+)', section)
        if match:
            samples.append(int(resample_match.group(1)) + samples[-1] if i > 0 else int(resample_match.group(1)))
            means.append(float(match.group(1)))
            stds.append(float(match.group(2)))
        if samples and samples[-1] >= MAX_SAMPLES:
            break

    return np.array(samples), np.array(means), np.array(stds)

# Parse both seed files
print("Parsing seed 2 training log...")
seed2_samples, seed2_means, seed2_stds = parse_training_log(args.seed2)
print(f"  Found {len(seed2_samples)} data points")

print("Parsing seed 4 training log...")
seed4_samples, seed4_means, seed4_stds = parse_training_log(args.seed4)
print(f"  Found {len(seed4_samples)} data points")

# Apply smoothing filter to seed 2
window_length_2 = min(11, len(seed2_means) if len(seed2_means) % 2 == 1 else len(seed2_means) - 1)
if window_length_2 >= 5:
    seed2_means_filtered = savgol_filter(seed2_means, window_length_2, 3)
    seed2_stds_filtered = savgol_filter(seed2_stds, window_length_2, 3)
else:
    seed2_means_filtered = seed2_means
    seed2_stds_filtered = seed2_stds

# Apply smoothing filter to seed 4
window_length_4 = min(11, len(seed4_means) if len(seed4_means) % 2 == 1 else len(seed4_means) - 1)
if window_length_4 >= 5:
    seed4_means_filtered = savgol_filter(seed4_means, window_length_4, 3)
    seed4_stds_filtered = savgol_filter(seed4_stds, window_length_4, 3)
else:
    seed4_means_filtered = seed4_means
    seed4_stds_filtered = seed4_stds

# Create smooth curves for seed 2
if len(seed2_samples) > 3:
    x_smooth_2 = np.linspace(seed2_samples.min(), seed2_samples.max(), 300)
    spl_2 = make_interp_spline(seed2_samples, seed2_means_filtered, k=3)
    seed2_means_smooth = spl_2(x_smooth_2)
    spl_std_2 = make_interp_spline(seed2_samples, seed2_stds_filtered, k=3)
    seed2_stds_smooth = spl_std_2(x_smooth_2)
else:
    x_smooth_2 = seed2_samples
    seed2_means_smooth = seed2_means_filtered
    seed2_stds_smooth = seed2_stds_filtered

# Create smooth curves for seed 4
if len(seed4_samples) > 3:
    x_smooth_4 = np.linspace(seed4_samples.min(), seed4_samples.max(), 300)
    spl_4 = make_interp_spline(seed4_samples, seed4_means_filtered, k=3)
    seed4_means_smooth = spl_4(x_smooth_4)
    spl_std_4 = make_interp_spline(seed4_samples, seed4_stds_filtered, k=3)
    seed4_stds_smooth = spl_std_4(x_smooth_4)
else:
    x_smooth_4 = seed4_samples
    seed4_means_smooth = seed4_means_filtered
    seed4_stds_smooth = seed4_stds_filtered

# Create the plot
plt.figure(figsize=(12, 6))

# Plot seed 2
plt.plot(x_smooth_2, seed2_means_smooth, linewidth=2, label='no feature weighting', color='blue')
plt.fill_between(x_smooth_2, seed2_means_smooth - seed2_stds_smooth,
                 seed2_means_smooth + seed2_stds_smooth, alpha=0.2, color='blue')

# Plot seed 4
plt.plot(x_smooth_4, seed4_means_smooth, linewidth=2, label='feature weighting', color='green')
plt.fill_between(x_smooth_4, seed4_means_smooth - seed4_stds_smooth,
                 seed4_means_smooth + seed4_stds_smooth, alpha=0.2, color='green')

plt.xlabel('Train Total Steps')
plt.ylabel('Train Return')
plt.title('Training Progress: Seed Comparison')
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(args.output)
print(f'\nPlot saved to {args.output}')
