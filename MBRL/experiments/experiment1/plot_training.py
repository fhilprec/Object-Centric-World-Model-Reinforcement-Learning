import re
import matplotlib.pyplot as plt
import argparse
import json
import sys
import os
import glob

parser = argparse.ArgumentParser(description='Plot training progress from log file')
parser.add_argument('--input', '-i', required=True, help='Path pattern for training log files (e.g., imagined_training_log_*)')
parser.add_argument('--metrics', '-m', required=True, help='Path to metrics.jsonl file')
parser.add_argument('--output', '-o', default='training_curve.png', help='Output path for plot')
args = parser.parse_args()

MAX_SAMPLES= 10_000_000

# Find all matching input files
input_files = sorted(glob.glob(args.input))
if not input_files:
    print(f"Error: No training log files found matching pattern: {args.input}", file=sys.stderr)
    sys.exit(1)

print(f"Found {len(input_files)} training log files:")
for f in input_files:
    print(f"  - {f}")

if not os.path.exists(args.metrics):
    print(f"Error: Metrics file not found: {args.metrics}", file=sys.stderr)
    sys.exit(1)

import numpy as np
from scipy.interpolate import make_interp_spline
from scipy.signal import savgol_filter

# Parse all training logs
all_samples = []
all_means = []
all_stds = []

for log_file in input_files:
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

    all_samples.append(np.array(samples))
    all_means.append(np.array(means))
    all_stds.append(np.array(stds))

# Find minimum length across all runs
min_length = min(len(s) for s in all_samples)
print(f"\nCropping all runs to minimum length: {min_length}")

# Crop all arrays to minimum length
all_samples = [s[:min_length] for s in all_samples]
all_means = [m[:min_length] for m in all_means]
all_stds = [s[:min_length] for s in all_stds]

# Compute mean across all runs
samples_arr = all_samples[0]  # Use samples from first run (should be same for all)
means_arr = np.mean(all_means, axis=0)
stds_arr = np.mean(all_stds, axis=0)


print(f'Samples: {samples_arr}')
print(f'Means: {means_arr}')
print(f'Stds: {stds_arr}')

# Parse DreamerV2 metrics
dreamer_steps = []
dreamer_returns = []
with open(args.metrics, 'r') as f:
    for line in f:
        data = json.loads(line)
        if 'train_total_steps' in data and 'train_return' in data:
            dreamer_steps.append(data['train_total_steps'])
            dreamer_returns.append(data['train_return'])
        if data.get('train_total_steps', 0) >= MAX_SAMPLES:
            break

dreamer_steps = np.array(dreamer_steps)
dreamer_returns = np.array(dreamer_returns)

# Apply smoothing filter to data
window_length = min(11, len(means_arr) if len(means_arr) % 2 == 1 else len(means_arr) - 1)
if window_length >= 5:
    means_filtered = savgol_filter(means_arr, window_length, 3)
    stds_filtered = savgol_filter(stds_arr, window_length, 3)
else:
    means_filtered = means_arr
    stds_filtered = stds_arr

# Create smooth curves
x_smooth = np.linspace(samples_arr.min(), samples_arr.max(), 300)
spl = make_interp_spline(samples_arr, means_filtered, k=3)
means_smooth = spl(x_smooth)
spl_std = make_interp_spline(samples_arr, stds_filtered, k=3)
stds_smooth = spl_std(x_smooth)

if len(dreamer_steps) > 3:
    window_length_dreamer = min(11, len(dreamer_returns) if len(dreamer_returns) % 2 == 1 else len(dreamer_returns) - 1)
    if window_length_dreamer >= 5:
        dreamer_filtered = savgol_filter(dreamer_returns, window_length_dreamer, 3)
    else:
        dreamer_filtered = dreamer_returns

    dreamer_x_smooth = np.linspace(dreamer_steps.min(), dreamer_steps.max(), 300)
    dreamer_spl = make_interp_spline(dreamer_steps, dreamer_filtered, k=3)
    dreamer_smooth = dreamer_spl(dreamer_x_smooth)
else:
    dreamer_x_smooth = dreamer_steps
    dreamer_smooth = dreamer_returns

plt.figure(figsize=(12, 6))
plt.plot(x_smooth, means_smooth, linewidth=2, label='Model-based (Ours)', color='blue')
plt.fill_between(x_smooth, means_smooth - stds_smooth, means_smooth + stds_smooth, alpha=0.3, color='blue')
plt.plot(dreamer_x_smooth, dreamer_smooth, linewidth=2, label='DreamerV2', color='orange')
plt.xlabel('Train Total Steps')
plt.ylabel('Train Return')
plt.title('Training Progress Comparison')
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(args.output)
print(f'Plot saved to {args.output}')
