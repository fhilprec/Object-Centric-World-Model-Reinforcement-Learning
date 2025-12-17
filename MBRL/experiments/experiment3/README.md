# Experiment 3: Background Robustness Test

## Overview

This experiment tests whether the trained Dreamer model has learned robust object representations or if it relies on background features when making predictions in Pong.

## Hypothesis

**If a human were playing Pong, the background color should not matter** - only the positions and movements of the ball and paddles should be relevant. We test whether the Dreamer model has learned similar robust representations.

**Expected outcome:** The model will likely show significant performance degradation with inverted backgrounds, suggesting it has learned to rely on background features rather than focusing solely on the game objects.

## Methodology

### 1. Normal Inference (Baseline)
- Run the trained Dreamer agent in the Pong environment
- Collect rewards over 20 episodes
- Expected performance: ~20 reward (near-optimal)

### 2. Inverted Inference (Background Test)
- Run the same trained agent with **pixel inversion**: `inverted = 255 - original`
- This changes the background color while preserving object edges and relative contrasts
- Collect rewards over 20 episodes
- Expected performance: Much lower if the model relies on background features

### 3. Comparison
- Statistical comparison (t-test) between normal and inverted conditions
- Visualizations:
  - Box plot comparison
  - Episode-wise reward plot
  - Histogram of reward distributions

## Files

- `run_inference_normal.py` - Normal inference baseline evaluation
- `run_inference_inverted.py` - Inverted pixel inference evaluation
- `compare_results.py` - Analysis and visualization script
- `run_experiment.sh` - Master orchestration script
- `README.md` - This file

## Usage

### Quick Start

Run the entire experiment:

```bash
cd /home/fhilprec/MBRL/MBRL/experiments/experiment3
./run_experiment.sh
```

This will:
1. Run normal inference (baseline) → saves to `/logdir/experiment3_normal_results.pkl`
2. Run inverted inference → saves to `/logdir/experiment3_inverted_results.pkl`
3. Create comparison plots and summary

### Individual Steps

Run each step separately:

```bash
# Step 1: Normal inference
sudo docker run -it --rm --device=nvidia.com/gpu=all \
  -e XLA_FLAGS=--xla_gpu_strict_conv_algorithm_picker=false \
  -e TF_XLA_FLAGS=--tf_xla_enable_xla_devices=false \
  -v ~/logdir:/logdir \
  -v $(pwd):/workspace/experiment3 \
  -w /workspace/experiment3 \
  dreamerv2 \
  python3 /workspace/experiment3/run_inference_normal.py

# Step 2: Inverted inference
sudo docker run -it --rm --device=nvidia.com/gpu=all \
  -e XLA_FLAGS=--xla_gpu_strict_conv_algorithm_picker=false \
  -e TF_XLA_FLAGS=--tf_xla_enable_xla_devices=false \
  -v ~/logdir:/logdir \
  -v $(pwd):/workspace/experiment3 \
  -w /workspace/experiment3 \
  dreamerv2 \
  python3 /workspace/experiment3/run_inference_inverted.py

# Step 3: Create comparison
python3 compare_results.py
```

## Output

### Result Files

- `/logdir/experiment3_normal_results.pkl` - Normal inference rewards and statistics
- `/logdir/experiment3_inverted_results.pkl` - Inverted inference rewards and statistics
- `reward_comparison.png` - Box plot and episode-wise comparison
- `reward_histogram.png` - Distribution histograms
- `comparison_summary.txt` - Detailed text summary

### Example Summary

```
Normal inference:
  Mean reward: 19.85 ± 1.23
  Range: [17.0, 21.0]

Inverted inference:
  Mean reward: -15.42 ± 3.45
  Range: [-21.0, -8.0]

Performance degradation: 177.7%
Statistical significance: t=35.123, p=0.0000
```

## Interpretation

### Performance Degradation Categories

- **< 10% degradation**: Model is ROBUST to background changes
  - Suggests object-centric representations

- **10-50% degradation**: Model shows MODERATE sensitivity
  - Partial reliance on background features

- **> 50% degradation**: Model shows STRONG sensitivity
  - Heavy reliance on background features
  - Model did not learn robust object representations

## Requirements

- Trained DreamerV2 checkpoint at `/logdir/atari_pong/dreamerv2/1/variables.pkl`
- DreamerV2 Docker image built and available
- Python packages: numpy, tensorflow, matplotlib, scipy

## Technical Details

### Pixel Inversion

The inversion is applied to observations before feeding them to the agent:

```python
def invert_pixels(obs):
    if obs.dtype == np.uint8:
        return 255 - obs
    else:
        return 1.0 - obs  # For normalized [0, 1] images
```

### Pong Reward Scale

Pong rewards range from -21 to +21:
- +1 when agent scores
- -1 when opponent scores
- Game ends at 21 points

A well-trained agent achieves ~20 average reward.

## Related Experiments

- **Experiment 1**: Initial world model evaluation
- **Experiment 2**: DreamerV2 vs JAX world model comparison
- **Experiment 3**: Background robustness test (this experiment)
