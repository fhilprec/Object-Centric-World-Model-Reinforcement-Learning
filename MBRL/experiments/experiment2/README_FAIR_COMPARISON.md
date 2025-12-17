#### LLM Generated File explaining what this experiment does ######


# Fair World Model Comparison

This directory contains scripts for a fair comparison between DreamerV2 and JAX World Model.

## Overview

Both models are evaluated using the **exact same methodology**:
- Sample test episodes from replay buffer
- Pick 10 random starting points
- Generate 20-step rollouts from each starting point
- Convert predictions to 84x84 grayscale images
- Compute MSE in pixel space

This ensures an apples-to-apples comparison.

## Files

### Evaluation Scripts

1. **`generate_render_sequences_dreamer.py`**
   - DreamerV2 evaluation script
   - Runs inside Docker container
   - Saves results to `/logdir/sequence_predictions_dreamer.pkl`
   - Creates visualization images in `/logdir/sequence_pngs_dreamer/`

2. **`generate_render_sequences.py`** (in ~/MBRL/MBRL/)
   - JAX world model evaluation script
   - Runs in venv_mbrl virtual environment
   - Saves results to `~/MBRL/MBRL/sequence_predictions.pkl`
   - Creates visualization images in `~/MBRL/MBRL/sequence_pngs/`

3. **`compare_models.py`**
   - Loads both pickle files and creates comparison plots
   - Shows MSE per horizon for both models
   - Creates error ratio visualization
   - Determines which model performs better

### Runner Script

**`run_comparison.sh`** - Single script that runs everything!

## Usage

### Quick Start (Recommended)

```bash
cd ~/dreamer
./run_comparison.sh
```

This will:
1. Run DreamerV2 evaluation (in Docker)
2. Run JAX evaluation (in venv)
3. Copy results to ~/dreamer/
4. Create comparison plots

### Manual Evaluation

If you want to run each step separately, see the commands in `run_comparison.sh`.

## Results

After running the evaluation, you'll find:

### Data Files
- `sequence_predictions_dreamer.pkl` - DreamerV2 results
- `sequence_predictions_jax.pkl` - JAX results

Each pickle file contains:
```python
{
    'start_indices': array of starting indices,
    'all_real_frames': (10, 21, 84, 84) array of ground truth,
    'all_predicted_frames': (10, 21, 84, 84) array of predictions,
    'mse_per_horizon_pixels': list of MSE values per horizon,
    'sequence_length': 20,
    'num_sequences': 10,
}
```

### Visualizations

1. **Sequence Images**
   - `sequence_pngs_dreamer/` - DreamerV2 frame-by-frame comparisons
   - `sequence_pngs_jax/` - JAX frame-by-frame comparisons
   - Each image shows: Real | Predicted | MSE value

2. **Comparison Plots**
   - `model_comparison_plot.png` - Side-by-side MSE comparison
     - Full 50-step comparison
     - Zoomed view of first 20 steps
   - `model_comparison_ratio.png` - Relative performance over time
     - Shows which model is better at each horizon

## Evaluation Methodology

### Why 84x84 Grayscale?

- **Standard Atari benchmark**: Most RL papers use 84x84 grayscale
- **Fair comparison**: Both models convert to same format
- **Pixel-space metric**: MSE computed on actual images, not features

### Conversion Process

**DreamerV2**: 64×64 RGB → Grayscale → Resize to 84×84
```python
img = Image.fromarray((obs * 255).astype(np.uint8))
img_gray = img.convert('L')
img_84 = img_gray.resize((84, 84), Image.BILINEAR)
```

**JAX Model**: State Vector → Render to 210×160 RGB → Grayscale → Resize to 84×84
```python
state = pong_flat_observation_to_state(obs, unflattener)
raster = renderer.render(state)  # (3, 210, 160)
img_gray = Image.fromarray(raster[0] * 255).convert('L')
img_84 = img_gray.resize((84, 84), Image.BILINEAR)
```

### MSE Calculation

```python
# For each horizon h from 1 to 50:
squared_errors = (predicted_frames[:, h] - real_frames[:, h]) ** 2
mse[h] = np.mean(squared_errors)
```

## Interpreting Results

### MSE Values

- **Lower is better**
- Typical range: 0-100 for normalized images
- MSE increases with horizon length (predictions get worse over time)

### Comparison Plots

1. **model_comparison_plot.png**
   - Shows both models' MSE curves
   - Look at which line is lower (better)
   - Check if gap widens or narrows over time

2. **model_comparison_ratio.png**
   - Ratio = DreamerV2 MSE / JAX MSE
   - < 1.0: DreamerV2 is better
   - > 1.0: JAX is better
   - = 1.0: Equal performance

### Example Interpretation

```
Horizon  DreamerV2  JAX Model  Difference  Ratio
1        2.5        3.2        -0.7        0.78
5        8.1        9.3        -1.2        0.87
10       15.2       18.5       -3.3        0.82
```

This shows DreamerV2 is better (lower MSE) across all horizons by about 18-22%.

## Troubleshooting

### DreamerV2 Docker Issues

If you get "permission denied" errors:
```bash
sudo usermod -aG docker $USER
newgrp docker
```

### JAX Environment Issues

If imports fail:
```bash
cd ~/MBRL
source venv_mbrl/bin/activate
pip install jax jaxlib pillow matplotlib tqdm
```

### Missing Checkpoints

**DreamerV2**: Checkpoint should be at `/logdir/pong_atari/1/checkpoint.ckpt`
**JAX**: Model should be at `~/MBRL/worldmodel_mlp.pkl`

If missing, you need to train the models first.

## Next Steps

After comparing models:

1. **Analyze visualizations** - Look at the sequence PNG images to see what errors look like
2. **Check horizon-specific performance** - Which model is better at short vs long-term predictions?
3. **Statistical significance** - Run multiple evaluations with different seeds
4. **Ablation studies** - Try different model architectures or hyperparameters

## Notes

- Both evaluations use the same random seed for reproducibility
- MSE is computed on normalized images [0, 255]
- Action conditioning: Both models use the same actions from the test episode
- LSTM state: Properly maintained throughout rollouts for both models
