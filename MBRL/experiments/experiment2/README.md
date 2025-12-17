# DreamerV2 vs JAX World Model Comparison

Fair comparison between DreamerV2 and a custom JAX-based world model for Atari Pong.

## Quick Start

Run the complete comparison with a single command:

```bash
cd ~/dreamer
./run_comparison.sh
```

This will:
1. ✅ Evaluate DreamerV2 (10 sequences, 20 steps)
2. ✅ Evaluate JAX world model (10 sequences, 20 steps)
3. ✅ Create comparison plots and visualizations

## Results

After completion, you'll find:

- **Comparison plots:**
  - `model_comparison_plot.png` - MSE curves for both models
  - `model_comparison_ratio.png` - Relative performance ratio

- **Sequence visualizations:**
  - `sequence_pngs_dreamer/` - DreamerV2 predictions vs ground truth
  - `sequence_pngs_jax/` - JAX model predictions vs ground truth

- **Raw data:**
  - `sequence_predictions_dreamer.pkl` - DreamerV2 results
  - `sequence_predictions_jax.pkl` - JAX model results

## Evaluation Methodology

Both models use **identical evaluation**:
- **Metric:** 84×84 grayscale pixel-space MSE
- **Sequences:** 10 random starting points
- **Horizon:** 20 timesteps
- **Actions:** Same ground-truth actions for fair comparison

## Files

- `run_comparison.sh` - Main script (runs everything)
- `generate_render_sequences_dreamer.py` - DreamerV2 evaluation
- `compare_models.py` - Creates comparison plots
- `README_FAIR_COMPARISON.md` - Detailed documentation

## Requirements

- DreamerV2 Docker container (`dreamerv2`)
- Trained DreamerV2 model at `/logdir/atari_pong/dreamerv2/1/`
- JAX world model at `~/MBRL/worldmodel_mlp.pkl`
- Python virtual environment at `~/MBRL/venv_mbrl/`

## Documentation

See [README_FAIR_COMPARISON.md](README_FAIR_COMPARISON.md) for complete documentation.
