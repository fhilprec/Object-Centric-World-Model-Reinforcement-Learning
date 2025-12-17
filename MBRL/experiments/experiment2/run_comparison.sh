#!/bin/bash
set -e

EXPERIMENT_DIR="/home/fhilprec/MBRL/MBRL/experiments/experiment2"

echo "================================================================================"
echo "DreamerV2 vs JAX World Model Comparison"
echo "================================================================================"
echo ""
echo "This script will:"
echo "  1. Generate DreamerV2 prediction sequences (10 sequences, 20 steps)"
echo "  2. Generate JAX world model prediction sequences (10 sequences, 20 steps)"
echo "  3. Create side-by-side comparison plots and images"
echo ""

# Step 1: Run DreamerV2 evaluation
echo "Step 1/3: Running DreamerV2 evaluation..."
echo "--------------------------------------------------------------------------------"

# Copy the evaluation script into the dreamerv2/dreamerv2 directory
cp ${EXPERIMENT_DIR}/generate_render_sequences_dreamer.py ${EXPERIMENT_DIR}/dreamerv2/dreamerv2/

echo "Running DreamerV2 in Docker container..."
sudo docker run -it --rm --device=nvidia.com/gpu=all \
  -e XLA_FLAGS=--xla_gpu_strict_conv_algorithm_picker=false \
  -e TF_XLA_FLAGS=--tf_xla_enable_xla_devices=false \
  -v ~/logdir:/logdir \
  -v ${EXPERIMENT_DIR}/dreamerv2:/workspace/dreamerv2 \
  -v ${EXPERIMENT_DIR}:/workspace/experiment2 \
  -w /workspace/dreamerv2 \
  dreamerv2 \
  python3 dreamerv2/generate_render_sequences_dreamer.py

echo ""
echo "✓ DreamerV2 evaluation complete!"
echo ""

# Step 2: Run JAX world model evaluation
echo "Step 2/3: Running JAX world model evaluation..."
echo "--------------------------------------------------------------------------------"

cd ~/MBRL/MBRL
source ~/MBRL/venv_mbrl/bin/activate
python3 generate_render_sequences.py
deactivate

echo ""
echo "✓ JAX world model evaluation complete!"
echo ""

# Step 3: Copy results and create comparison
echo "Step 3/3: Creating comparison plots..."
echo "--------------------------------------------------------------------------------"

cd ${EXPERIMENT_DIR}

# Copy JAX results
echo "Copying JAX results..."
cp ~/MBRL/MBRL/sequence_predictions.pkl sequence_predictions_jax.pkl
mkdir -p sequence_pngs_jax
cp ~/MBRL/MBRL/sequence_pngs/*.png sequence_pngs_jax/ 2>/dev/null || true
rm -rf ~/MBRL/MBRL/sequence_pngs/

# Copy DreamerV2 results
echo "Copying DreamerV2 results..."
cp ~/logdir/sequence_predictions_dreamer.pkl .
mkdir -p sequence_pngs_dreamer
cp ~/logdir/sequence_pngs_dreamer/*.png sequence_pngs_dreamer/ 2>/dev/null || true

# Create comparison plots
echo "Generating comparison plots..."
source ~/MBRL/venv_mbrl/bin/activate
python3 compare_models.py
deactivate

echo ""
echo "================================================================================"
echo "COMPARISON COMPLETE!"
echo "================================================================================"
echo ""
echo "Results:"
echo "  DreamerV2 Sequences:    ${EXPERIMENT_DIR}/sequence_pngs_dreamer/"
echo "  JAX Sequences:          ${EXPERIMENT_DIR}/sequence_pngs_jax/"
echo "  Comparison Plot:        ${EXPERIMENT_DIR}/model_comparison_plot.png"
echo ""
echo "Both models evaluated on:"
echo "  - 10 random starting points"
echo "  - 20-step prediction horizon"
echo "  - 84x74 cropped grayscale pixel-space MSE (score area removed)"
echo ""
echo "Open the plot to see which model performs better!"
echo ""
