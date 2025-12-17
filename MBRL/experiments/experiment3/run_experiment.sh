#!/bin/bash
#
# Experiment 3: Background Robustness Test
#
# This script runs both normal and inverted inference evaluations,
# then creates comparison visualizations.
#

set -e  # Exit on error

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
EXPERIMENT_DIR="$SCRIPT_DIR"

echo "================================================================================"
echo "Experiment 3: Background Robustness Test"
echo "================================================================================"
echo ""
echo "This experiment tests whether the Dreamer model is robust to background changes"
echo "by comparing performance with normal vs. inverted pixel values."
echo ""
echo "Expected outcome:"
echo "  - Normal inference: ~20 reward (trained performance)"
echo "  - Inverted inference: Much lower reward if model relies on background features"
echo ""
echo "Experiment directory: $EXPERIMENT_DIR"
echo ""

# Check if Docker image exists
if ! docker image inspect dreamerv2 >/dev/null 2>&1; then
    echo "❌ Docker image 'dreamerv2' not found."
    echo "Please build the DreamerV2 Docker image first."
    exit 1
fi

# Check if checkpoint exists
if [ ! -f ~/logdir/atari_pong/dreamerv2/1/variables.pkl ]; then
    echo "❌ No trained checkpoint found at ~/logdir/atari_pong/dreamerv2/1/variables.pkl"
    echo "Please train DreamerV2 first."
    exit 1
fi

echo "✓ Docker image found"
echo "✓ Checkpoint found"
echo ""

# Step 1: Run normal inference
echo "================================================================================"
echo "Step 1/3: Running normal inference (baseline)"
echo "================================================================================"
echo ""

sudo docker run -it --rm --device=nvidia.com/gpu=all \
    -e XLA_FLAGS=--xla_gpu_strict_conv_algorithm_picker=false \
    -e TF_XLA_FLAGS=--tf_xla_enable_xla_devices=false \
    -v ~/logdir:/logdir \
    -v ${EXPERIMENT_DIR}/../experiment2/dreamerv2:/workspace/dreamerv2 \
    -v ${EXPERIMENT_DIR}:/workspace/experiment3 \
    -w /workspace/experiment3 \
    dreamerv2 \
    python3 run_inference_normal.py

echo ""
echo "✓ Normal inference complete"
echo ""

# Step 2: Run inverted inference
echo "================================================================================"
echo "Step 2/3: Running inverted inference (background test)"
echo "================================================================================"
echo ""

sudo docker run -it --rm --device=nvidia.com/gpu=all \
    -e XLA_FLAGS=--xla_gpu_strict_conv_algorithm_picker=false \
    -e TF_XLA_FLAGS=--tf_xla_enable_xla_devices=false \
    -v ~/logdir:/logdir \
    -v ${EXPERIMENT_DIR}/../experiment2/dreamerv2:/workspace/dreamerv2 \
    -v ${EXPERIMENT_DIR}:/workspace/experiment3 \
    -w /workspace/experiment3 \
    dreamerv2 \
    python3 run_inference_inverted.py

echo ""
echo "✓ Inverted inference complete"
echo ""

# Step 3: Create comparison plots
echo "================================================================================"
echo "Step 3/3: Creating comparison plots"
echo "================================================================================"
echo ""

cd ${EXPERIMENT_DIR}
python3 compare_results.py

echo ""
echo "================================================================================"
echo "EXPERIMENT 3 COMPLETE"
echo "================================================================================"
echo ""
echo "Results saved to:"
echo "  - ${EXPERIMENT_DIR}/reward_comparison.png"
echo "  - ${EXPERIMENT_DIR}/reward_histogram.png"
echo "  - ${EXPERIMENT_DIR}/comparison_summary.txt"
echo ""
echo "To view results:"
echo "  cat ${EXPERIMENT_DIR}/comparison_summary.txt"
echo ""
