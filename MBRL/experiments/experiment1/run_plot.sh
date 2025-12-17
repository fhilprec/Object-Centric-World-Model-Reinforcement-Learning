#!/bin/bash
source /home/fhilprec/MBRL/venv_mbrl/bin/activate
python3 plot_training.py \
    --input "/home/fhilprec/MBRL/MBRL/experiments/experiment1/imagined_training_log_*" \
    --metrics /home/fhilprec/MBRL/MBRL/experiments/experiment1/metrics.jsonl \
    --output training_curve.png
deactivate