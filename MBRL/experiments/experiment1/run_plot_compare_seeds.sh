#!/bin/bash
source /home/florian/Object-Centric-World-Model-Reinforcement-Learning/.venv/bin/activate
python3 plot_training_compare_seeds.py \
    --seed2 "/home/florian/Object-Centric-World-Model-Reinforcement-Learning/MBRL/experiments/experiment1/seed_2_training_log" \
    --seed4 "/home/florian/Object-Centric-World-Model-Reinforcement-Learning/MBRL/experiments/experiment1/seed_4_training_log" \
    --output training_curve_seed_comparison.png
deactivate
