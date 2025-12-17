#!/usr/bin/env python3
"""
Comparison script for experiment3.
Loads results from normal and inverted inference runs and creates visualizations.
"""
import pickle
import numpy as np
import matplotlib.pyplot as plt
import pathlib


def main():
    print("=" * 80)
    print("Experiment 3: Background Robustness Comparison")
    print("=" * 80)
    print()

    # Load results
    normal_path = pathlib.Path("experiment3_normal_results.pkl")
    inverted_path = pathlib.Path("experiment3_inverted_results.pkl")

    if not normal_path.exists():
        print(f"❌ Normal results not found at {normal_path}")
        print("Please run run_inference_normal.py first")
        return

    if not inverted_path.exists():
        print(f"❌ Inverted results not found at {inverted_path}")
        print("Please run run_inference_inverted.py first")
        return

    print("Loading results...")
    with open(normal_path, "rb") as f:
        normal_results = pickle.load(f)

    with open(inverted_path, "rb") as f:
        inverted_results = pickle.load(f)

    print("✓ Results loaded")
    print()

    # Extract data
    normal_rewards = normal_results['rewards']
    inverted_rewards = inverted_results['rewards']

    # Print statistics
    print("=" * 80)
    print("STATISTICS COMPARISON")
    print("=" * 80)
    print()
    print(f"{'Metric':<25} {'Normal':<20} {'Inverted':<20} {'Difference':<15}")
    print("-" * 80)
    print(f"{'Mean reward':<25} {normal_results['mean_reward']:>8.2f} ± {normal_results['std_reward']:>6.2f}   "
          f"{inverted_results['mean_reward']:>8.2f} ± {inverted_results['std_reward']:>6.2f}   "
          f"{normal_results['mean_reward'] - inverted_results['mean_reward']:>8.2f}")
    print(f"{'Min reward':<25} {normal_results['min_reward']:>8.1f}             "
          f"{inverted_results['min_reward']:>8.1f}             "
          f"{normal_results['min_reward'] - inverted_results['min_reward']:>8.1f}")
    print(f"{'Max reward':<25} {normal_results['max_reward']:>8.1f}             "
          f"{inverted_results['max_reward']:>8.1f}             "
          f"{normal_results['max_reward'] - inverted_results['max_reward']:>8.1f}")
    print("-" * 80)
    print()

    # Calculate performance degradation
    degradation = ((normal_results['mean_reward'] - inverted_results['mean_reward']) /
                   abs(normal_results['mean_reward']) * 100)
    print(f"Performance degradation: {degradation:.1f}%")
    print()

    # Statistical test (t-test)
    from scipy import stats
    t_stat, p_value = stats.ttest_ind(normal_rewards, inverted_rewards)
    print(f"T-test: t={t_stat:.3f}, p={p_value:.4f}")
    if p_value < 0.05:
        print("  ✓ Difference is statistically significant (p < 0.05)")
    else:
        print("  ✗ Difference is not statistically significant (p >= 0.05)")
    print()

    # Create visualizations
    print("Creating visualizations...")
    output_dir = pathlib.Path("/home/fhilprec/MBRL/MBRL/experiments/experiment3")

    # Figure 1: Box plot comparison
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # Box plot
    ax = axes[0]
    box_data = [normal_rewards, inverted_rewards]
    bp = ax.boxplot(box_data, labels=['Normal', 'Inverted'], patch_artist=True)
    bp['boxes'][0].set_facecolor('lightblue')
    bp['boxes'][1].set_facecolor('lightcoral')
    ax.set_ylabel('Episode Reward', fontsize=12)
    ax.set_title('Reward Distribution Comparison', fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3, axis='y')
    ax.axhline(y=0, color='gray', linestyle='--', linewidth=1, alpha=0.5)

    # Add mean markers
    ax.plot(1, normal_results['mean_reward'], 'D', color='darkblue', markersize=10,
            label=f"Mean: {normal_results['mean_reward']:.1f}", zorder=5)
    ax.plot(2, inverted_results['mean_reward'], 'D', color='darkred', markersize=10,
            label=f"Mean: {inverted_results['mean_reward']:.1f}", zorder=5)
    ax.legend()

    # Episode-wise comparison
    ax = axes[1]
    episodes = np.arange(1, len(normal_rewards) + 1)
    ax.plot(episodes, normal_rewards, 'o-', color='blue', label='Normal', alpha=0.7, linewidth=2)
    ax.plot(episodes, inverted_rewards, 's-', color='red', label='Inverted', alpha=0.7, linewidth=2)
    ax.axhline(y=normal_results['mean_reward'], color='blue', linestyle='--',
               linewidth=2, alpha=0.5, label=f"Normal mean: {normal_results['mean_reward']:.1f}")
    ax.axhline(y=inverted_results['mean_reward'], color='red', linestyle='--',
               linewidth=2, alpha=0.5, label=f"Inverted mean: {inverted_results['mean_reward']:.1f}")
    ax.set_xlabel('Episode', fontsize=12)
    ax.set_ylabel('Reward', fontsize=12)
    ax.set_title('Episode-wise Reward Comparison', fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3)
    ax.legend()
    ax.axhline(y=0, color='gray', linestyle='--', linewidth=1, alpha=0.5)

    plt.tight_layout()
    plot_path = output_dir / 'reward_comparison.png'
    plt.savefig(plot_path, dpi=150, bbox_inches='tight')
    print(f"✓ Saved comparison plot to {plot_path}")

    # Figure 2: Histogram comparison
    fig, ax = plt.subplots(1, 1, figsize=(10, 6))
    ax.hist(normal_rewards, bins=15, alpha=0.6, color='blue', label='Normal', edgecolor='black')
    ax.hist(inverted_rewards, bins=15, alpha=0.6, color='red', label='Inverted', edgecolor='black')
    ax.axvline(x=normal_results['mean_reward'], color='blue', linestyle='--',
               linewidth=2, label=f"Normal mean: {normal_results['mean_reward']:.1f}")
    ax.axvline(x=inverted_results['mean_reward'], color='red', linestyle='--',
               linewidth=2, label=f"Inverted mean: {inverted_results['mean_reward']:.1f}")
    ax.set_xlabel('Reward', fontsize=12)
    ax.set_ylabel('Frequency', fontsize=12)
    ax.set_title('Reward Distribution Histogram', fontsize=14, fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')

    plt.tight_layout()
    hist_path = output_dir / 'reward_histogram.png'
    plt.savefig(hist_path, dpi=150, bbox_inches='tight')
    print(f"✓ Saved histogram to {hist_path}")

    # Save summary text file
    summary_path = output_dir / 'comparison_summary.txt'
    with open(summary_path, 'w') as f:
        f.write("=" * 80 + "\n")
        f.write("Experiment 3: Background Robustness Test\n")
        f.write("=" * 80 + "\n\n")
        f.write("HYPOTHESIS:\n")
        f.write("If the model learned robust object representations, performance should be\n")
        f.write("similar between normal and inverted backgrounds. If the model relies on\n")
        f.write("background features, performance will degrade significantly.\n\n")
        f.write("=" * 80 + "\n")
        f.write("RESULTS\n")
        f.write("=" * 80 + "\n\n")
        f.write(f"Normal inference:\n")
        f.write(f"  Mean reward: {normal_results['mean_reward']:.2f} ± {normal_results['std_reward']:.2f}\n")
        f.write(f"  Range: [{normal_results['min_reward']:.1f}, {normal_results['max_reward']:.1f}]\n\n")
        f.write(f"Inverted inference:\n")
        f.write(f"  Mean reward: {inverted_results['mean_reward']:.2f} ± {inverted_results['std_reward']:.2f}\n")
        f.write(f"  Range: [{inverted_results['min_reward']:.1f}, {inverted_results['max_reward']:.1f}]\n\n")
        f.write(f"Performance degradation: {degradation:.1f}%\n")
        f.write(f"Statistical significance: t={t_stat:.3f}, p={p_value:.4f}\n\n")
        f.write("=" * 80 + "\n")
        f.write("INTERPRETATION\n")
        f.write("=" * 80 + "\n\n")
        if abs(degradation) < 10:
            f.write("The model appears ROBUST to background changes (<10% degradation).\n")
            f.write("This suggests the model learned object-centric representations.\n")
        elif abs(degradation) < 50:
            f.write("The model shows MODERATE sensitivity to background changes (10-50% degradation).\n")
            f.write("The model partially relies on background features.\n")
        else:
            f.write("The model shows STRONG sensitivity to background changes (>50% degradation).\n")
            f.write("This suggests the model heavily relies on background features rather than\n")
            f.write("learning robust object representations.\n")
        f.write("\n")

    print(f"✓ Saved summary to {summary_path}")
    print()
    print("=" * 80)
    print("COMPARISON COMPLETE")
    print("=" * 80)


if __name__ == "__main__":
    main()
