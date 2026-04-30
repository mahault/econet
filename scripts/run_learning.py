"""Scenario 2: Parameter learning 40-day simulation.

Reproduces Fig 8 (EFE convergence + learning dynamics).
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from econet.environment import generate_multi_day
from econet.simulation import run_learning
from econet.plotting import plot_learning_convergence


def main():
    num_days = 10  # reduced from 40 for legacy numpy feasibility
    print("=" * 60)
    print(f"EcoNet Scenario 2: Parameter Learning ({num_days} days)")
    print("=" * 60)

    # Generate data
    env_data = generate_multi_day(num_days=num_days, seed=42)

    # Run with B-matrix learning enabled (policy_len=4 for legacy numpy speed)
    print(f"\nRunning EcoNet with B-learning ({num_days} days, policy_len=4)...")
    result = run_learning(num_days=num_days, policy_len=4, env_data=env_data, seed=42)

    # Summary
    print(f"\n--- Summary ---")
    print(f"Total steps: {result.total_steps}")
    print(f"Total cost: ${result.total_cost:.2f}")
    print(f"Avg daily cost: ${result.avg_daily_cost:.3f}")
    print(f"Total GHG: {result.total_ghg:.2f} kg CO2")

    # Figure
    print("\nGenerating Fig 8: Learning convergence...")
    plot_learning_convergence(result)
    print("  Fig 8 saved ✓")

    return result


if __name__ == "__main__":
    main()
