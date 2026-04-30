"""Scenario 3: Baseline comparisons.

Runs EcoNet, No-HEMS, Rule-Based, and Oracle baselines.
Generates Fig 9 (comparison bar chart).
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from econet.environment import generate_multi_day
from econet.simulation import run_simulation
from econet.baselines import run_no_hems, run_rule_based, run_oracle
from econet.metrics import compute_metrics, format_metrics_table
from econet.plotting import plot_baseline_comparison


def main():
    print("=" * 60)
    print("EcoNet Scenario 3: Baseline Comparisons (2 days)")
    print("=" * 60)

    env_data = generate_multi_day(num_days=2, seed=42)

    # Run all methods
    print("\nRunning No-HEMS baseline...")
    no_hems = run_no_hems(env_data)

    print("Running Rule-Based baseline...")
    rule_based = run_rule_based(env_data)

    print("Running Oracle baseline...")
    oracle = run_oracle(env_data)

    print("Running EcoNet...")
    econet = run_simulation(env_data=env_data, num_days=2, seed=42, verbose=False)

    # Compute metrics
    m_no_hems = compute_metrics(no_hems, num_days=2)
    m_rule = compute_metrics(rule_based, num_days=2, no_hems_result=no_hems)
    m_oracle = compute_metrics(oracle, num_days=2, no_hems_result=no_hems)
    m_econet = compute_metrics(econet, num_days=2, no_hems_result=no_hems)

    metrics_dict = {
        "No-HEMS": m_no_hems,
        "Rule-Based": m_rule,
        "Oracle": m_oracle,
        "EcoNet": m_econet,
    }

    # Print table
    print("\n" + format_metrics_table(metrics_dict))

    # Savings summary
    print(f"\n--- Savings vs No-HEMS ---")
    print(f"EcoNet cost savings:     {m_econet.cost_savings_vs_no_hems:.1%}")
    print(f"Rule-Based cost savings: {m_rule.cost_savings_vs_no_hems:.1%}")
    print(f"Oracle cost savings:     {m_oracle.cost_savings_vs_no_hems:.1%}")

    # Plot
    print("\nGenerating Fig 9: Baseline comparison...")
    plot_baseline_comparison(metrics_dict)
    print("  Fig 9 saved ✓")

    return metrics_dict


if __name__ == "__main__":
    main()
