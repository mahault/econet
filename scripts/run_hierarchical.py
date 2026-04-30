#!/usr/bin/env python
"""Run hierarchical two-level EcoNet simulation.

Scenario: 7-day simulation with StrategyAgent selecting among 5 strategies
every 12 hours, injecting C vectors into low-level thermostat + battery agents.
Both levels learn B matrices online via Dirichlet updates.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
from econet.environment import generate_multi_day
from econet.simulation import run_hierarchical_simulation, run_simulation
from econet.baselines import run_no_hems, run_rule_based
from econet.metrics import compute_metrics


def main():
    num_days = 7
    seed = 42
    env_data = generate_multi_day(num_days=num_days, seed=seed)

    print("=" * 60)
    print(f"EcoNet Hierarchical Simulation — {num_days} days")
    print("=" * 60)

    # --- Baselines ---
    print("\n[1/4] No-HEMS baseline...")
    no_hems = run_no_hems(env_data)
    m_no = compute_metrics(no_hems, num_days)
    print(f"  Cost: ${m_no.total_cost:.2f}, "
          f"Comfort: {m_no.comfort_deviation_total:.1f} C-h")

    print("\n[2/4] Rule-based baseline...")
    rule = run_rule_based(env_data)
    m_rule = compute_metrics(rule, num_days, no_hems_result=no_hems)
    print(f"  Cost: ${m_rule.total_cost:.2f}, "
          f"Comfort: {m_rule.comfort_deviation_total:.1f} C-h, "
          f"Savings: {m_rule.cost_savings_vs_no_hems:.1%}")

    # --- Flat EcoNet ---
    print("\n[3/4] Flat EcoNet (learn_B=True)...")
    flat = run_simulation(
        env_data=env_data, num_days=num_days,
        policy_len=4, gamma=16.0, learn_B=True, verbose=True,
    )
    m_flat = compute_metrics(flat, num_days, no_hems_result=no_hems)
    print(f"  Cost: ${m_flat.total_cost:.2f}, "
          f"Comfort: {m_flat.comfort_deviation_total:.1f} C-h, "
          f"Savings: {m_flat.cost_savings_vs_no_hems:.1%}")

    # --- Hierarchical EcoNet ---
    print("\n[4/4] Hierarchical EcoNet (learn_B=True)...")
    hier = run_hierarchical_simulation(
        env_data=env_data, num_days=num_days,
        policy_len=4, gamma=16.0, learn_B=True, verbose=True,
    )
    m_hier = compute_metrics(hier, num_days, no_hems_result=no_hems)
    print(f"  Cost: ${m_hier.total_cost:.2f}, "
          f"Comfort: {m_hier.comfort_deviation_total:.1f} C-h, "
          f"Savings: {m_hier.cost_savings_vs_no_hems:.1%}")

    # --- Summary ---
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"{'Strategy':<25} {'Cost ($)':>10} {'Comfort (C-h)':>15} {'Savings':>10}")
    print("-" * 60)
    for name, m in [("No-HEMS", m_no), ("Rule-based", m_rule),
                    ("Flat EcoNet", m_flat), ("Hierarchical EcoNet", m_hier)]:
        savings = getattr(m, 'cost_savings_vs_no_hems', 0.0)
        print(f"{name:<25} {m.total_cost:>10.2f} {m.comfort_deviation_total:>15.1f} "
              f"{savings:>10.1%}")
    print("=" * 60)


if __name__ == "__main__":
    main()
