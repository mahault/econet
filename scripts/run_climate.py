"""Scenario 4: Multi-climate sensitivity analysis.

Runs EcoNet across 4 cities × 2 seasons = 8 scenarios.
Generates Figs 10-11.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from econet.climate import generate_all_scenarios, get_scenario_label
from econet.simulation import run_simulation
from econet.metrics import compute_metrics
from econet.plotting import plot_climate_sensitivity, plot_extended_simulation


def main():
    print("=" * 60)
    print("EcoNet Scenario 4: Multi-Climate Sensitivity (7 days each)")
    print("=" * 60)

    # Generate all climate scenarios
    scenarios = generate_all_scenarios(seed=42)

    climate_results = {}
    climate_metrics = {}

    for key, env_data in scenarios.items():
        label = get_scenario_label(key)
        print(f"\nRunning {label}...")
        result = run_simulation(
            env_data=env_data, num_days=7,
            policy_len=4, seed=42, verbose=False,
        )
        climate_results[key] = result
        climate_metrics[key] = compute_metrics(result, num_days=7)

        print(f"  Cost: ${climate_metrics[key].total_cost:.2f}, "
              f"GHG: {climate_metrics[key].total_ghg:.2f} kg, "
              f"Comfort: {climate_metrics[key].comfort_deviation_total:.1f} °C·h")

    # Plot Fig 10: Climate sensitivity
    print("\nGenerating Fig 10: Climate sensitivity...")
    plot_climate_sensitivity(climate_metrics)
    print("  Fig 10 saved ✓")

    # Plot Fig 11: Extended simulation (pick London summer as representative)
    print("Generating Fig 11: Extended 7-day (London summer)...")
    plot_extended_simulation(climate_results["london_summer"])
    print("  Fig 11 saved ✓")

    return climate_results, climate_metrics


if __name__ == "__main__":
    main()
