"""Scenario 1: Deterministic 2-day simulation.

Reproduces the core EcoNet paper results (Figs 2-7).
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from econet.environment import generate_multi_day
from econet.simulation import run_deterministic
from econet.plotting import (
    plot_input_data, plot_efe_landscape, plot_temperature_control,
    plot_battery_soc, plot_energy_breakdown, plot_ghg_vs_energy,
)


def main():
    print("=" * 60)
    print("EcoNet Scenario 1: Deterministic 2-Day Simulation")
    print("=" * 60)

    # Generate synthetic data
    env_data = generate_multi_day(num_days=2, seed=42)

    # Run simulation
    print("\nRunning EcoNet (policy_len=4, gamma=16)...")
    result = run_deterministic(num_days=2, policy_len=4, env_data=env_data, seed=42)

    # Print summary
    print(f"\n--- Summary ---")
    print(f"Total steps: {result.total_steps}")
    print(f"Total cost: ${result.total_cost:.2f}")
    print(f"Total GHG: {result.total_ghg:.2f} kg CO2")
    print(f"Cumulative temp deviation: {result.cumulative_temp_deviation:.1f}°C")
    print(f"Comfort violation hours: {result.comfort_violation_hours}")

    # Generate figures
    print("\nGenerating figures...")
    plot_input_data(env_data)
    print("  Fig 2: Input data ✓")

    plot_efe_landscape(result, agent="thermostat")
    print("  Fig 3: EFE landscape ✓")

    plot_temperature_control(result)
    print("  Fig 4: Temperature control ✓")

    plot_battery_soc(result)
    print("  Fig 5: Battery SoC ✓")

    plot_energy_breakdown(result)
    print("  Fig 6: Energy breakdown ✓")

    plot_ghg_vs_energy(result)
    print("  Fig 7: GHG vs energy ✓")

    print("\nAll figures saved to figures/ directory.")
    return result


if __name__ == "__main__":
    main()
