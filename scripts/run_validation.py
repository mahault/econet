"""Run full validation suite.

Verifies energy balance, thermodynamic consistency, battery constraints,
and produces comparison metrics.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from econet.validation import run_full_validation, run_sensitivity_sweep
from econet.plotting import plot_sensitivity_analysis


def main():
    print("=" * 60)
    print("EcoNet Validation Suite")
    print("=" * 60)

    # Full validation (policy_len=4 for legacy numpy speed)
    val = run_full_validation(num_days=2, seed=42, verbose=True, policy_len=4)

    # Sensitivity sweep: gamma
    print("\n\n=== Sensitivity Analysis: gamma ===")
    gamma_values = [4.0, 8.0, 16.0, 32.0, 64.0]
    gamma_results = run_sensitivity_sweep(
        "gamma", gamma_values, num_days=2, seed=42,
    )
    print("\nGenerating Fig 12: Sensitivity (gamma)...")
    plot_sensitivity_analysis(gamma_results, "gamma", gamma_values)
    print("  Fig 12 saved ✓")

    # Summary
    print("\n\n=== Validation Summary ===")
    all_pass = (val["energy_balance"]["passed"]
                and val["thermodynamics"]["passed"]
                and val["battery_constraints"]["passed"])
    print(f"Energy balance:    {'PASS' if val['energy_balance']['passed'] else 'FAIL'}")
    print(f"Thermodynamics:    {'PASS' if val['thermodynamics']['passed'] else 'FAIL'}")
    print(f"Battery:           {'PASS' if val['battery_constraints']['passed'] else 'FAIL'}")
    print(f"\nOverall: {'ALL PASS' if all_pass else 'SOME FAILURES'}")

    return val


if __name__ == "__main__":
    main()
