"""Compare Aligned vs Sophisticated Inference (Pitliya et al., 2025).

Runs both simulation types across 4 climate scenarios and prints a
comparison table showing cost, comfort, battery utilisation, and GHG.

The sophisticated battery maintains a phantom thermostat that predicts
HVAC activity T steps ahead, building step-dependent B[energy] matrices
for EFE computation — enabling anticipatory cost management.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np

from econet.climate import generate_climate_week
from econet.simulation import run_simulation, run_sophisticated_simulation
from econet.metrics import compute_metrics

SCENARIOS = [
    ("london", "summer"),
    ("london", "winter"),
    ("montreal", "winter"),
    ("phoenix", "summer"),
]

NUM_DAYS = 7


def main():
    print("=" * 100)
    print("Sophisticated Inference Comparison (Pitliya et al., 2025)")
    print(f"  Scenarios: {len(SCENARIOS)}, Days: {NUM_DAYS}")
    print("=" * 100)

    # Generate environment data
    scenarios = {}
    for city, season in SCENARIOS:
        key = f"{city}_{season}"
        try:
            from econet.real_weather import build_real_weather_env
            env_data = build_real_weather_env(city, season, use_cache=True)
        except Exception:
            env_data = None
        if env_data is None:
            env_data = generate_climate_week(city, season, seed=42)
        scenarios[key] = env_data
        temps = env_data["outdoor_temp"]
        print(f"  {key}: T_out = [{temps.min():.1f}, {temps.max():.1f}]C")

    keys = list(scenarios.keys())

    # --- Aligned baseline ---
    print("\n--- Running Aligned baseline ---")
    aligned_metrics = {}
    for key, env_data in scenarios.items():
        print(f"  Aligned: {key}...", end=" ", flush=True)
        result = run_simulation(
            env_data=env_data, num_days=NUM_DAYS,
            policy_len=4, gamma=16.0,
            learn_B=True, aligned=True, seed=42, verbose=False,
        )
        aligned_metrics[key] = compute_metrics(result, num_days=NUM_DAYS)
        ma = aligned_metrics[key]
        print(f"Cost: ${ma.total_cost:.2f}, Comfort: {ma.comfort_deviation_total:.1f}, "
              f"Batt: {ma.battery_utilization*100:.1f}%")

    # --- Sophisticated inference ---
    print("\n--- Running Sophisticated inference ---")
    soph_metrics = {}
    for key, env_data in scenarios.items():
        print(f"  Sophisticated: {key}...", end=" ", flush=True)
        result = run_sophisticated_simulation(
            env_data=env_data, num_days=NUM_DAYS,
            policy_len=4, gamma=16.0,
            learn_B=True, initial_room_temp=20.0, initial_soc=0.5,
            seed=42, verbose=False,
        )
        soph_metrics[key] = compute_metrics(result, num_days=NUM_DAYS)
        ms = soph_metrics[key]
        print(f"Cost: ${ms.total_cost:.2f}, Comfort: {ms.comfort_deviation_total:.1f}, "
              f"Batt: {ms.battery_utilization*100:.1f}%")

        # Phantom prediction diagnostic
        if hasattr(result, 'phantom_history') and result.phantom_history:
            p_hvacs = [p for p in result.phantom_history if p is not None]
            if p_hvacs:
                p_active = [p[0] + p[1] for p in p_hvacs]
                print(f"    Phantom P(active): mean={np.mean(p_active):.3f}, "
                      f"std={np.std(p_active):.3f}")

    # --- Summary table ---
    print("\n")
    print("=" * 90)
    print(f"{'Scenario':<24} {'Metric':<16} {'Aligned':>12} {'Sophisticated':>14} {'Delta':>10}")
    print("-" * 90)

    for key in keys:
        ma = aligned_metrics[key]
        ms = soph_metrics[key]
        for metric_name, get_val, get_delta in [
            ("Comfort",
             lambda m: f"{m.comfort_deviation_total:.1f}",
             lambda ma, ms: ms.comfort_deviation_total - ma.comfort_deviation_total),
            ("Cost ($)",
             lambda m: f"{m.total_cost:.2f}",
             lambda ma, ms: ms.total_cost - ma.total_cost),
            ("Batt util %",
             lambda m: f"{m.battery_utilization*100:.1f}",
             lambda ma, ms: (ms.battery_utilization - ma.battery_utilization) * 100),
            ("GHG (kg)",
             lambda m: f"{m.total_ghg:.2f}",
             lambda ma, ms: ms.total_ghg - ma.total_ghg),
        ]:
            delta = get_delta(ma, ms)
            if metric_name == "Comfort":
                row = f"{key:<24} {metric_name:.<16} {get_val(ma):>12} {get_val(ms):>14} {delta:>+10.2f}"
            else:
                row = f"{'':<24} {metric_name:.<16} {get_val(ma):>12} {get_val(ms):>14} {delta:>+10.2f}"
            print(row)
        print()

    # --- Delta summary ---
    print("=" * 90)
    print("DELTA vs ALIGNED (negative = Sophisticated better)")
    print("-" * 90)
    for key in keys:
        ma = aligned_metrics[key]
        ms = soph_metrics[key]
        cd = ms.total_cost - ma.total_cost
        fd = ms.comfort_deviation_total - ma.comfort_deviation_total
        bd = ms.battery_utilization - ma.battery_utilization
        gd = ms.total_ghg - ma.total_ghg
        print(f"  {key:<24} cost: ${cd:+.2f}  comfort: {fd:+.1f}  "
              f"batt: {bd*100:+.1f}%  ghg: {gd:+.2f}kg")
    print("=" * 90)


if __name__ == "__main__":
    main()
