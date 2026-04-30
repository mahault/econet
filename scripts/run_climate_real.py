"""Scenario 5: Real weather data climate sensitivity.

Fetches actual hourly weather observations from IEM ASOS for 4 cities
(London, Phoenix, Montreal, Miami) across summer + winter weeks.
Runs both Aligned and ToM+Belief modes on real data.
Generates Fig 13 (real-data climate sensitivity) and Fig 14 (extended real).
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from econet.real_weather import generate_all_real_scenarios, get_real_scenario_label
from econet.simulation import run_simulation, run_tom_simulation
from econet.metrics import compute_metrics, compute_communication_metrics
from econet.plotting import FIGURE_DIR
from econet.environment import (
    TARGET_TEMP_OCCUPIED, TARGET_TEMP_UNOCCUPIED, STEPS_PER_DAY,
)


def main():
    print("=" * 70)
    print("EcoNet Scenario 5: Real Weather Data (4 cities x 2 seasons)")
    print("  Modes: Aligned + ToM+Belief")
    print("=" * 70)

    # Download and build real weather scenarios
    print("\nFetching real weather data from IEM ASOS...")
    scenarios = generate_all_real_scenarios(use_cache=True)
    print(f"  Got {len(scenarios)} scenarios")

    aligned_results = {}
    aligned_metrics = {}
    tom_results = {}
    tom_metrics = {}

    for key, env_data in scenarios.items():
        label = get_real_scenario_label(key)

        # Aligned mode
        print(f"\nRunning Aligned: {label}...")
        a_result = run_simulation(
            env_data=env_data, num_days=7,
            policy_len=4, gamma=16.0,
            learn_B=True, aligned=True, seed=42, verbose=False,
        )
        aligned_results[key] = a_result
        aligned_metrics[key] = compute_metrics(a_result, num_days=7)
        ma = aligned_metrics[key]
        print(f"  Aligned  -> Cost: ${ma.total_cost:.2f}, "
              f"GHG: {ma.total_ghg:.2f} kg, "
              f"Comfort: {ma.comfort_deviation_total:.1f} deg-h")

        # ToM+Belief mode
        print(f"  Running ToM+Belief: {label}...")
        t_result = run_tom_simulation(
            env_data=env_data, num_days=7,
            policy_len=4, gamma=16.0,
            learn_B=True, social_weight=2.0, seed=42, verbose=False,
        )
        tom_results[key] = t_result
        tom_metrics[key] = compute_metrics(t_result, num_days=7)
        mt = tom_metrics[key]
        print(f"  ToM+Bel  -> Cost: ${mt.total_cost:.2f}, "
              f"GHG: {mt.total_ghg:.2f} kg, "
              f"Comfort: {mt.comfort_deviation_total:.1f} deg-h")

    # ── Fig 13: Side-by-side climate sensitivity ─────────────────────────
    print("\nGenerating Fig 13: Real weather climate sensitivity (Aligned vs ToM+Belief)...")
    keys = list(scenarios.keys())
    n = len(keys)

    fig, axes = plt.subplots(2, 2, figsize=(14, 9))

    attrs = [
        ("avg_daily_cost", "Avg Daily Cost ($)"),
        ("total_ghg", "Total GHG (kg CO\u2082)"),
        ("comfort_deviation_total", "Comfort Deviation (\u00b0C\u00b7h)"),
        ("battery_utilization", "Battery Utilization"),
    ]

    city_colors = {
        "london": ("#3498db", "#85c1e9"),
        "phoenix": ("#e74c3c", "#f1948a"),
        "montreal": ("#2980b9", "#7fb3d8"),
        "miami": ("#e67e22", "#f0b27a"),
    }

    bar_width = 0.35
    x_pos = np.arange(n)

    for idx, (attr, title) in enumerate(attrs):
        ax = axes[idx // 2, idx % 2]
        a_vals = [getattr(aligned_metrics[k], attr, 0) for k in keys]
        t_vals = [getattr(tom_metrics[k], attr, 0) for k in keys]

        labels_short = [k.replace("_real", "").replace("_", "\n") for k in keys]
        colors_a = [city_colors.get(k.split("_")[0], ("#95a5a6", "#bdc3c7"))[0]
                     for k in keys]
        colors_t = [city_colors.get(k.split("_")[0], ("#95a5a6", "#bdc3c7"))[1]
                     for k in keys]

        bars_a = ax.bar(x_pos - bar_width / 2, a_vals, bar_width,
                        color=colors_a, alpha=0.9, label="Aligned")
        bars_t = ax.bar(x_pos + bar_width / 2, t_vals, bar_width,
                        color=colors_t, alpha=0.9, edgecolor="k",
                        linewidth=0.5, label="ToM+Belief")

        ax.set_xticks(x_pos)
        ax.set_xticklabels(labels_short, fontsize=7)
        ax.set_title(title, fontsize=11)
        ax.grid(True, alpha=0.3, axis="y")
        if idx == 0:
            ax.legend(fontsize=9)

    fig.suptitle("EcoNet Climate Sensitivity: Aligned vs ToM+Belief (Real Weather Data)",
                 fontsize=13)
    fig.tight_layout()
    fig.savefig(FIGURE_DIR / "fig13_climate_real_weather.pdf", bbox_inches="tight")
    fig.savefig(FIGURE_DIR / "fig13_climate_real_weather.png", bbox_inches="tight")
    plt.close(fig)
    print("  Fig 13 saved")

    # ── Fig 14: Extended real simulation (London summer) ────────────────
    london_key = "london_summer_real"
    if london_key in tom_results:
        print("Generating Fig 14: Extended 7-day (London summer, real, ToM+Belief)...")
        result = tom_results[london_key]
        arr = result.to_arrays()
        n_steps = len(arr["step"])
        x = np.arange(n_steps)
        day_bd = np.arange(0, n_steps + 1, STEPS_PER_DAY)

        fig, axes = plt.subplots(4, 1, figsize=(14, 10), sharex=True)

        target = np.where(arr["occupancy"], TARGET_TEMP_OCCUPIED,
                          TARGET_TEMP_UNOCCUPIED)
        axes[0].plot(x, arr["room_temp"], "r-", lw=1, label="Room")
        axes[0].plot(x, arr["outdoor_temp"], "b-", lw=1, alpha=0.5, label="Outdoor")
        axes[0].plot(x, target, "k--", alpha=0.7, label="Target")
        axes[0].set_ylabel("Temperature (C)")
        axes[0].set_title("Temperature (London, Real Weather, 7 Days, ToM+Belief)")
        axes[0].legend(loc="upper right", ncol=3)

        axes[1].plot(x, arr["soc"], "b-", lw=1.5)
        for i in range(n_steps):
            if arr["tou_high"][i]:
                axes[1].axvspan(i - 0.5, i + 0.5, alpha=0.1, color="red")
        axes[1].set_ylabel("SoC")
        axes[1].set_title("Battery State of Charge vs TOU")
        axes[1].set_ylim(-0.05, 1.05)

        axes[2].plot(x, arr["cost"], "g-", lw=1)
        axes[2].set_ylabel("Cost ($)")
        axes[2].set_title("Step Cost")

        axes[3].bar(x, arr["solar_gen"], color="gold", alpha=0.7, label="Solar")
        axes[3].bar(x, -arr["baseline_load"], color="gray", alpha=0.5, label="Load")
        axes[3].set_ylabel("kWh")
        axes[3].set_title("Solar Generation vs Base Load")
        axes[3].legend()
        axes[3].set_xlabel("Step (2h intervals)")

        for ax in axes:
            for bd in day_bd:
                ax.axvline(bd, color="gray", alpha=0.3, ls="--")

        fig.tight_layout()
        fig.savefig(FIGURE_DIR / "fig14_extended_real_weather.pdf")
        fig.savefig(FIGURE_DIR / "fig14_extended_real_weather.png")
        plt.close(fig)
        print("  Fig 14 saved")

    # ── Summary table ────────────────────────────────────────────────────
    print()
    print("=" * 95)
    print(f"{'Scenario':<28} {'Mode':<12} {'Cost($)':<10} {'GHG(kg)':<10} "
          f"{'Comfort':<12} {'Batt%':<8}")
    print("-" * 95)
    for key in keys:
        label = get_real_scenario_label(key)[:26]
        ma = aligned_metrics[key]
        mt = tom_metrics[key]
        print(f"{label:<28} {'Aligned':<12} {ma.total_cost:<10.2f} {ma.total_ghg:<10.2f} "
              f"{ma.comfort_deviation_total:<12.1f} {ma.battery_utilization*100:<8.1f}")
        print(f"{'':<28} {'ToM+Belief':<12} {mt.total_cost:<10.2f} {mt.total_ghg:<10.2f} "
              f"{mt.comfort_deviation_total:<12.1f} {mt.battery_utilization*100:<8.1f}")
    print("=" * 95)

    # ── Aggregate comparison ─────────────────────────────────────────────
    print()
    a_cost_avg = np.mean([aligned_metrics[k].total_cost for k in keys])
    t_cost_avg = np.mean([tom_metrics[k].total_cost for k in keys])
    a_comf_avg = np.mean([aligned_metrics[k].comfort_deviation_total for k in keys])
    t_comf_avg = np.mean([tom_metrics[k].comfort_deviation_total for k in keys])
    a_ghg_avg = np.mean([aligned_metrics[k].total_ghg for k in keys])
    t_ghg_avg = np.mean([tom_metrics[k].total_ghg for k in keys])

    print(f"AGGREGATE (mean across {len(keys)} scenarios):")
    print(f"  Aligned    -> Cost: ${a_cost_avg:.2f}, Comfort: {a_comf_avg:.1f}, GHG: {a_ghg_avg:.1f}")
    print(f"  ToM+Belief -> Cost: ${t_cost_avg:.2f}, Comfort: {t_comf_avg:.1f}, GHG: {t_ghg_avg:.1f}")
    cost_delta = (t_cost_avg - a_cost_avg) / a_cost_avg * 100
    comf_delta = (t_comf_avg - a_comf_avg) / a_comf_avg * 100
    ghg_delta = (t_ghg_avg - a_ghg_avg) / a_ghg_avg * 100
    print(f"  ToM vs Aligned: Cost {cost_delta:+.1f}%, Comfort {comf_delta:+.1f}%, GHG {ghg_delta:+.1f}%")

    return aligned_results, aligned_metrics, tom_results, tom_metrics


if __name__ == "__main__":
    main()
