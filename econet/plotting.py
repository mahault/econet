"""Plotting functions for EcoNet paper figures.

Original paper figures (2-8) + new reviewer figures (9-12).
"""

import numpy as np
import matplotlib
matplotlib.use("Agg")  # headless backend for server
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from pathlib import Path
from typing import Optional

from .environment import TARGET_TEMP_OCCUPIED, TARGET_TEMP_UNOCCUPIED, STEPS_PER_DAY
from .metrics import EconomicMetrics, compute_metrics

FIGURE_DIR = Path(__file__).parent.parent / "figures"
FIGURE_DIR.mkdir(exist_ok=True)

# Style
plt.rcParams.update({
    "font.size": 11,
    "axes.labelsize": 12,
    "axes.titlesize": 13,
    "legend.fontsize": 9,
    "figure.dpi": 150,
    "savefig.dpi": 300,
})

ACTION_LABELS_HVAC = {0: "Cool", 1: "Heat", 2: "Off"}
ACTION_LABELS_BATT = {0: "Charge", 1: "Discharge", 2: "Off"}
ACTION_COLORS_HVAC = {0: "#3498db", 1: "#e74c3c", 2: "#95a5a6"}
ACTION_COLORS_BATT = {0: "#2ecc71", 1: "#e67e22", 2: "#95a5a6"}


def _step_axis(n_steps, n_days):
    """Create x-axis: step indices and day boundary markers."""
    x = np.arange(n_steps)
    day_boundaries = np.arange(0, n_steps + 1, STEPS_PER_DAY)
    return x, day_boundaries


# =========================================================================
# Fig 2: Input data (outdoor temp + baseline load)
# =========================================================================

def plot_input_data(env_data: dict, save: bool = True) -> plt.Figure:
    """Fig 2: Environment input data overview."""
    n = len(env_data["time_of_day"])
    n_days = n // STEPS_PER_DAY
    x, day_bd = _step_axis(n, n_days)

    fig, axes = plt.subplots(4, 1, figsize=(10, 8), sharex=True)

    # Outdoor temperature
    axes[0].plot(x, env_data["outdoor_temp"], "b-o", ms=3, label="Outdoor Temp")
    axes[0].set_ylabel("°C")
    axes[0].set_title("Outdoor Temperature")
    axes[0].legend()

    # Solar generation
    axes[1].fill_between(x, 0, env_data["solar_gen"], alpha=0.5, color="orange")
    axes[1].plot(x, env_data["solar_gen"], "o-", ms=3, color="darkorange")
    axes[1].set_ylabel("kWh")
    axes[1].set_title("Solar Generation")

    # Baseline load
    axes[2].plot(x, env_data["baseline_load"], "g-o", ms=3, label="Baseline Load")
    axes[2].set_ylabel("kWh")
    axes[2].set_title("Baseline Household Load")
    axes[2].legend()

    # TOU rate + occupancy
    ax3 = axes[3]
    ax3.fill_between(x, 0, env_data["tou_high"], alpha=0.3, color="red",
                     step="mid", label="High TOU")
    ax3_twin = ax3.twinx()
    ax3_twin.step(x, env_data["occupancy"], "k--", alpha=0.7, where="mid",
                  label="Occupancy")
    ax3.set_ylabel("TOU (high=1)")
    ax3_twin.set_ylabel("Occupancy")
    ax3.set_title("TOU Periods & Occupancy")
    ax3.set_xlabel("Time Step (2h intervals)")

    # Day boundaries
    for ax in axes:
        for bd in day_bd:
            ax.axvline(bd, color="gray", ls=":", alpha=0.3)
        ax.grid(True, alpha=0.3)

    fig.tight_layout()
    if save:
        fig.savefig(FIGURE_DIR / "fig2_input_data.pdf")
        fig.savefig(FIGURE_DIR / "fig2_input_data.png")
    return fig


# =========================================================================
# Fig 3: EFE distribution over policies
# =========================================================================

def plot_efe_landscape(result, agent: str = "thermostat",
                       max_steps: int = 24, save: bool = True) -> plt.Figure:
    """Fig 3: Negative EFE (G) distribution across policies per step."""
    if agent == "thermostat":
        efe_hist = result.thermo_efe_history
        title = "Thermostat Agent: Negative EFE (G) per Policy"
    else:
        efe_hist = result.battery_efe_history
        title = "Battery Agent: Negative EFE (G) per Policy"

    n_steps = min(len(efe_hist), max_steps)
    fig, ax = plt.subplots(figsize=(12, 5))

    # Build matrix: (n_steps, n_policies)
    efe_0 = np.asarray(efe_hist[0]).flatten()
    n_policies = len(efe_0)
    efe_matrix = np.zeros((n_steps, n_policies))
    for t in range(n_steps):
        efe_matrix[t] = np.asarray(efe_hist[t]).flatten()[:n_policies]

    im = ax.imshow(efe_matrix.T, aspect="auto", cmap="viridis",
                   origin="lower")
    ax.set_xlabel("Time Step")
    ax.set_ylabel("Policy Index")
    ax.set_title(title)
    plt.colorbar(im, ax=ax, label="Negative EFE (G)")

    fig.tight_layout()
    if save:
        fig.savefig(FIGURE_DIR / f"fig3_efe_{agent}.pdf")
        fig.savefig(FIGURE_DIR / f"fig3_efe_{agent}.png")
    return fig


# =========================================================================
# Fig 4: Temperature control
# =========================================================================

def plot_temperature_control(result, save: bool = True) -> plt.Figure:
    """Fig 4: Room temp, target, delta, and HVAC actions."""
    arr = result.to_arrays()
    n = len(arr["step"])
    n_days = result.num_days
    x, day_bd = _step_axis(n, n_days)

    fig, axes = plt.subplots(3, 1, figsize=(10, 7), sharex=True)

    # Room temp vs target
    axes[0].plot(x, arr["room_temp"], "r-o", ms=3, label="Room Temp")
    # Show target as step function
    target = np.where(arr["occupancy"], TARGET_TEMP_OCCUPIED,
                      TARGET_TEMP_UNOCCUPIED)
    axes[0].plot(x, target, "k--", alpha=0.7, label="Target")
    axes[0].fill_between(x, target - 2, target + 2, alpha=0.1, color="green",
                         label="Comfort Band (±2°C)")
    axes[0].set_ylabel("°C")
    axes[0].set_title("Room Temperature Control")
    axes[0].legend(loc="upper right")

    # Delta from target
    delta = arr["room_temp"] - target
    colors = ["green" if abs(d) <= 2 else "red" for d in delta]
    axes[1].bar(x, delta, color=colors, alpha=0.7, width=0.8)
    axes[1].axhline(0, color="k", ls="-", lw=0.5)
    axes[1].axhline(2, color="r", ls="--", alpha=0.5)
    axes[1].axhline(-2, color="r", ls="--", alpha=0.5)
    axes[1].set_ylabel("ΔT (°C)")
    axes[1].set_title("Temperature Deviation from Target")

    # HVAC actions
    for a, label in ACTION_LABELS_HVAC.items():
        mask = arr["hvac_action"] == a
        if mask.any():
            axes[2].scatter(x[mask], np.ones(mask.sum()) * a, c=ACTION_COLORS_HVAC[a],
                          s=50, label=label, zorder=3)
    axes[2].set_yticks([0, 1, 2])
    axes[2].set_yticklabels(["Cool", "Heat", "Off"])
    axes[2].set_ylabel("Action")
    axes[2].set_title("HVAC Actions")
    axes[2].set_xlabel("Time Step (2h intervals)")
    axes[2].legend(loc="upper right")

    for ax in axes:
        for bd in day_bd:
            ax.axvline(bd, color="gray", ls=":", alpha=0.3)
        ax.grid(True, alpha=0.3)

    fig.tight_layout()
    if save:
        fig.savefig(FIGURE_DIR / "fig4_temperature_control.pdf")
        fig.savefig(FIGURE_DIR / "fig4_temperature_control.png")
    return fig


# =========================================================================
# Fig 5: Battery SoC vs TOU
# =========================================================================

def plot_battery_soc(result, save: bool = True) -> plt.Figure:
    """Fig 5: Battery state of charge vs TOU periods."""
    arr = result.to_arrays()
    n = len(arr["step"])
    n_days = result.num_days
    x, day_bd = _step_axis(n, n_days)

    fig, axes = plt.subplots(2, 1, figsize=(10, 5), sharex=True)

    # SoC with TOU shading
    axes[0].plot(x, arr["soc"], "b-o", ms=4, lw=2, label="SoC")
    # Shade high-TOU periods
    for i in range(n):
        if arr["tou_high"][i]:
            axes[0].axvspan(i - 0.5, i + 0.5, alpha=0.15, color="red")
    axes[0].set_ylabel("State of Charge")
    axes[0].set_title("Battery SoC vs TOU Periods (red = high TOU)")
    axes[0].set_ylim(-0.05, 1.05)
    axes[0].legend()

    # Battery actions
    for a, label in ACTION_LABELS_BATT.items():
        mask = arr["battery_action"] == a
        if mask.any():
            axes[1].scatter(x[mask], np.ones(mask.sum()) * a,
                          c=ACTION_COLORS_BATT[a], s=50, label=label, zorder=3)
    axes[1].set_yticks([0, 1, 2])
    axes[1].set_yticklabels(["Charge", "Discharge", "Off"])
    axes[1].set_ylabel("Action")
    axes[1].set_title("Battery Actions")
    axes[1].set_xlabel("Time Step (2h intervals)")
    axes[1].legend(loc="upper right")

    for ax in axes:
        for bd in day_bd:
            ax.axvline(bd, color="gray", ls=":", alpha=0.3)
        ax.grid(True, alpha=0.3)

    fig.tight_layout()
    if save:
        fig.savefig(FIGURE_DIR / "fig5_battery_soc.pdf")
        fig.savefig(FIGURE_DIR / "fig5_battery_soc.png")
    return fig


# =========================================================================
# Fig 6: Total energy, battery, solar
# =========================================================================

def plot_energy_breakdown(result, save: bool = True) -> plt.Figure:
    """Fig 6: Energy breakdown — baseline, HVAC, battery, solar, total."""
    arr = result.to_arrays()
    n = len(arr["step"])
    x, day_bd = _step_axis(n, result.num_days)

    fig, ax = plt.subplots(figsize=(10, 4))

    ax.plot(x, arr["baseline_load"], "g-", alpha=0.7, label="Baseline Load")
    ax.plot(x, arr["hvac_energy"], "r-", alpha=0.7, label="HVAC Energy")
    ax.plot(x, arr["battery_energy"], "b-", alpha=0.7, label="Battery (+ = grid charge)")
    ax.plot(x, -arr["solar_gen"], "orange", alpha=0.7, label="Solar (offset)")
    ax.plot(x, arr["total_energy"], "k-", lw=2, label="Net Grid Energy")
    ax.axhline(0, color="gray", ls="-", lw=0.5)

    ax.set_xlabel("Time Step (2h intervals)")
    ax.set_ylabel("Energy (kWh)")
    ax.set_title("Energy Breakdown per Time Step")
    ax.legend(loc="upper right", ncol=2)
    ax.grid(True, alpha=0.3)

    for bd in day_bd:
        ax.axvline(bd, color="gray", ls=":", alpha=0.3)

    fig.tight_layout()
    if save:
        fig.savefig(FIGURE_DIR / "fig6_energy_breakdown.pdf")
        fig.savefig(FIGURE_DIR / "fig6_energy_breakdown.png")
    return fig


# =========================================================================
# Fig 7: GHG emissions vs energy use
# =========================================================================

def plot_ghg_vs_energy(result, save: bool = True) -> plt.Figure:
    """Fig 7: GHG emissions vs energy use scatter."""
    arr = result.to_arrays()

    fig, axes = plt.subplots(1, 2, figsize=(10, 4))

    # Time series
    n = len(arr["step"])
    x = np.arange(n)
    axes[0].plot(x, arr["ghg"], "g-o", ms=3, label="GHG")
    ax2 = axes[0].twinx()
    ax2.plot(x, arr["total_energy"], "b-s", ms=3, alpha=0.6, label="Energy")
    axes[0].set_xlabel("Time Step")
    axes[0].set_ylabel("GHG (kg CO2)", color="green")
    ax2.set_ylabel("Energy (kWh)", color="blue")
    axes[0].set_title("GHG & Energy over Time")

    # Scatter
    axes[1].scatter(arr["total_energy"], arr["ghg"], c=arr["tou_high"],
                   cmap="coolwarm", alpha=0.7, edgecolors="k", lw=0.5)
    axes[1].set_xlabel("Net Energy (kWh)")
    axes[1].set_ylabel("GHG (kg CO2)")
    axes[1].set_title("GHG vs Energy (red = high TOU)")
    axes[1].grid(True, alpha=0.3)

    fig.tight_layout()
    if save:
        fig.savefig(FIGURE_DIR / "fig7_ghg_energy.pdf")
        fig.savefig(FIGURE_DIR / "fig7_ghg_energy.png")
    return fig


# =========================================================================
# Fig 8: Parameter learning convergence
# =========================================================================

def plot_learning_convergence(result, save: bool = True) -> plt.Figure:
    """Fig 8: EFE convergence over multi-day learning simulation."""
    if not result.thermo_efe_history:
        return None

    n = len(result.thermo_efe_history)
    # Mean absolute EFE per step
    mean_efe = []
    for efe in result.thermo_efe_history:
        mean_efe.append(float(np.mean(np.abs(efe))))

    fig, axes = plt.subplots(2, 1, figsize=(10, 5))

    # EFE trajectory
    axes[0].plot(mean_efe, "b-", alpha=0.7)
    # Rolling average
    window = min(STEPS_PER_DAY, n)
    if n > window:
        rolling = np.convolve(mean_efe, np.ones(window) / window, mode="valid")
        axes[0].plot(np.arange(window - 1, n), rolling, "r-", lw=2,
                    label=f"Rolling mean ({window} steps)")
    axes[0].set_ylabel("Mean |EFE|")
    axes[0].set_title("Thermostat EFE Convergence over Learning")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    # Cumulative temperature deviation per day
    arr = result.to_arrays()
    target = np.where(arr["occupancy"], TARGET_TEMP_OCCUPIED,
                      TARGET_TEMP_UNOCCUPIED)
    daily_dev = []
    for d in range(result.num_days):
        start = d * STEPS_PER_DAY
        end = start + STEPS_PER_DAY
        if end <= len(arr["room_temp"]):
            dev = np.sum(np.abs(arr["room_temp"][start:end] - target[start:end]))
            daily_dev.append(dev)

    axes[1].bar(range(len(daily_dev)), daily_dev, color="coral", alpha=0.7)
    axes[1].set_xlabel("Day")
    axes[1].set_ylabel("Cumulative |ΔT| (°C)")
    axes[1].set_title("Daily Temperature Deviation (should decrease)")
    axes[1].grid(True, alpha=0.3)

    fig.tight_layout()
    if save:
        fig.savefig(FIGURE_DIR / "fig8_learning.pdf")
        fig.savefig(FIGURE_DIR / "fig8_learning.png")
    return fig


# =========================================================================
# Fig 9 [NEW]: Baseline comparison bar chart
# =========================================================================

def plot_baseline_comparison(metrics_dict: dict, save: bool = True) -> plt.Figure:
    """Fig 9: Bar chart comparing EcoNet vs baselines on key metrics."""
    labels = list(metrics_dict.keys())
    n = len(labels)

    metrics_to_plot = [
        ("Total Cost ($)", "total_cost"),
        ("Total GHG (kg CO₂)", "total_ghg"),
        ("Comfort Dev (°C·h)", "comfort_deviation_total"),
        ("Violation Hours", "comfort_violation_hours"),
    ]

    fig, axes = plt.subplots(1, 4, figsize=(14, 4))
    colors = ["#e74c3c", "#3498db", "#2ecc71", "#9b59b6", "#f39c12"]

    for i, (title, attr) in enumerate(metrics_to_plot):
        values = [getattr(metrics_dict[l], attr, 0) for l in labels]
        bars = axes[i].bar(range(n), values, color=colors[:n], alpha=0.8)
        axes[i].set_xticks(range(n))
        axes[i].set_xticklabels(labels, rotation=30, ha="right", fontsize=9)
        axes[i].set_title(title, fontsize=10)
        axes[i].grid(True, alpha=0.3, axis="y")

        # Add value labels on bars
        for bar, val in zip(bars, values):
            axes[i].text(bar.get_x() + bar.get_width() / 2, bar.get_height(),
                        f"{val:.1f}", ha="center", va="bottom", fontsize=8)

    fig.suptitle("EcoNet vs Baselines: Key Performance Metrics", fontsize=13)
    fig.tight_layout()
    if save:
        fig.savefig(FIGURE_DIR / "fig9_baseline_comparison.pdf")
        fig.savefig(FIGURE_DIR / "fig9_baseline_comparison.png")
    return fig


# =========================================================================
# Fig 10 [NEW]: Climate sensitivity (4 panels)
# =========================================================================

def plot_climate_sensitivity(climate_results: dict, save: bool = True,
                             fig_name: str = "fig10_climate_sensitivity",
                             title_suffix: str = "") -> plt.Figure:
    """Fig 10/13: Performance across climate scenarios.

    Parameters
    ----------
    climate_results : dict
        Keys: scenario names (e.g., 'london_summer')
        Values: SimulationResult or EconomicMetrics
    fig_name : str
        Base filename for saved figure.
    title_suffix : str
        Extra text appended to suptitle.
    """
    scenarios = list(climate_results.keys())
    n = len(scenarios)

    # Extract metrics
    if hasattr(list(climate_results.values())[0], "to_arrays"):
        metrics = {k: compute_metrics(v, num_days=7)
                   for k, v in climate_results.items()}
    else:
        metrics = climate_results

    fig, axes = plt.subplots(2, 2, figsize=(12, 8))

    attrs = [
        ("avg_daily_cost", "Avg Daily Cost ($)"),
        ("total_ghg", "Total GHG (kg CO₂)"),
        ("comfort_deviation_total", "Comfort Deviation (°C·h)"),
        ("battery_utilization", "Battery Utilization"),
    ]

    for idx, (attr, title) in enumerate(attrs):
        ax = axes[idx // 2, idx % 2]
        vals = [getattr(metrics[s], attr, 0) for s in scenarios]
        labels_short = [s.replace("_real", "").replace("_", "\n")
                        for s in scenarios]

        # Color by city
        city_colors = {
            "london": "#3498db", "phoenix": "#e74c3c",
            "montreal": "#2980b9", "miami": "#e67e22",
        }
        colors = [city_colors.get(s.split("_")[0], "#95a5a6") for s in scenarios]

        bars = ax.bar(range(n), vals, color=colors, alpha=0.8)
        ax.set_xticks(range(n))
        ax.set_xticklabels(labels_short, fontsize=8)
        ax.set_title(title)
        ax.grid(True, alpha=0.3, axis="y")

    fig.suptitle("EcoNet Performance Across Climate Scenarios" + title_suffix,
                 fontsize=14)
    fig.tight_layout()
    if save:
        fig.savefig(FIGURE_DIR / f"{fig_name}.pdf")
        fig.savefig(FIGURE_DIR / f"{fig_name}.png")
    return fig


# =========================================================================
# Fig 11 [NEW]: Extended 7-day simulation
# =========================================================================

def plot_extended_simulation(result, save: bool = True) -> plt.Figure:
    """Fig 11: 7-day simulation overview."""
    arr = result.to_arrays()
    n = len(arr["step"])
    x = np.arange(n)
    day_bd = np.arange(0, n + 1, STEPS_PER_DAY)

    fig, axes = plt.subplots(4, 1, figsize=(14, 10), sharex=True)

    # Room temp
    target = np.where(arr["occupancy"], TARGET_TEMP_OCCUPIED,
                      TARGET_TEMP_UNOCCUPIED)
    axes[0].plot(x, arr["room_temp"], "r-", lw=1, label="Room")
    axes[0].plot(x, arr["outdoor_temp"], "b-", lw=1, alpha=0.5, label="Outdoor")
    axes[0].plot(x, target, "k--", alpha=0.7, label="Target")
    axes[0].set_ylabel("°C")
    axes[0].set_title("Temperature over 7 Days")
    axes[0].legend(loc="upper right", ncol=3)

    # SoC
    axes[1].plot(x, arr["soc"], "b-", lw=1.5)
    for i in range(n):
        if arr["tou_high"][i]:
            axes[1].axvspan(i - 0.5, i + 0.5, alpha=0.1, color="red")
    axes[1].set_ylabel("SoC")
    axes[1].set_title("Battery State of Charge")

    # Cost per step
    axes[2].bar(x, arr["cost"], color="green", alpha=0.7, width=0.8)
    axes[2].set_ylabel("Cost ($)")
    axes[2].set_title("Step Cost")

    # Cumulative cost
    axes[3].plot(x, np.cumsum(arr["cost"]), "k-", lw=2)
    axes[3].set_ylabel("Cumulative ($)")
    axes[3].set_xlabel("Time Step (2h intervals)")
    axes[3].set_title("Cumulative Cost")

    for ax in axes:
        for bd in day_bd:
            ax.axvline(bd, color="gray", ls=":", alpha=0.3)
        ax.grid(True, alpha=0.3)

    fig.tight_layout()
    if save:
        fig.savefig(FIGURE_DIR / "fig11_extended_7day.pdf")
        fig.savefig(FIGURE_DIR / "fig11_extended_7day.png")
    return fig


# =========================================================================
# Fig 12 [NEW]: Sensitivity analysis
# =========================================================================

def plot_sensitivity_analysis(results_dict: dict, param_name: str,
                              param_values: list, save: bool = True) -> plt.Figure:
    """Fig 12: Sensitivity analysis — vary one parameter, plot KPIs.

    Parameters
    ----------
    results_dict : dict[str, SimulationResult]
        One result per parameter value.
    param_name : str
        Name of the varied parameter (for labeling).
    param_values : list
        Corresponding parameter values.
    """
    n = len(param_values)
    costs = []
    ghgs = []
    comforts = []

    for key in results_dict:
        r = results_dict[key]
        arr = r.to_arrays()
        costs.append(float(arr["cost"].sum()))
        ghgs.append(float(arr["ghg"].sum()))
        target = np.where(arr["occupancy"], TARGET_TEMP_OCCUPIED,
                          TARGET_TEMP_UNOCCUPIED)
        comforts.append(float(np.abs(arr["room_temp"] - target).sum()))

    fig, axes = plt.subplots(1, 3, figsize=(12, 4))

    axes[0].plot(param_values, costs, "bo-", lw=2)
    axes[0].set_xlabel(param_name)
    axes[0].set_ylabel("Total Cost ($)")
    axes[0].set_title("Cost Sensitivity")
    axes[0].grid(True, alpha=0.3)

    axes[1].plot(param_values, ghgs, "gs-", lw=2)
    axes[1].set_xlabel(param_name)
    axes[1].set_ylabel("Total GHG (kg CO₂)")
    axes[1].set_title("GHG Sensitivity")
    axes[1].grid(True, alpha=0.3)

    axes[2].plot(param_values, comforts, "r^-", lw=2)
    axes[2].set_xlabel(param_name)
    axes[2].set_ylabel("Comfort Deviation (°C)")
    axes[2].set_title("Comfort Sensitivity")
    axes[2].grid(True, alpha=0.3)

    fig.suptitle(f"Sensitivity Analysis: {param_name}", fontsize=13)
    fig.tight_layout()
    if save:
        fig.savefig(FIGURE_DIR / f"fig12_sensitivity_{param_name}.pdf")
        fig.savefig(FIGURE_DIR / f"fig12_sensitivity_{param_name}.png")
    return fig
