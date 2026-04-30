"""Economic KPIs and analysis metrics for EcoNet simulations."""

import numpy as np
from dataclasses import dataclass
from typing import Optional


@dataclass
class EconomicMetrics:
    """Summary economic metrics from a simulation run."""
    total_cost: float = 0.0          # $
    total_ghg: float = 0.0           # kg CO2
    total_energy_kwh: float = 0.0
    grid_import_kwh: float = 0.0
    solar_self_consumption: float = 0.0
    avg_daily_cost: float = 0.0
    peak_demand_kw: float = 0.0
    comfort_deviation_total: float = 0.0  # °C·hours
    comfort_violation_hours: float = 0.0
    battery_cycles: float = 0.0
    battery_utilization: float = 0.0     # fraction of steps with action != off
    cost_savings_vs_no_hems: float = 0.0
    ghg_savings_vs_no_hems: float = 0.0
    comfort_improvement_vs_no_hems: float = 0.0


def compute_metrics(result, num_days: int = 1,
                    no_hems_result=None) -> EconomicMetrics:
    """Compute economic metrics from a SimulationResult or baseline dict.

    Parameters
    ----------
    result : SimulationResult or dict
        If dict, expects keys: cost, ghg, total_energy, room_temp, target_temp,
        occupancy, battery_action, soc, solar_gen, baseline_load
    num_days : int
        Number of simulation days.
    no_hems_result : SimulationResult or dict, optional
        Baseline for computing savings.
    """
    if hasattr(result, "to_arrays"):
        arrays = result.to_arrays()
    else:
        arrays = result

    m = EconomicMetrics()
    m.total_cost = float(arrays["cost"].sum())
    m.total_ghg = float(arrays["ghg"].sum())
    m.total_energy_kwh = float(arrays["total_energy"].sum())
    m.grid_import_kwh = float(np.maximum(arrays["total_energy"], 0).sum())
    m.avg_daily_cost = m.total_cost / max(num_days, 1)
    m.peak_demand_kw = float(np.maximum(arrays["total_energy"], 0).max()) / 2.0

    # Solar self-consumption
    solar_total = float(arrays["solar_gen"].sum())
    if solar_total > 0:
        exported = float(np.maximum(-arrays["total_energy"], 0).sum())
        m.solar_self_consumption = 1.0 - exported / solar_total
    else:
        m.solar_self_consumption = 0.0

    # Comfort
    deviations = np.abs(arrays["room_temp"] - arrays["target_temp"])
    m.comfort_deviation_total = float(deviations.sum() * 2)  # °C·hours (2h steps)

    occupied_mask = arrays["occupancy"].astype(bool)
    violations = (deviations > 2.0) & occupied_mask
    m.comfort_violation_hours = float(violations.sum() * 2)

    # Battery utilization
    if "battery_action" in arrays:
        active_battery = arrays["battery_action"] != 2
        m.battery_utilization = float(active_battery.mean())
        # Estimate cycles: each charge or discharge is 0.2 SoC change
        soc_changes = np.abs(np.diff(arrays["soc"]))
        m.battery_cycles = float(soc_changes.sum() / 2.0)  # full cycle = 1.0 SoC change

    # Savings vs no-HEMS baseline
    if no_hems_result is not None:
        if hasattr(no_hems_result, "to_arrays"):
            base = no_hems_result.to_arrays()
        else:
            base = no_hems_result

        base_cost = float(base["cost"].sum())
        base_ghg = float(base["ghg"].sum())
        base_dev = float(np.abs(base["room_temp"] - base["target_temp"]).sum() * 2)

        if base_cost > 0:
            m.cost_savings_vs_no_hems = (base_cost - m.total_cost) / base_cost
        if base_ghg > 0:
            m.ghg_savings_vs_no_hems = (base_ghg - m.total_ghg) / base_ghg
        if base_dev > 0:
            m.comfort_improvement_vs_no_hems = (
                (base_dev - m.comfort_deviation_total) / base_dev
            )

    return m


@dataclass
class CommunicationMetrics:
    """Metrics for belief sharing and ToM in ToM+Belief simulations."""
    avg_comfort_entropy: float = 0.0       # avg entropy of shared q(comfort)
    avg_soc_entropy: float = 0.0           # avg entropy of shared q(SoC)
    final_thermo_tom_reliability: float = 0.0
    final_battery_tom_reliability: float = 0.0
    avg_thermo_tom_reliability: float = 0.0
    avg_battery_tom_reliability: float = 0.0
    comfort_belief_diversity: float = 0.0  # fraction of steps where argmax != 2 (COMFY)
    soc_belief_diversity: float = 0.0      # fraction of steps where argmax != 2 (mid SoC)


def _entropy(q: np.ndarray) -> float:
    """Shannon entropy of a discrete distribution."""
    q = np.asarray(q)
    q = q[q > 0]
    return float(-np.sum(q * np.log(q + 1e-12)))


def compute_communication_metrics(result) -> CommunicationMetrics:
    """Compute communication metrics from a ToM simulation result.

    Parameters
    ----------
    result : SimulationResult
        Must have .belief_history attribute (set by run_tom_simulation).

    Returns
    -------
    CommunicationMetrics
    """
    m = CommunicationMetrics()
    bh = getattr(result, "belief_history", None)
    if bh is None:
        return m

    q_comforts = bh.get("q_comfort", [])
    q_socs = bh.get("q_soc", [])
    thermo_rels = bh.get("thermo_tom_reliability", [])
    battery_rels = bh.get("battery_tom_reliability", [])

    if q_comforts:
        entropies = [_entropy(q) for q in q_comforts]
        m.avg_comfort_entropy = float(np.mean(entropies))
        argmaxes = [int(np.argmax(q)) for q in q_comforts]
        m.comfort_belief_diversity = float(np.mean([a != 2 for a in argmaxes]))

    if q_socs:
        entropies = [_entropy(q) for q in q_socs]
        m.avg_soc_entropy = float(np.mean(entropies))
        argmaxes = [int(np.argmax(q)) for q in q_socs]
        m.soc_belief_diversity = float(np.mean([a != 2 for a in argmaxes]))

    if thermo_rels:
        m.final_thermo_tom_reliability = float(thermo_rels[-1])
        m.avg_thermo_tom_reliability = float(np.mean(thermo_rels))

    if battery_rels:
        m.final_battery_tom_reliability = float(battery_rels[-1])
        m.avg_battery_tom_reliability = float(np.mean(battery_rels))

    return m


def format_metrics_table(metrics_dict: dict) -> str:
    """Format multiple metrics as a comparison table.

    Parameters
    ----------
    metrics_dict : dict[str, EconomicMetrics]
        Mapping from label to metrics.

    Returns
    -------
    str
        Formatted table string.
    """
    labels = list(metrics_dict.keys())
    header = f"{'Metric':<30}" + "".join(f"{l:>15}" for l in labels)
    sep = "-" * len(header)

    rows = [
        ("Total Cost ($)", "total_cost", ".2f"),
        ("Total GHG (kg CO2)", "total_ghg", ".2f"),
        ("Grid Import (kWh)", "grid_import_kwh", ".1f"),
        ("Peak Demand (kW)", "peak_demand_kw", ".2f"),
        ("Comfort Dev (°C·h)", "comfort_deviation_total", ".1f"),
        ("Comfort Violations (h)", "comfort_violation_hours", ".0f"),
        ("Battery Utilization", "battery_utilization", ".1%"),
        ("Battery Cycles", "battery_cycles", ".2f"),
        ("Solar Self-Consumption", "solar_self_consumption", ".1%"),
        ("Avg Daily Cost ($)", "avg_daily_cost", ".3f"),
    ]

    lines = [header, sep]
    for row_label, attr, fmt in rows:
        line = f"{row_label:<30}"
        for label in labels:
            val = getattr(metrics_dict[label], attr, 0.0)
            line += f"{val:>15{fmt}}"
        lines.append(line)

    return "\n".join(lines)
