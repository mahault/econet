"""Comprehensive comparison: baselines vs decentralized AIF coordination mechanisms.

Conditions:
  BASELINES (non-decentralized / non-AIF):
    1. No-HEMS         — no control at all (lower bound)
    2. Rule-based       — industry-standard thermostat + TOU battery heuristic
    3. Oracle           — centralized DP with perfect foresight (upper bound)

  DECENTRALIZED AIF:
    4. Selfish           — both agents non-aligned, no comm, no phantom
    5. Aligned           — shared C vectors (implicit coordination)
    6. Comm only         — ToM belief sharing, non-aligned C (auditory_mode="none")
    7. Phantom only      — sophisticated inference, non-aligned thermostat
    8. Aligned + Comm    — ToM belief sharing + aligned C
    9. Aligned + Phantom — sophisticated inference + aligned thermostat
   10. Full              — aligned + phantom + belief sharing
   11. Symmetric          — both agents sophisticated (phantom of each other)

Scenarios: london_summer, london_winter, montreal_winter, phoenix_summer
Duration: 7 days per scenario
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
from collections import OrderedDict

from econet.climate import generate_climate_week
from econet.environment import (
    Environment, generate_multi_day, STEPS_PER_DAY,
    HVAC_KWH_PER_STEP, TARGET_TEMP_OCCUPIED, TARGET_TEMP_UNOCCUPIED,
)
from econet.baselines import run_no_hems, run_rule_based, run_oracle
from econet.simulation import (
    SimulationResult, run_simulation, run_tom_simulation,
    run_sophisticated_simulation, run_sophisticated_tom_simulation,
    run_full_sophisticated_simulation,
)
from econet.agents import (
    ThermostatAgent, BatteryAgent, SophisticatedBatteryAgent,
)
from econet.metrics import compute_metrics

SCENARIOS = [
    ("london", "summer"),
    ("london", "winter"),
    ("montreal", "winter"),
    ("phoenix", "summer"),
]

NUM_DAYS = 7
POLICY_LEN = 4
GAMMA = 16.0
SEED = 42


def _run_custom_aif(env_data, thermo_aligned, battery_type="standard"):
    """Run a custom AIF simulation with explicit control over alignment.

    battery_type: "standard" or "sophisticated"
    """
    total_steps = len(env_data["time_of_day"])
    env = Environment(env_data, initial_room_temp=20.0, initial_soc=0.5)
    thermo = ThermostatAgent(env_data, policy_len=POLICY_LEN, gamma=GAMMA,
                              learn_B=True, aligned=thermo_aligned)

    if battery_type == "standard":
        battery = BatteryAgent(env_data, policy_len=POLICY_LEN, gamma=GAMMA,
                                initial_soc=0.5, aligned=thermo_aligned)
    else:
        battery = SophisticatedBatteryAgent(env_data, policy_len=POLICY_LEN,
                                             gamma=GAMMA, initial_soc=0.5)

    result = SimulationResult(num_days=NUM_DAYS, policy_len=POLICY_LEN, learn_B=True)

    for step in range(total_steps):
        thermo_obs = env.get_thermostat_obs(step)
        hvac_action, hvac_energy, thermo_info = thermo.step(thermo_obs)
        actual_hvac_energy = env.apply_thermostat(hvac_action, step)
        battery_obs = env.get_battery_obs(step, actual_hvac_energy)

        if battery_type == "standard":
            battery_action, battery_info = battery.step(battery_obs)
        else:
            battery_action, battery_info = battery.step(battery_obs, step_idx=step)

        record = env.apply_battery(battery_action, step, actual_hvac_energy)
        record.hvac_action = hvac_action
        result.history.append(record)
        result.thermo_efe_history.append(thermo_info["neg_efe"])
        result.battery_efe_history.append(battery_info["neg_efe"])
        result.thermo_qpi_history.append(thermo_info["q_pi"])
        result.battery_qpi_history.append(battery_info["q_pi"])

    return result


def run_all_conditions(env_data):
    """Run all 11 conditions for one scenario. Returns OrderedDict of results."""
    results = OrderedDict()
    total_steps = len(env_data["time_of_day"])

    # --- BASELINES ---
    print("    [1/11] No-HEMS...", end=" ", flush=True)
    results["No-HEMS"] = run_no_hems(env_data)
    print("done")

    print("    [2/11] Rule-based...", end=" ", flush=True)
    results["Rule-based"] = run_rule_based(env_data)
    print("done")

    print("    [3/11] Oracle...", end=" ", flush=True)
    results["Oracle"] = run_oracle(env_data, max_steps=total_steps)
    print("done")

    # --- DECENTRALIZED AIF ---
    print("    [4/11] Selfish...", end=" ", flush=True)
    results["Selfish"] = _run_custom_aif(env_data, thermo_aligned=False,
                                          battery_type="standard")
    print("done")

    print("    [5/11] Aligned...", end=" ", flush=True)
    results["Aligned"] = run_simulation(
        env_data=env_data, num_days=NUM_DAYS, policy_len=POLICY_LEN,
        gamma=GAMMA, learn_B=True, aligned=True, seed=SEED, verbose=False)
    print("done")

    print("    [6/11] Comm only...", end=" ", flush=True)
    results["Comm only"] = run_tom_simulation(
        env_data=env_data, num_days=NUM_DAYS, policy_len=POLICY_LEN,
        gamma=GAMMA, learn_B=True, social_weight=1.0,
        auditory_mode="none", seed=SEED, verbose=False)
    print("done")

    print("    [7/11] Phantom only...", end=" ", flush=True)
    results["Phantom only"] = _run_custom_aif(env_data, thermo_aligned=False,
                                               battery_type="sophisticated")
    print("done")

    print("    [8/11] Aligned+Comm...", end=" ", flush=True)
    results["Aligned+Comm"] = run_tom_simulation(
        env_data=env_data, num_days=NUM_DAYS, policy_len=POLICY_LEN,
        gamma=GAMMA, learn_B=True, social_weight=1.0,
        auditory_mode="full", seed=SEED, verbose=False)
    print("done")

    print("    [9/11] Aligned+Phantom...", end=" ", flush=True)
    results["Aligned+Phantom"] = run_sophisticated_simulation(
        env_data=env_data, num_days=NUM_DAYS, policy_len=POLICY_LEN,
        gamma=GAMMA, learn_B=True, seed=SEED, verbose=False)
    print("done")

    print("    [10/11] Full...", end=" ", flush=True)
    results["Full"] = run_sophisticated_tom_simulation(
        env_data=env_data, num_days=NUM_DAYS, policy_len=POLICY_LEN,
        gamma=GAMMA, learn_B=True, social_weight=1.0,
        seed=SEED, verbose=False)
    print("done")

    print("    [11/11] Symmetric...", end=" ", flush=True)
    results["Symmetric"] = run_full_sophisticated_simulation(
        env_data=env_data, num_days=NUM_DAYS, policy_len=POLICY_LEN,
        gamma=GAMMA, learn_B=True, cost_weight=1.0,
        seed=SEED, verbose=False)
    print("done")

    return results


def main():
    print("=" * 120)
    print("COMPREHENSIVE COMPARISON: Baselines vs Decentralized AIF Coordination")
    print(f"  Scenarios: {len(SCENARIOS)}, Days: {NUM_DAYS}, "
          f"policy_len: {POLICY_LEN}, gamma: {GAMMA}")
    print("=" * 120)

    # Generate environment data
    scenarios = {}
    for city, season in SCENARIOS:
        key = f"{city}_{season}"
        env_data = generate_climate_week(city, season, seed=SEED)
        scenarios[key] = env_data
        temps = env_data["outdoor_temp"]
        print(f"  {key}: T_out = [{temps.min():.1f}, {temps.max():.1f}]C, "
              f"steps={len(temps)}")

    # Run all conditions for each scenario
    all_metrics = {}
    for key, env_data in scenarios.items():
        print(f"\n--- {key} ---")
        results = run_all_conditions(env_data)

        no_hems = results["No-HEMS"]
        all_metrics[key] = OrderedDict()
        for cond_name, result in results.items():
            all_metrics[key][cond_name] = compute_metrics(
                result, num_days=NUM_DAYS, no_hems_result=no_hems)

    # --- PRINT RESULTS ---
    conditions = list(next(iter(all_metrics.values())).keys())

    for scenario_key in scenarios:
        print(f"\n\n{'=' * 120}")
        print(f"  {scenario_key}")
        print(f"{'=' * 120}")

        # Header
        header = f"{'Condition':<20}"
        for col in ["Cost($)", "Comfort", "Batt%", "GHG(kg)",
                     "CostSav%", "GHGSav%"]:
            header += f" {col:>10}"
        print(header)
        print("-" * 90)

        for cond in conditions:
            m = all_metrics[scenario_key][cond]
            cost_sav = m.cost_savings_vs_no_hems * 100
            ghg_sav = m.ghg_savings_vs_no_hems * 100
            row = f"{cond:<20}"
            row += f" {m.total_cost:>10.2f}"
            row += f" {m.comfort_deviation_total:>10.1f}"
            row += f" {m.battery_utilization*100:>10.1f}"
            row += f" {m.total_ghg:>10.2f}"
            row += f" {cost_sav:>10.1f}"
            row += f" {ghg_sav:>10.1f}"
            print(row)

    # --- CROSS-SCENARIO SUMMARY ---
    print(f"\n\n{'=' * 120}")
    print("CROSS-SCENARIO SUMMARY: Mean metrics across all scenarios")
    print(f"{'=' * 120}")

    header = f"{'Condition':<20} {'Avg Cost':>10} {'Avg Comfort':>12} {'Avg Batt%':>10} {'Avg GHG':>10} {'CostSav%':>10} {'GHGSav%':>10}"
    print(header)
    print("-" * 92)

    for cond in conditions:
        costs = [all_metrics[s][cond].total_cost for s in scenarios]
        comforts = [all_metrics[s][cond].comfort_deviation_total for s in scenarios]
        batts = [all_metrics[s][cond].battery_utilization for s in scenarios]
        ghgs = [all_metrics[s][cond].total_ghg for s in scenarios]
        csavs = [all_metrics[s][cond].cost_savings_vs_no_hems for s in scenarios]
        gsavs = [all_metrics[s][cond].ghg_savings_vs_no_hems for s in scenarios]

        row = f"{cond:<20}"
        row += f" {np.mean(costs):>10.2f}"
        row += f" {np.mean(comforts):>12.1f}"
        row += f" {np.mean(batts)*100:>10.1f}"
        row += f" {np.mean(ghgs):>10.2f}"
        row += f" {np.mean(csavs)*100:>10.1f}"
        row += f" {np.mean(gsavs)*100:>10.1f}"
        print(row)

    # --- DELTA TABLE: each AIF condition vs Oracle ---
    print(f"\n\n{'=' * 120}")
    print("GAP TO ORACLE (negative = AIF is cheaper/better)")
    print(f"{'=' * 120}")

    aif_conditions = [c for c in conditions if c not in ("No-HEMS", "Rule-based", "Oracle")]

    for scenario_key in scenarios:
        oracle_m = all_metrics[scenario_key]["Oracle"]
        print(f"\n  {scenario_key}:")
        for cond in aif_conditions:
            m = all_metrics[scenario_key][cond]
            dc = m.total_cost - oracle_m.total_cost
            df = m.comfort_deviation_total - oracle_m.comfort_deviation_total
            dg = m.total_ghg - oracle_m.total_ghg
            print(f"    {cond:<20} cost:{dc:>+8.2f}  comfort:{df:>+8.1f}  ghg:{dg:>+8.2f}")

    print(f"\n{'=' * 120}")
    print("Done.")


if __name__ == "__main__":
    main()
