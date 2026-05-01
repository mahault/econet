"""Household energy environment with 2-hour time steps.

Generates synthetic data (outdoor temp, solar, load, TOU rates, occupancy)
and simulates thermodynamics + battery storage.
"""

import numpy as np
from dataclasses import dataclass, field
from typing import Optional


# ---------------------------------------------------------------------------
# Discretization constants
# ---------------------------------------------------------------------------
TEMP_MIN = -20     # °C (covers Montreal winter)
TEMP_MAX = 46      # °C (covers Phoenix summer)
TEMP_STEP = 2      # °C per bin
TEMP_LEVELS = (TEMP_MAX - TEMP_MIN) // TEMP_STEP + 1   # 34 (2 °C bins)
SOC_LEVELS = 5     # 0.0, 0.2, 0.4, 0.6, 0.8
ENERGY_LEVELS = 10
COST_LEVELS = 10
GHG_LEVELS = 10

# Physical parameters
THERMAL_LEAKAGE = 0.15          # α — fraction of (T_out - T_in) per step
HVAC_POWER = 2.0                # β — °C change per step when heating/cooling
BATTERY_CAPACITY_KWH = 5.0      # kWh total
BATTERY_STEP_FRAC = 0.2         # 20 % of capacity per step
SOC_MIN_DISCHARGE = 0.2         # cannot discharge below this
SOC_MAX_CHARGE = 0.8            # cannot charge above this
HVAC_KWH_PER_STEP = 1.5         # kWh consumed when HVAC is active
BATTERY_EFFICIENCY = 0.9         # one-way efficiency; round-trip = 0.81

# Comfort targets
TARGET_TEMP_OCCUPIED = 18       # °C
TARGET_TEMP_UNOCCUPIED = 16     # °C

STEPS_PER_DAY = 12              # 24 h / 2 h = 12


# ---------------------------------------------------------------------------
# Synthetic data generation
# ---------------------------------------------------------------------------

def generate_synthetic_day(day_offset: int = 0,
                           temp_base: float = 16.5,
                           temp_amp: float = 8.5,
                           solar_peak: float = 3.0,
                           seed: Optional[int] = None) -> dict:
    """Generate one day (12 steps) of synthetic environment data.

    Parameters
    ----------
    day_offset : int
        Day number (for multi-day simulations).
    temp_base, temp_amp : float
        Outdoor temperature = temp_base + temp_amp * sin(…).
    solar_peak : float
        Peak solar generation in kWh.
    seed : int, optional
        RNG seed for reproducibility.

    Returns
    -------
    dict with keys: time_of_day, outdoor_temp, solar_gen, baseline_load,
                    tou_high, tou_value, occupancy, ghg_rate
    """
    rng = np.random.RandomState(seed)
    hours = np.arange(0, 24, 2)  # 0, 2, 4, …, 22

    # Outdoor temperature — sinusoidal with min at ~4 am, peak at ~16:00
    outdoor_temp = temp_base + temp_amp * np.sin(
        2 * np.pi * (hours - 4) / 24
    )
    # Add small noise
    outdoor_temp += rng.normal(0, 0.3, size=len(hours))
    outdoor_temp = np.clip(outdoor_temp, TEMP_MIN, TEMP_MAX)

    # Solar generation — bell curve peaking at noon
    solar_gen = solar_peak * np.exp(-0.5 * ((hours - 12) / 3) ** 2)
    solar_gen = np.clip(solar_gen, 0, None)

    # Baseline household load — morning + evening peaks
    baseline_load = (
        0.5
        + 1.5 * np.exp(-0.5 * ((hours - 7) / 2) ** 2)
        + 2.0 * np.exp(-0.5 * ((hours - 19) / 2) ** 2)
    )

    # Time-of-use tariff — high during 6-10 and 16-20
    tou_high = np.zeros(12, dtype=int)
    for i, h in enumerate(hours):
        if 6 <= h < 10 or 16 <= h < 20:
            tou_high[i] = 1
    tou_value = np.where(tou_high, 0.30, 0.10)  # $/kWh

    # Occupancy — home morning/evening/night, away midday
    occupancy = np.ones(12, dtype=int)
    for i, h in enumerate(hours):
        if 8 <= h < 16:
            occupancy[i] = 0

    # GHG emission rate — higher during peak
    ghg_rate = np.where(tou_high, 0.5, 0.3)  # kg CO2 / kWh

    return {
        "day": np.full(12, day_offset, dtype=int),
        "step_in_day": np.arange(12),
        "time_of_day": hours,
        "outdoor_temp": outdoor_temp,
        "solar_gen": solar_gen,
        "baseline_load": baseline_load,
        "tou_high": tou_high,
        "tou_value": tou_value,
        "occupancy": occupancy,
        "ghg_rate": ghg_rate,
    }


def generate_multi_day(num_days: int = 2, **kwargs) -> dict:
    """Generate multi-day data by concatenating single days."""
    days = []
    for d in range(num_days):
        seed = kwargs.get("seed", None)
        if seed is not None:
            seed = seed + d
        day = generate_synthetic_day(day_offset=d, seed=seed, **{
            k: v for k, v in kwargs.items() if k != "seed"
        })
        days.append(day)

    return {k: np.concatenate([d[k] for d in days]) for k in days[0]}


# ---------------------------------------------------------------------------
# Discretization helpers
# ---------------------------------------------------------------------------

def discretize_temp(temp_c: float) -> int:
    """Map continuous temperature to discrete bin index (0..TEMP_LEVELS-1)."""
    return int(np.clip(round((temp_c - TEMP_MIN) / TEMP_STEP), 0, TEMP_LEVELS - 1))


def continuous_temp(idx: int) -> float:
    """Map discrete bin index back to °C."""
    return float(TEMP_MIN + idx * TEMP_STEP)


def discretize_soc(soc: float) -> int:
    """Map SoC ∈ [0,1] to 5 levels: 0→0, 0.2→1, 0.4→2, 0.6→3, 0.8→4."""
    return int(np.clip(round(soc / 0.2), 0, SOC_LEVELS - 1))


def continuous_soc(idx: int) -> float:
    """Map discrete index to SoC value."""
    return idx * 0.2


def discretize_energy(energy_kwh: float, e_min: float = -3.0,
                      e_max: float = 6.0) -> int:
    """Map energy to 0..ENERGY_LEVELS-1."""
    frac = (energy_kwh - e_min) / (e_max - e_min)
    return int(np.clip(round(frac * (ENERGY_LEVELS - 1)), 0, ENERGY_LEVELS - 1))


def discretize_cost(cost: float, c_max: float = 2.0) -> int:
    """Map cost to 0..COST_LEVELS-1."""
    frac = cost / c_max
    return int(np.clip(round(frac * (COST_LEVELS - 1)), 0, COST_LEVELS - 1))


def discretize_ghg(ghg: float, g_max: float = 3.0) -> int:
    """Map GHG emissions to 0..GHG_LEVELS-1."""
    frac = ghg / g_max
    return int(np.clip(round(frac * (GHG_LEVELS - 1)), 0, GHG_LEVELS - 1))


# ---------------------------------------------------------------------------
# Environment
# ---------------------------------------------------------------------------

@dataclass
class StepRecord:
    """Record of one simulation step."""
    step: int = 0
    day: int = 0
    time_of_day: int = 0
    outdoor_temp: float = 0.0
    room_temp: float = 0.0
    target_temp: float = 0.0
    occupancy: int = 0
    tou_high: int = 0
    tou_value: float = 0.0
    solar_gen: float = 0.0
    baseline_load: float = 0.0
    hvac_action: int = 2        # 0=cool, 1=heat, 2=off
    hvac_energy: float = 0.0
    battery_action: int = 2     # 0=charge, 1=discharge, 2=off
    battery_energy: float = 0.0  # positive = consumption from grid
    soc: float = 0.5
    total_energy: float = 0.0
    cost: float = 0.0
    ghg: float = 0.0


class Environment:
    """Household energy simulation environment."""

    def __init__(self, env_data: dict, initial_room_temp: float = 20.0,
                 initial_soc: float = 0.5):
        self.data = env_data
        self.total_steps = len(env_data["time_of_day"])
        self.room_temp = initial_room_temp
        self.soc = initial_soc
        self.history: list[StepRecord] = []

    def get_target_temp(self, step: int) -> float:
        occ = self.data["occupancy"][step]
        return TARGET_TEMP_OCCUPIED if occ else TARGET_TEMP_UNOCCUPIED

    def get_thermostat_obs(self, step: int) -> dict:
        """Return observation dict for the thermostat agent."""
        return {
            "room_temp": discretize_temp(self.room_temp),
            "outdoor_temp": discretize_temp(self.data["outdoor_temp"][step]),
            "occupancy": int(self.data["occupancy"][step]),
            "tou_high": int(self.data["tou_high"][step]),
        }

    def apply_thermostat(self, action: int, step: int) -> float:
        """Apply thermostat action, update room temp. Returns HVAC energy (kWh).

        Actions: 0=cool, 1=heat, 2=off.
        """
        outdoor = self.data["outdoor_temp"][step]

        # Thermodynamic model
        hvac_effect = 0.0
        if action == 0:    # cool
            hvac_effect = -HVAC_POWER
        elif action == 1:  # heat
            hvac_effect = +HVAC_POWER

        self.room_temp = (
            self.room_temp
            + THERMAL_LEAKAGE * (outdoor - self.room_temp)
            + hvac_effect
        )
        self.room_temp = np.clip(self.room_temp, TEMP_MIN, TEMP_MAX)

        hvac_energy = HVAC_KWH_PER_STEP if action != 2 else 0.0
        return hvac_energy

    def get_battery_obs(self, step: int, hvac_energy: float) -> dict:
        """Return observation dict for the battery agent."""
        baseline = self.data["baseline_load"][step]
        solar = self.data["solar_gen"][step]
        total = baseline + hvac_energy - solar

        tou_val = self.data["tou_value"][step]
        cost = max(0, total) * tou_val
        ghg = max(0, total) * self.data["ghg_rate"][step]

        return {
            "soc": discretize_soc(self.soc),
            "cost": discretize_cost(cost),
            "ghg": discretize_ghg(ghg),
            "tou_high": int(self.data["tou_high"][step]),
            "energy_level": discretize_energy(total),
            # Raw values for metrics
            "_raw_total_energy": total,
            "_raw_cost": cost,
            "_raw_ghg": ghg,
        }

    def apply_battery(self, action: int, step: int, hvac_energy: float) -> StepRecord:
        """Apply battery action, compute final energy/cost/GHG. Returns StepRecord.

        Actions: 0=charge, 1=discharge, 2=off.
        """
        baseline = self.data["baseline_load"][step]
        solar = self.data["solar_gen"][step]

        # Battery dynamics with constraints
        batt_delta = 0.0  # change in SoC
        if action == 0 and self.soc < SOC_MAX_CHARGE:
            batt_delta = BATTERY_STEP_FRAC
        elif action == 1 and self.soc > SOC_MIN_DISCHARGE:
            batt_delta = -BATTERY_STEP_FRAC

        self.soc = np.clip(self.soc + batt_delta, 0.0, 1.0)

        # battery_energy: positive = consuming from grid, negative = supplying
        # Asymmetric efficiency: charging draws more from grid, discharging delivers less
        if batt_delta > 0:  # charging: grid delivers more than stored
            battery_energy_kwh = batt_delta * BATTERY_CAPACITY_KWH / BATTERY_EFFICIENCY
        elif batt_delta < 0:  # discharging: less delivered than drawn from battery
            battery_energy_kwh = batt_delta * BATTERY_CAPACITY_KWH * BATTERY_EFFICIENCY
        else:
            battery_energy_kwh = 0.0

        total_energy = baseline + hvac_energy + battery_energy_kwh - solar
        tou_val = self.data["tou_value"][step]
        cost = max(0, total_energy) * tou_val
        ghg = max(0, total_energy) * self.data["ghg_rate"][step]

        rec = StepRecord(
            step=step,
            day=int(self.data["day"][step]),
            time_of_day=int(self.data["time_of_day"][step]),
            outdoor_temp=float(self.data["outdoor_temp"][step]),
            room_temp=float(self.room_temp),
            target_temp=self.get_target_temp(step),
            occupancy=int(self.data["occupancy"][step]),
            tou_high=int(self.data["tou_high"][step]),
            tou_value=float(tou_val),
            solar_gen=float(solar),
            baseline_load=float(baseline),
            hvac_action=action if isinstance(action, int) else 2,
            hvac_energy=float(hvac_energy),
            battery_action=action,
            battery_energy=float(battery_energy_kwh),
            soc=float(self.soc),
            total_energy=float(total_energy),
            cost=float(cost),
            ghg=float(ghg),
        )
        self.history.append(rec)
        return rec

    def reset(self, initial_room_temp: float = 20.0, initial_soc: float = 0.5):
        """Reset environment state."""
        self.room_temp = initial_room_temp
        self.soc = initial_soc
        self.history = []
