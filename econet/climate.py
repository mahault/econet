"""Multi-climate scenario generation for EcoNet.

Generates realistic synthetic weather profiles for 4 representative cities:
1. London (temperate)
2. Phoenix (hot-arid)
3. Montreal (cold)
4. Miami (tropical)

Each with summer + winter variants, 7 days each.
"""

import numpy as np
from .environment import TEMP_MIN, TEMP_MAX, STEPS_PER_DAY


# ---------------------------------------------------------------------------
# Climate profiles
# ---------------------------------------------------------------------------

CLIMATE_PROFILES = {
    "london": {
        "name": "London, UK",
        "type": "Temperate",
        "summer": {
            "temp_base": 18.0, "temp_amp": 5.0,    # 13-23°C
            "solar_peak": 2.5,
            "cloud_factor": 0.6,   # frequent clouds
        },
        "winter": {
            "temp_base": 6.0, "temp_amp": 3.0,     # 3-9°C
            "solar_peak": 0.8,
            "cloud_factor": 0.4,
        },
        "ghg_base": 0.25,    # relatively clean grid (UK)
        "ghg_peak": 0.40,
        "tou_low": 0.12,
        "tou_high": 0.28,
    },
    "phoenix": {
        "name": "Phoenix, AZ",
        "type": "Hot-Arid",
        "summer": {
            "temp_base": 36.0, "temp_amp": 8.0,    # 28-44°C → clipped to 30
            "solar_peak": 4.5,
            "cloud_factor": 0.95,  # clear skies
        },
        "winter": {
            "temp_base": 16.0, "temp_amp": 8.0,    # 8-24°C
            "solar_peak": 3.0,
            "cloud_factor": 0.9,
        },
        "ghg_base": 0.35,
        "ghg_peak": 0.55,
        "tou_low": 0.08,
        "tou_high": 0.35,
    },
    "montreal": {
        "name": "Montreal, QC",
        "type": "Cold",
        "summer": {
            "temp_base": 22.0, "temp_amp": 6.0,    # 16-28°C
            "solar_peak": 3.0,
            "cloud_factor": 0.7,
        },
        "winter": {
            "temp_base": -8.0, "temp_amp": 5.0,    # -13 to -3°C → clipped to 5
            "solar_peak": 1.0,
            "cloud_factor": 0.5,
        },
        "ghg_base": 0.02,    # hydro-dominated grid
        "ghg_peak": 0.08,
        "tou_low": 0.06,
        "tou_high": 0.15,
    },
    "miami": {
        "name": "Miami, FL",
        "type": "Tropical",
        "summer": {
            "temp_base": 30.0, "temp_amp": 4.0,    # 26-34°C → clipped to 30
            "solar_peak": 3.5,
            "cloud_factor": 0.65,  # afternoon thunderstorms
        },
        "winter": {
            "temp_base": 22.0, "temp_amp": 5.0,    # 17-27°C
            "solar_peak": 2.8,
            "cloud_factor": 0.8,
        },
        "ghg_base": 0.40,    # natural gas heavy
        "ghg_peak": 0.60,
        "tou_low": 0.09,
        "tou_high": 0.30,
    },
}


def generate_climate_day(profile: dict, season: str, day_offset: int = 0,
                         weekend: bool = False, seed: int = 42) -> dict:
    """Generate one day of climate-specific data.

    Parameters
    ----------
    profile : dict
        One of CLIMATE_PROFILES values.
    season : str
        'summer' or 'winter'.
    day_offset : int
        Day number.
    weekend : bool
        If True, modify occupancy for weekend pattern.
    seed : int
        RNG seed.
    """
    rng = np.random.RandomState(seed + day_offset)
    hours = np.arange(0, 24, 2)
    sp = profile[season]

    # Temperature
    outdoor_temp = sp["temp_base"] + sp["temp_amp"] * np.sin(
        2 * np.pi * (hours - 4) / 24
    )
    outdoor_temp += rng.normal(0, 0.5, size=12)
    outdoor_temp = np.clip(outdoor_temp, TEMP_MIN, TEMP_MAX)

    # Solar: bell curve × cloud factor with random variation
    cloud_day = sp["cloud_factor"] * (1.0 + rng.uniform(-0.15, 0.15))
    solar_gen = sp["solar_peak"] * cloud_day * np.exp(
        -0.5 * ((hours - 12) / 3) ** 2
    )
    solar_gen = np.clip(solar_gen, 0, None)

    # Baseline load
    if weekend:
        # Weekend: more uniform, slight late-morning peak
        baseline_load = (
            0.8
            + 1.0 * np.exp(-0.5 * ((hours - 10) / 3) ** 2)
            + 1.5 * np.exp(-0.5 * ((hours - 19) / 2.5) ** 2)
        )
    else:
        baseline_load = (
            0.5
            + 1.5 * np.exp(-0.5 * ((hours - 7) / 2) ** 2)
            + 2.0 * np.exp(-0.5 * ((hours - 19) / 2) ** 2)
        )

    # TOU
    tou_high = np.zeros(12, dtype=int)
    for i, h in enumerate(hours):
        if 6 <= h < 10 or 16 <= h < 20:
            tou_high[i] = 1
    tou_value = np.where(tou_high, profile["tou_high"], profile["tou_low"])

    # Occupancy
    occupancy = np.ones(12, dtype=int)
    if not weekend:
        for i, h in enumerate(hours):
            if 8 <= h < 16:
                occupancy[i] = 0
    # Weekend: home all day (occupancy stays 1)

    # GHG rate
    ghg_rate = np.where(tou_high, profile["ghg_peak"], profile["ghg_base"])

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


def generate_climate_week(city: str, season: str, seed: int = 42) -> dict:
    """Generate 7 days of climate data for a city/season.

    Days 0-4 = weekdays, days 5-6 = weekend.
    """
    profile = CLIMATE_PROFILES[city]
    days = []
    for d in range(7):
        weekend = d >= 5
        day = generate_climate_day(profile, season, day_offset=d,
                                   weekend=weekend, seed=seed)
        days.append(day)

    return {k: np.concatenate([d[k] for d in days]) for k in days[0]}


def generate_all_scenarios(seed: int = 42) -> dict:
    """Generate all climate scenarios.

    Returns
    -------
    dict[str, dict]
        Keys like 'london_summer', 'phoenix_winter', etc.
        Values are env_data dicts.
    """
    scenarios = {}
    for city in CLIMATE_PROFILES:
        for season in ["summer", "winter"]:
            key = f"{city}_{season}"
            scenarios[key] = generate_climate_week(city, season, seed=seed)
    return scenarios


def get_scenario_label(key: str) -> str:
    """Human-readable label for a scenario key."""
    city, season = key.split("_")
    profile = CLIMATE_PROFILES[city]
    return f"{profile['name']} ({season.capitalize()})"
