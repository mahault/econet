"""Real weather data from IEM ASOS (Iowa Environmental Mesonet).

Downloads actual hourly observations for 4 cities, resamples to 2h steps,
estimates solar generation from cloud cover + latitude, and packages
into the env_data format used by run_simulation.

Data source: https://mesonet.agron.iastate.edu/request/download.phtml
No authentication required.
"""

import csv
import io
import math
import time
import numpy as np
from pathlib import Path
from typing import Optional
from urllib.request import urlopen
from urllib.error import URLError

from .environment import TEMP_MIN, TEMP_MAX, STEPS_PER_DAY


# ---------------------------------------------------------------------------
# Station definitions
# ---------------------------------------------------------------------------

STATIONS = {
    "london": {
        "icao": "EGLL",
        "name": "London Heathrow",
        "lat": 51.478,
        "lon": -0.461,
        "tz_offset": 0,   # UTC
        "panel_kwp": 3.0,  # assumed residential PV capacity (kWp)
        "ghg_base": 0.25,
        "ghg_peak": 0.40,
        "tou_low": 0.12,
        "tou_high": 0.28,
    },
    "phoenix": {
        "icao": "KPHX",
        "name": "Phoenix Sky Harbor",
        "lat": 33.428,
        "lon": -112.004,
        "tz_offset": -7,  # MST (no DST in AZ)
        "panel_kwp": 5.0,  # higher solar capacity in sunny climate
        "ghg_base": 0.35,
        "ghg_peak": 0.55,
        "tou_low": 0.08,
        "tou_high": 0.35,
    },
    "montreal": {
        "icao": "CYUL",
        "name": "Montreal Trudeau",
        "lat": 45.467,
        "lon": -73.733,
        "tz_offset": -5,  # EST
        "panel_kwp": 3.0,
        "ghg_base": 0.02,
        "ghg_peak": 0.08,
        "tou_low": 0.06,
        "tou_high": 0.15,
    },
    "miami": {
        "icao": "KMIA",
        "name": "Miami International",
        "lat": 25.788,
        "lon": -80.317,
        "tz_offset": -5,  # EST
        "panel_kwp": 4.0,
        "ghg_base": 0.40,
        "ghg_peak": 0.60,
        "tou_low": 0.09,
        "tou_high": 0.30,
    },
}

# Representative weeks (avoiding holidays/extremes)
PERIODS = {
    "summer": {"year1": 2023, "month1": 7, "day1": 15,
               "year2": 2023, "month2": 7, "day2": 22},
    "winter": {"year1": 2024, "month1": 1, "day1": 15,
               "year2": 2024, "month2": 1, "day2": 22},
}


# ---------------------------------------------------------------------------
# Cloud cover → solar estimation
# ---------------------------------------------------------------------------

# METAR sky cover codes → cloud fraction
SKY_COVER_FRACTION = {
    "CLR": 0.0,
    "SKC": 0.0,
    "FEW": 0.15,
    "SCT": 0.40,
    "BKN": 0.70,
    "OVC": 1.0,
    "VV":  1.0,   # vertical visibility (fog/low cloud)
    "M":   None,  # missing
}


def _solar_elevation(lat_deg: float, day_of_year: int, hour_utc: float,
                     lon_deg: float) -> float:
    """Approximate solar elevation angle in degrees.

    Uses a simplified model (accurate to ~1 degree).
    """
    lat = math.radians(lat_deg)
    # Solar declination
    decl = math.radians(23.45 * math.sin(math.radians(360 / 365 * (day_of_year - 81))))
    # Equation of time (minutes)
    B = math.radians(360 / 365 * (day_of_year - 81))
    eot = 9.87 * math.sin(2 * B) - 7.53 * math.cos(B) - 1.5 * math.sin(B)
    # Solar hour angle
    solar_time = hour_utc + lon_deg / 15.0 + eot / 60.0
    hour_angle = math.radians(15 * (solar_time - 12))
    # Elevation
    sin_elev = (math.sin(lat) * math.sin(decl)
                + math.cos(lat) * math.cos(decl) * math.cos(hour_angle))
    return math.degrees(math.asin(max(-1, min(1, sin_elev))))


def _estimate_solar_gen(lat: float, lon: float, day_of_year: int,
                        hour_utc: float, cloud_frac: float,
                        panel_kwp: float) -> float:
    """Estimate solar PV generation (kWh per 2h step).

    Parameters
    ----------
    lat, lon : float
        Station coordinates.
    day_of_year : int
        1-365.
    hour_utc : float
        Hour of day in UTC (center of 2h window).
    cloud_frac : float
        Cloud cover fraction [0, 1].
    panel_kwp : float
        PV panel rated capacity (kWp).

    Returns
    -------
    float
        Estimated generation in kWh for a 2h period.
    """
    elev = _solar_elevation(lat, day_of_year, hour_utc, lon)
    if elev <= 0:
        return 0.0

    # Clear-sky irradiance model (simplified)
    # At solar elevation α: DNI ≈ 1000 * sin(α) * atmospheric_transmission
    air_mass = 1.0 / max(math.sin(math.radians(elev)), 0.05)
    # Atmospheric transmission ~ 0.7^(air_mass^0.678)
    clearsky_factor = 0.7 ** (air_mass ** 0.678)
    ghi_clear = 1000 * math.sin(math.radians(elev)) * clearsky_factor  # W/m²

    # Cloud attenuation: GHI_cloudy ≈ GHI_clear × (1 - 0.75 × cloud_frac^3.4)
    # (Kasten-Czeplak model, simplified)
    cloud_attenuation = 1.0 - 0.75 * (cloud_frac ** 3.4)
    ghi = ghi_clear * cloud_attenuation

    # PV output: rated_kWp × (GHI / 1000) × performance_ratio
    perf_ratio = 0.80  # typical residential system
    power_kw = panel_kwp * (ghi / 1000.0) * perf_ratio

    # Energy over 2h period
    return power_kw * 2.0  # kWh


def _parse_cloud_fraction(skyc1, skyc2, skyc3) -> float:
    """Parse METAR sky cover layers into a single cloud fraction.

    Takes the maximum cloud cover from up to 3 layers.
    """
    max_frac = 0.0
    for code in [skyc1, skyc2, skyc3]:
        code = str(code).strip().upper()
        frac = SKY_COVER_FRACTION.get(code, None)
        if frac is not None and frac > max_frac:
            max_frac = frac
    return max_frac


# ---------------------------------------------------------------------------
# Data fetching
# ---------------------------------------------------------------------------

CACHE_DIR = Path(__file__).parent.parent / "data" / "weather_cache"


def _iem_url(station: str, period: dict) -> str:
    """Build IEM ASOS download URL."""
    return (
        f"https://mesonet.agron.iastate.edu/cgi-bin/request/asos.py?"
        f"station={station}"
        f"&data=tmpf&data=dwpf&data=skyc1&data=skyc2&data=skyc3"
        f"&year1={period['year1']}&month1={period['month1']}&day1={period['day1']}"
        f"&year2={period['year2']}&month2={period['month2']}&day2={period['day2']}"
        f"&tz=UTC&format=onlycomma&missing=M&trace=T&direct=no&report_type=3"
    )


def fetch_weather_data(city: str, season: str,
                       use_cache: bool = True) -> list[dict]:
    """Fetch hourly weather observations from IEM ASOS.

    Parameters
    ----------
    city : str
        One of: london, phoenix, montreal, miami.
    season : str
        One of: summer, winter.
    use_cache : bool
        If True, cache downloaded data locally.

    Returns
    -------
    list[dict]
        Each dict has: timestamp (str), temp_c (float), cloud_frac (float).
    """
    station_info = STATIONS[city]
    icao = station_info["icao"]
    period = PERIODS[season]

    # Check cache
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    cache_file = CACHE_DIR / f"{city}_{season}.csv"

    if use_cache and cache_file.exists():
        raw_text = cache_file.read_text()
    else:
        url = _iem_url(icao, period)
        print(f"    Fetching {station_info['name']} ({season}) from IEM...",
              flush=True)
        try:
            with urlopen(url, timeout=30) as resp:
                raw_text = resp.read().decode("utf-8")
        except URLError as e:
            print(f"    WARNING: Could not fetch {url}: {e}")
            print(f"    Falling back to synthetic data.")
            return []

        if use_cache:
            cache_file.write_text(raw_text)

        # Rate limit courtesy: wait before next request
        time.sleep(5)

    # Parse CSV
    observations = []
    reader = csv.DictReader(io.StringIO(raw_text))
    for row in reader:
        try:
            tmpf = float(row["tmpf"])
        except (ValueError, KeyError):
            continue  # skip missing temps

        temp_c = (tmpf - 32) * 5 / 9
        cloud_frac = _parse_cloud_fraction(
            row.get("skyc1", "M"),
            row.get("skyc2", "M"),
            row.get("skyc3", "M"),
        )
        observations.append({
            "timestamp": row["valid"],
            "temp_c": temp_c,
            "cloud_frac": cloud_frac,
        })

    return observations


# ---------------------------------------------------------------------------
# Resample to 2h env_data format
# ---------------------------------------------------------------------------

def _parse_timestamp(ts: str) -> tuple:
    """Parse 'YYYY-MM-DD HH:MM' into (year, month, day, hour, minute)."""
    date_part, time_part = ts.split(" ")
    y, m, d = date_part.split("-")
    hh, mm = time_part.split(":")
    return int(y), int(m), int(d), int(hh), int(mm)


def _day_of_year(year: int, month: int, day: int) -> int:
    """Compute day of year (1-365)."""
    import datetime
    return datetime.date(year, month, day).timetuple().tm_yday


def build_real_weather_env(city: str, season: str,
                           use_cache: bool = True) -> Optional[dict]:
    """Build env_data from real weather observations.

    Downloads hourly data, resamples to 2h steps (12 per day, 7 days),
    estimates solar generation, and adds TOU/occupancy/load profiles.

    Parameters
    ----------
    city : str
        One of: london, phoenix, montreal, miami.
    season : str
        One of: summer, winter.

    Returns
    -------
    dict or None
        env_data compatible with run_simulation, or None if fetch fails.
    """
    obs = fetch_weather_data(city, season, use_cache=use_cache)
    if not obs:
        return None

    station = STATIONS[city]
    lat = station["lat"]
    lon = station["lon"]
    panel_kwp = station["panel_kwp"]

    # Group observations by (day_offset, 2h_bin)
    # day_offset: 0-6 for 7 days
    # 2h_bin: 0-11 for hours 0-2, 2-4, ..., 22-24

    # Find the first observation's date to compute offsets
    y0, m0, d0, _, _ = _parse_timestamp(obs[0]["timestamp"])

    bins = {}  # (day_offset, bin_idx) → list of {temp_c, cloud_frac}
    for o in obs:
        y, m, d, h, mi = _parse_timestamp(o["timestamp"])
        # Day offset from first observation
        day_off = (d - d0) % 31  # handle month boundaries approximately
        if day_off >= 7:
            continue  # only 7 days

        bin_idx = h // 2  # 0-11
        key = (day_off, bin_idx)
        if key not in bins:
            bins[key] = []
        bins[key].append(o)

    # Build arrays: 7 days × 12 steps = 84 steps
    n_steps = 7 * STEPS_PER_DAY
    outdoor_temp = np.zeros(n_steps)
    solar_gen = np.zeros(n_steps)
    day_arr = np.zeros(n_steps, dtype=int)
    step_in_day = np.zeros(n_steps, dtype=int)
    time_of_day = np.zeros(n_steps, dtype=int)

    for day_off in range(7):
        for bin_idx in range(STEPS_PER_DAY):
            step = day_off * STEPS_PER_DAY + bin_idx
            key = (day_off, bin_idx)

            day_arr[step] = day_off
            step_in_day[step] = bin_idx
            time_of_day[step] = bin_idx * 2  # hours: 0, 2, 4, ..., 22

            if key in bins and bins[key]:
                # Average temperature in this 2h window
                temps = [o["temp_c"] for o in bins[key]]
                outdoor_temp[step] = np.mean(temps)

                # Average cloud fraction
                clouds = [o["cloud_frac"] for o in bins[key]]
                avg_cloud = np.mean(clouds)
            else:
                # Interpolate from neighbors
                prev_key = (day_off, bin_idx - 1) if bin_idx > 0 else None
                next_key = (day_off, bin_idx + 1) if bin_idx < 11 else None
                prev_obs = bins.get(prev_key, [])
                next_obs = bins.get(next_key, [])

                if prev_obs and next_obs:
                    outdoor_temp[step] = (np.mean([o["temp_c"] for o in prev_obs])
                                          + np.mean([o["temp_c"] for o in next_obs])) / 2
                    avg_cloud = (np.mean([o["cloud_frac"] for o in prev_obs])
                                 + np.mean([o["cloud_frac"] for o in next_obs])) / 2
                elif prev_obs:
                    outdoor_temp[step] = np.mean([o["temp_c"] for o in prev_obs])
                    avg_cloud = np.mean([o["cloud_frac"] for o in prev_obs])
                else:
                    outdoor_temp[step] = 15.0  # fallback
                    avg_cloud = 0.5

            # Compute day of year for solar calculation
            first_doy = _day_of_year(y0, m0, d0)
            doy = first_doy + day_off
            hour_utc = bin_idx * 2 + 1  # center of 2h window

            solar_gen[step] = _estimate_solar_gen(
                lat, lon, doy, hour_utc, avg_cloud, panel_kwp
            )

    # Clip outdoor temp to discretization range
    outdoor_temp = np.clip(outdoor_temp, TEMP_MIN, TEMP_MAX)

    # Generate household-specific profiles (same structure as synthetic)
    hours = time_of_day

    # Baseline load — morning + evening peaks
    baseline_load = np.zeros(n_steps)
    for i in range(n_steps):
        h = hours[i]
        weekend = day_arr[i] >= 5
        if weekend:
            baseline_load[i] = (0.8
                                + 1.0 * np.exp(-0.5 * ((h - 10) / 3) ** 2)
                                + 1.5 * np.exp(-0.5 * ((h - 19) / 2.5) ** 2))
        else:
            baseline_load[i] = (0.5
                                + 1.5 * np.exp(-0.5 * ((h - 7) / 2) ** 2)
                                + 2.0 * np.exp(-0.5 * ((h - 19) / 2) ** 2))

    # TOU rates
    tou_high_arr = np.zeros(n_steps, dtype=int)
    for i in range(n_steps):
        h = hours[i]
        if 6 <= h < 10 or 16 <= h < 20:
            tou_high_arr[i] = 1
    tou_value = np.where(tou_high_arr, station["tou_high"], station["tou_low"])

    # Occupancy
    occupancy = np.ones(n_steps, dtype=int)
    for i in range(n_steps):
        h = hours[i]
        weekend = day_arr[i] >= 5
        if not weekend and 8 <= h < 16:
            occupancy[i] = 0

    # GHG rate
    ghg_rate = np.where(tou_high_arr, station["ghg_peak"], station["ghg_base"])

    return {
        "day": day_arr,
        "step_in_day": step_in_day,
        "time_of_day": time_of_day,
        "outdoor_temp": outdoor_temp,
        "solar_gen": solar_gen,
        "baseline_load": baseline_load,
        "tou_high": tou_high_arr,
        "tou_value": tou_value,
        "occupancy": occupancy,
        "ghg_rate": ghg_rate,
    }


def generate_all_real_scenarios(use_cache: bool = True) -> dict:
    """Download and build env_data for all 4 cities × 2 seasons.

    Falls back to synthetic data if download fails for a city.
    Includes 5s delay between API requests to avoid rate limiting.

    Returns
    -------
    dict[str, dict]
        Keys like 'london_summer_real', values are env_data dicts.
    """
    from .climate import generate_climate_week

    scenarios = {}
    for city in STATIONS:
        for season in ["summer", "winter"]:
            key = f"{city}_{season}_real"
            env_data = build_real_weather_env(city, season, use_cache=use_cache)
            if env_data is not None:
                scenarios[key] = env_data
            else:
                # Fallback to synthetic
                print(f"    Using synthetic fallback for {city} {season}")
                scenarios[key] = generate_climate_week(city, season, seed=42)
    return scenarios


def get_real_scenario_label(key: str) -> str:
    """Human-readable label for a real scenario key."""
    parts = key.replace("_real", "").split("_")
    city, season = parts[0], parts[1]
    station = STATIONS[city]
    return f"{station['name']} ({season.capitalize()}, real)"
