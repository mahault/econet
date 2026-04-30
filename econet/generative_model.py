"""Generative models (A, B, C, D matrices) for thermostat and battery agents.

Uses JAX pymdp (v1.0) backend. Arrays are plain numpy — the Agent constructor
broadcasts them and converts to JAX internally.

Key improvement: B_dependencies allows room_temp transitions to depend on
outdoor_temp factor, encoding the FULL thermodynamic model across all
outdoor temperatures (no more NOMINAL_OUTDOOR_TEMP hack).
"""

import numpy as np

from .environment import (
    TEMP_MIN, TEMP_MAX, TEMP_LEVELS, SOC_LEVELS, ENERGY_LEVELS,
    COST_LEVELS, GHG_LEVELS, THERMAL_LEAKAGE, HVAC_POWER,
    BATTERY_STEP_FRAC, SOC_MIN_DISCHARGE, SOC_MAX_CHARGE,
    TARGET_TEMP_OCCUPIED, TARGET_TEMP_UNOCCUPIED,
    STEPS_PER_DAY, continuous_temp, continuous_soc,
    discretize_temp, discretize_soc, discretize_energy,
    discretize_cost, discretize_ghg,
)


# ======================================================================
# THERMOSTAT AGENT
# ======================================================================
# State factors: room_temp (TEMP_LEVELS), outdoor_temp (TEMP_LEVELS), occupancy (2), tou_high (2)
# Observations:  o_room_temp (TEMP_LEVELS), o_outdoor_temp (TEMP_LEVELS), o_occupancy (2), o_tou_high (2)
# Actions:       3 (cool=0, heat=1, off=2) — only factor 0 is controlled
#
# B_dependencies: [[0, 1], [1], [2], [3]]
#   Factor 0 (room_temp) depends on factors 0 AND 1 (outdoor_temp)
#   Factors 1-3 are exogenous (self-dependent only)
#
# A_dependencies: [[0], [1], [2], [3]]
#   Each obs modality depends on ONE factor only
# ======================================================================

THERMO_NUM_STATES = [TEMP_LEVELS, TEMP_LEVELS, 2, 2]
THERMO_NUM_OBS = [TEMP_LEVELS, TEMP_LEVELS, 2, 2]
THERMO_NUM_ACTIONS = 3
THERMO_NUM_CONTROLS = [THERMO_NUM_ACTIONS, 1, 1, 1]

THERMO_A_DEPS = [[0], [1], [2], [3]]
THERMO_B_DEPS = [[0, 1], [1], [2], [3]]


def _softmax_identity(n: int, sigma: float = 0.5) -> np.ndarray:
    """Near-identity A matrix: peaked at diagonal with Gaussian spread."""
    A = np.zeros((n, n))
    for s in range(n):
        for o in range(n):
            A[o, s] = np.exp(-0.5 * ((o - s) / sigma) ** 2)
        A[:, s] /= A[:, s].sum()
    return A


def build_thermostat_A() -> list:
    """Build A matrices for thermostat agent.

    With A_dependencies=[[0],[1],[2],[3]], each A[m] only indexes its
    relevant factor:
        A[0] shape: (26, 26)  -- P(o_room | s_room)
        A[1] shape: (26, 26)  -- P(o_outdoor | s_outdoor)
        A[2] shape: (2, 2)    -- P(o_occ | s_occ)
        A[3] shape: (2, 2)    -- P(o_tou | s_tou)
    """
    A = []

    # o_room_temp: softmax identity over room_temp factor
    A.append(_softmax_identity(TEMP_LEVELS, sigma=0.5))

    # o_outdoor_temp: softmax identity over outdoor_temp factor
    A.append(_softmax_identity(TEMP_LEVELS, sigma=0.5))

    # o_occupancy: identity
    A.append(np.eye(2))

    # o_tou: identity
    A.append(np.eye(2))

    return A


def build_thermostat_B() -> list:
    """Build B matrices for thermostat agent.

    With B_dependencies=[[0,1],[1],[2],[3]]:
        B[0] shape: (26, 26, 26, 3) -- P(s_room' | s_room, s_outdoor, action)
        B[1] shape: (26, 26, 1)     -- P(s_outdoor' | s_outdoor, no-op)
        B[2] shape: (2, 2, 1)       -- P(s_occ' | s_occ, no-op)
        B[3] shape: (2, 2, 1)       -- P(s_tou' | s_tou, no-op)
    """
    B = []

    # --- B_room_temp: (26, 26, 26, 3) ---
    # Axes: (next_room, prev_room, prev_outdoor, action)
    B_room = np.zeros((TEMP_LEVELS, TEMP_LEVELS, TEMP_LEVELS, THERMO_NUM_ACTIONS))

    for prev_r in range(TEMP_LEVELS):
        T_room = continuous_temp(prev_r)

        for prev_o in range(TEMP_LEVELS):
            T_out = continuous_temp(prev_o)  # actual outdoor temp per state

            for a in range(THERMO_NUM_ACTIONS):
                hvac_effect = 0.0
                if a == 0:    # cool
                    hvac_effect = -HVAC_POWER
                elif a == 1:  # heat
                    hvac_effect = +HVAC_POWER

                T_new = T_room + THERMAL_LEAKAGE * (T_out - T_room) + hvac_effect
                T_new = np.clip(T_new, TEMP_MIN, TEMP_MAX)
                idx = discretize_temp(T_new)

                # Gaussian spread around deterministic prediction
                for delta in range(-2, 3):
                    nxt = idx + delta
                    if 0 <= nxt < TEMP_LEVELS:
                        w = np.exp(-0.5 * (delta / 0.7) ** 2)
                        B_room[nxt, prev_r, prev_o, a] += w

    # Normalize over next_state dimension (axis 0)
    for prev_r in range(TEMP_LEVELS):
        for prev_o in range(TEMP_LEVELS):
            for a in range(THERMO_NUM_ACTIONS):
                col_sum = B_room[:, prev_r, prev_o, a].sum()
                if col_sum > 0:
                    B_room[:, prev_r, prev_o, a] /= col_sum

    B.append(B_room)

    # --- Exogenous factors: identity with single no-op action ---
    B.append(np.eye(TEMP_LEVELS).reshape(TEMP_LEVELS, TEMP_LEVELS, 1))
    B.append(np.eye(2).reshape(2, 2, 1))
    B.append(np.eye(2).reshape(2, 2, 1))

    return B


def build_thermostat_C() -> list:
    """Build C vectors (prior preferences) for thermostat agent.

    C[m] shape: (num_obs[m],)
    Higher values = more preferred observations.

    The amplitude must be strong enough that utility dominates info gain
    in EFE (~1-3 nats). Using -2.0 * dist²/4.0 gives -2.0 at dist=2,
    -4.5 at dist=3, ensuring comfort-seeking dominates exploration.
    """
    C = []

    # o_room_temp: Gaussian preference around target (strong)
    # Must dominate info gain (~0.5-2 nats). Using -4.0/4.0 = -dist² gives
    # -1.0 at 1°C, -4.0 at 2°C, -9.0 at 3°C — overwhelms epistemic drive.
    target = TARGET_TEMP_OCCUPIED
    target_idx = discretize_temp(target)
    c_room = np.zeros(TEMP_LEVELS)
    for i in range(TEMP_LEVELS):
        dist = abs(i - target_idx)
        c_room[i] = -4.0 * dist ** 2 / 4.0
    # Shift so max = 0
    c_room -= c_room.max()
    C.append(c_room)

    # Other modalities: flat (no preference)
    C.append(np.zeros(TEMP_LEVELS))
    # Occupancy: slight preference for occupied (triggers comfort tracking)
    C.append(np.array([0.0, 0.5]))
    C.append(np.zeros(2))

    return C


def build_thermostat_D(initial_room_temp: float = 20.0,
                       initial_outdoor_temp: float = 10.0,
                       initial_occupancy: int = 1,
                       initial_tou: int = 0) -> list:
    """Build D vectors (initial state priors)."""
    D = []

    d0 = np.zeros(TEMP_LEVELS)
    d0[discretize_temp(initial_room_temp)] = 1.0
    D.append(d0)

    d1 = np.zeros(TEMP_LEVELS)
    d1[discretize_temp(initial_outdoor_temp)] = 1.0
    D.append(d1)

    d2 = np.zeros(2)
    d2[initial_occupancy] = 1.0
    D.append(d2)

    d3 = np.zeros(2)
    d3[initial_tou] = 1.0
    D.append(d3)

    return D


def build_thermostat_model(env_data: dict,
                           initial_room_temp: float = 20.0,
                           policy_len: int = 4) -> dict:
    """Build complete thermostat generative model for JAX pymdp Agent."""
    return {
        "A": build_thermostat_A(),
        "B": build_thermostat_B(),
        "C": build_thermostat_C(),
        "D": build_thermostat_D(
            initial_room_temp=initial_room_temp,
            initial_outdoor_temp=env_data["outdoor_temp"][0],
            initial_occupancy=int(env_data["occupancy"][0]),
            initial_tou=int(env_data["tou_high"][0]),
        ),
        "A_dependencies": THERMO_A_DEPS,
        "B_dependencies": THERMO_B_DEPS,
        "num_controls": THERMO_NUM_CONTROLS,
    }


# ======================================================================
# COST-AWARE THERMOSTAT (5th factor: energy_demand, 5th modality: cost)
# ======================================================================
# Extends the 4-factor thermostat with energy_demand as a 5th state
# factor and cost as a 5th observation modality. Cost enters through
# the standard EFE (utility + ambiguity) — no custom loops needed.
# ======================================================================

THERMO_COST_NUM_STATES = [TEMP_LEVELS, TEMP_LEVELS, 2, 2, ENERGY_LEVELS]
THERMO_COST_NUM_OBS = [TEMP_LEVELS, TEMP_LEVELS, 2, 2, COST_LEVELS]
THERMO_COST_A_DEPS = [[0], [1], [2], [3], [4, 3]]
THERMO_COST_B_DEPS = [[0, 1], [1], [2], [3], [4]]
THERMO_COST_NUM_CONTROLS = [THERMO_NUM_ACTIONS, 1, 1, 1, THERMO_NUM_ACTIONS]


def build_thermo_A_cost() -> np.ndarray:
    """Build A[4]: P(o_cost | energy_demand, tou_high).

    Shape: (COST_LEVELS, ENERGY_LEVELS, 2)

    Maps energy demand and TOU period to expected cost observation.
    Same physics as battery's cost likelihood but without SoC coupling.
    """
    A_cost = np.zeros((COST_LEVELS, ENERGY_LEVELS, 2))
    for s_en in range(ENERGY_LEVELS):
        energy_kwh = -3.0 + s_en * 9.0 / (ENERGY_LEVELS - 1)
        for s_tou in range(2):
            tou_val = 0.30 if s_tou == 1 else 0.10
            cost = max(0, energy_kwh) * tou_val
            cost_idx = discretize_cost(cost)
            for delta in range(-1, 2):
                nxt = cost_idx + delta
                if 0 <= nxt < COST_LEVELS:
                    w = np.exp(-0.5 * (delta / 0.5) ** 2)
                    A_cost[nxt, s_en, s_tou] += w
    # Normalize over observation dimension
    sums = A_cost.sum(axis=0, keepdims=True)
    sums = np.where(sums == 0, 1, sums)
    A_cost = A_cost / sums
    return A_cost


def build_thermo_B_energy_initial() -> np.ndarray:
    """Build initial B[4]: P(energy' | energy, action).

    Shape: (ENERGY_LEVELS, ENERGY_LEVELS, 3)

    Placeholder transition matrix for energy_demand factor.
    Updated dynamically each step from phantom battery prediction.
    Initial: identity (no transition until phantom provides info).
    """
    B_en = np.zeros((ENERGY_LEVELS, ENERGY_LEVELS, THERMO_NUM_ACTIONS))
    for a in range(THERMO_NUM_ACTIONS):
        B_en[:, :, a] = np.eye(ENERGY_LEVELS)
    return B_en


def build_thermo_C_cost(cost_scale: float = 3.0) -> np.ndarray:
    """Build C[4]: cost preference vector.

    Shape: (COST_LEVELS,)

    Linear penalty: higher cost index = less preferred.
    cost_scale controls amplitude relative to comfort C (peak ~4.0 nats).
    """
    c_cost = np.zeros(COST_LEVELS)
    for i in range(COST_LEVELS):
        c_cost[i] = -cost_scale * i / (COST_LEVELS - 1)
    return c_cost


def build_thermostat_model_cost_aware(
    env_data: dict,
    initial_room_temp: float = 20.0,
    policy_len: int = 4,
    cost_scale: float = 3.0,
) -> dict:
    """Build 5-factor thermostat generative model with cost modality.

    Extends standard thermostat with energy_demand (factor 4) and
    cost (observation 4). All inference flows through standard pymdp
    machinery — no custom EFE loops.

    Parameters
    ----------
    env_data : dict
        Environment data arrays.
    initial_room_temp : float
        Starting room temperature.
    policy_len : int
        Planning horizon.
    cost_scale : float
        Amplitude of cost preference C[4]. Higher = more cost-sensitive.
        Default 3.0 (comfort C peaks at ~4.0, so 3.0 is meaningful).

    Returns
    -------
    dict with A, B, C, D, A_dependencies, B_dependencies, num_controls.
    """
    # Factors 0-3: standard thermostat
    A = build_thermostat_A()
    B = build_thermostat_B()
    C = build_thermostat_C()
    D = build_thermostat_D(
        initial_room_temp=initial_room_temp,
        initial_outdoor_temp=env_data["outdoor_temp"][0],
        initial_occupancy=int(env_data["occupancy"][0]),
        initial_tou=int(env_data["tou_high"][0]),
    )

    # Factor 4: energy_demand
    A.append(build_thermo_A_cost())
    B.append(build_thermo_B_energy_initial())
    C.append(build_thermo_C_cost(cost_scale))

    # D[4]: uniform prior over energy demand (no initial info)
    d4 = np.ones(ENERGY_LEVELS) / ENERGY_LEVELS
    D.append(d4)

    return {
        "A": A,
        "B": B,
        "C": C,
        "D": D,
        "A_dependencies": THERMO_COST_A_DEPS,
        "B_dependencies": THERMO_COST_B_DEPS,
        "num_controls": THERMO_COST_NUM_CONTROLS,
    }


# ======================================================================
# BATTERY AGENT
# ======================================================================
# State factors:  soc (5), tou_high (2), energy_level (10)
# Observations:   o_soc (5), o_cost (10), o_ghg (10)
# Actions:        3 (charge=0, discharge=1, off=2) — only factor 0 is controlled
#
# B_dependencies: [[0], [1], [2]]  (no cross-factor dependencies)
# A_dependencies: [[0], [0, 1, 2], [0, 1, 2]]
#   o_soc depends on soc only
#   o_cost depends on soc, tou, energy
#   o_ghg depends on soc, tou, energy
# ======================================================================

BATTERY_NUM_STATES = [SOC_LEVELS, 2, ENERGY_LEVELS]
BATTERY_NUM_OBS = [SOC_LEVELS, COST_LEVELS, GHG_LEVELS]
BATTERY_NUM_ACTIONS = 3
BATTERY_NUM_CONTROLS = [BATTERY_NUM_ACTIONS, 1, 1]

BATTERY_A_DEPS = [[0], [0, 1, 2], [0, 1, 2]]
BATTERY_B_DEPS = [[0], [1], [2]]


def build_battery_A() -> list:
    """Build A matrices for battery agent.

    With A_dependencies:
        A[0] shape: (5, 5)          -- P(o_soc | s_soc)
        A[1] shape: (10, 5, 2, 10)  -- P(o_cost | s_soc, s_tou, s_energy)
        A[2] shape: (10, 5, 2, 10)  -- P(o_ghg | s_soc, s_tou, s_energy)
    """
    A = []

    # o_soc: identity mapping from soc factor
    A.append(np.eye(SOC_LEVELS))

    # o_cost: depends on soc, tou, energy
    # SoC-cost coupling: the battery's marginal effect on cost is ±1 kWh × tou.
    # High SoC = recently charged (added cost), low SoC = recently discharged (saved cost).
    # Moderate coupling: ±0.6 kWh offset (enough for EFE signal, not overwhelming).
    A_cost = np.zeros((COST_LEVELS, SOC_LEVELS, 2, ENERGY_LEVELS))
    for s_soc in range(SOC_LEVELS):
        soc_offset_kwh = (s_soc - 2) * 0.3  # [-0.6, -0.3, 0, +0.3, +0.6]
        for s_tou in range(2):
            tou_val = 0.30 if s_tou == 1 else 0.10
            for s_en in range(ENERGY_LEVELS):
                energy_kwh = -3.0 + s_en * 9.0 / (ENERGY_LEVELS - 1)
                # Include SoC-dependent marginal cost
                effective_energy = energy_kwh + soc_offset_kwh
                cost = max(0, effective_energy) * tou_val
                cost_idx = discretize_cost(cost)
                for delta in range(-1, 2):
                    nxt = cost_idx + delta
                    if 0 <= nxt < COST_LEVELS:
                        w = np.exp(-0.5 * (delta / 0.5) ** 2)
                        A_cost[nxt, s_soc, s_tou, s_en] += w
    # Normalize over obs dimension
    sums = A_cost.sum(axis=0, keepdims=True)
    sums = np.where(sums == 0, 1, sums)
    A_cost = A_cost / sums
    A.append(A_cost)

    # o_ghg: depends on soc, tou, energy (same SoC-coupling as cost)
    A_ghg = np.zeros((GHG_LEVELS, SOC_LEVELS, 2, ENERGY_LEVELS))
    for s_soc in range(SOC_LEVELS):
        soc_offset_kwh = (s_soc - 2) * 0.3
        for s_tou in range(2):
            ghg_rate = 0.5 if s_tou == 1 else 0.3
            for s_en in range(ENERGY_LEVELS):
                energy_kwh = -3.0 + s_en * 9.0 / (ENERGY_LEVELS - 1)
                effective_energy = energy_kwh + soc_offset_kwh
                ghg = max(0, effective_energy) * ghg_rate
                ghg_idx = discretize_ghg(ghg)
                for delta in range(-1, 2):
                    nxt = ghg_idx + delta
                    if 0 <= nxt < GHG_LEVELS:
                        w = np.exp(-0.5 * (delta / 0.5) ** 2)
                        A_ghg[nxt, s_soc, s_tou, s_en] += w
    sums = A_ghg.sum(axis=0, keepdims=True)
    sums = np.where(sums == 0, 1, sums)
    A_ghg = A_ghg / sums
    A.append(A_ghg)

    return A


def build_battery_B() -> list:
    """Build B matrices for battery agent.

    With B_dependencies=[[0],[1],[2]]:
        B[0] shape: (5, 5, 3)       -- P(s_soc' | s_soc, action)
        B[1] shape: (2, 2, 1)       -- P(s_tou' | s_tou, no-op)
        B[2] shape: (10, 10, 1)     -- P(s_energy' | s_energy, no-op)
    """
    B = []

    # B_soc: (5, 5, 3)
    B_soc = np.zeros((SOC_LEVELS, SOC_LEVELS, BATTERY_NUM_ACTIONS))
    for prev_s in range(SOC_LEVELS):
        soc_val = continuous_soc(prev_s)
        for a in range(BATTERY_NUM_ACTIONS):
            if a == 0:  # charge
                new_soc = soc_val + BATTERY_STEP_FRAC if soc_val < SOC_MAX_CHARGE else soc_val
            elif a == 1:  # discharge
                new_soc = soc_val - BATTERY_STEP_FRAC if soc_val > SOC_MIN_DISCHARGE else soc_val
            else:  # off
                new_soc = soc_val
            new_soc = np.clip(new_soc, 0.0, 1.0)
            B_soc[discretize_soc(new_soc), prev_s, a] = 1.0
    B.append(B_soc)

    # Exogenous factors: identity with single no-op action
    B.append(np.eye(2).reshape(2, 2, 1))
    B.append(np.eye(ENERGY_LEVELS).reshape(ENERGY_LEVELS, ENERGY_LEVELS, 1))

    return B


def build_battery_C() -> list:
    """Build C vectors for battery agent.

    Higher values = more preferred observations.
    Amplitudes must dominate info gain (~1-3 nats) for good TOU arbitrage.
    """
    C = []

    # SoC preference: mild preference for mid SoC (dynamic TOU flip handles arbitrage)
    c_soc = np.zeros(SOC_LEVELS)
    for i in range(SOC_LEVELS):
        c_soc[i] = 0.3 * (i - 2)  # [-0.6, -0.3, 0, 0.3, 0.6]
    C.append(c_soc)

    # Strong preference for low cost (6x scale to dominate info gain)
    c_cost = np.zeros(COST_LEVELS)
    for i in range(COST_LEVELS):
        c_cost[i] = -6.0 * i / (COST_LEVELS - 1)
    C.append(c_cost)

    # Preference for low GHG
    c_ghg = np.zeros(GHG_LEVELS)
    for i in range(GHG_LEVELS):
        c_ghg[i] = -3.0 * i / (GHG_LEVELS - 1)
    C.append(c_ghg)

    return C


def build_battery_D(initial_soc: float = 0.5,
                    initial_tou: int = 0,
                    initial_energy_level: int = 5) -> list:
    """Build D vectors for battery agent."""
    D = []

    d0 = np.zeros(SOC_LEVELS)
    d0[discretize_soc(initial_soc)] = 1.0
    D.append(d0)

    d1 = np.zeros(2)
    d1[initial_tou] = 1.0
    D.append(d1)

    d2 = np.zeros(ENERGY_LEVELS)
    d2[initial_energy_level] = 1.0
    D.append(d2)

    return D


def build_battery_model(env_data: dict,
                        initial_soc: float = 0.5,
                        policy_len: int = 4) -> dict:
    """Build complete battery generative model for JAX pymdp Agent."""
    return {
        "A": build_battery_A(),
        "B": build_battery_B(),
        "C": build_battery_C(),
        "D": build_battery_D(
            initial_soc=initial_soc,
            initial_tou=int(env_data["tou_high"][0]),
        ),
        "A_dependencies": BATTERY_A_DEPS,
        "B_dependencies": BATTERY_B_DEPS,
        "num_controls": BATTERY_NUM_CONTROLS,
    }


# ======================================================================
# ToM-EXTENDED GENERATIVE MODELS (Phase 3: Belief Sharing)
# ======================================================================
# Each agent gets an additional "auditory" observation modality that
# encodes received beliefs from the other agent. The auditory A matrix
# models what the agent expects to receive given its own hidden state.
# Social preferences (C vector on auditory modality) drive G_social.
# ======================================================================

COMFORT_LEVELS = 5  # COLD, COOL, COMFY, WARM, HOT

# --- Thermostat + auditory (receives battery q(SoC)) ---
# A_deps[4] = [0]: auditory obs depends on ROOM_TEMP (thermostat's key state)
# Rationale: when room is far from target, thermostat expects battery to
# discharge (low SoC) to help offset HVAC cost. When room is at target,
# expects battery to be charging (high SoC). This means receiving unexpected
# SoC beliefs updates thermostat's own room_temp inference — genuine new info.
THERMO_TOM_NUM_OBS = [TEMP_LEVELS, TEMP_LEVELS, 2, 2, SOC_LEVELS]
THERMO_TOM_A_DEPS = [[0], [1], [2], [3], [0]]  # auditory depends on room_temp


def build_thermostat_A_tom() -> list:
    """Build A matrices for thermostat with auditory belief modality.

    A[0..3]: identical to standard thermostat
    A[4]: auditory — P(o_batt_soc | room_temp), shape (5, 26)
      - room cold/hot (far from target): expect low SoC (battery helping)
      - room at target: expect high SoC (battery idle/charging)

    Conditioning on room_temp (not tou_high) means the received SoC belief
    provides genuinely new information that updates room_temp inference.
    """
    A = build_thermostat_A()
    target_idx = discretize_temp(TARGET_TEMP_OCCUPIED)

    # Auditory A: P(received_soc_obs | room_temp)
    # When room is comfortable → battery likely charging (high SoC expected)
    # When room is uncomfortable → battery likely discharging to help (low SoC)
    A_aud = np.zeros((SOC_LEVELS, TEMP_LEVELS))
    for r in range(TEMP_LEVELS):
        dist = abs(r - target_idx)
        # comfort_factor: 1.0 at target, 0.0 far away
        comfort_factor = max(0.0, 1.0 - dist / 4.0)
        # Blend between "expect high SoC" (comfortable) and "expect low SoC" (uncomfortable)
        high_soc = np.array([0.05, 0.08, 0.17, 0.35, 0.35])
        low_soc = np.array([0.30, 0.30, 0.20, 0.12, 0.08])
        A_aud[:, r] = comfort_factor * high_soc + (1 - comfort_factor) * low_soc
    A.append(A_aud)

    return A


def build_thermostat_C_tom(social_weight: float = 1.0) -> list:
    """Build C vectors for thermostat with social preference on auditory obs.

    C[0..3]: identical to standard thermostat
    C[4]: preference for HIGH battery SoC (thermostat wants battery available)
          Scaled to compete with comfort C: peak ~3.0 nats.
    """
    C = build_thermostat_C()

    # Social C: thermostat prefers observing battery with capacity available
    # Scale: comfort C peaks at ~4.0 nats, so 3.0 is enough to shift marginal decisions
    c_batt = np.array([-4.0, -2.0, 0.0, 1.0, 2.0]) * social_weight
    C.append(c_batt)

    return C


def build_thermostat_model_tom(env_data: dict,
                               initial_room_temp: float = 20.0,
                               policy_len: int = 4,
                               social_weight: float = 1.0) -> dict:
    """Build thermostat generative model with auditory belief modality."""
    return {
        "A": build_thermostat_A_tom(),
        "B": build_thermostat_B(),
        "C": build_thermostat_C_tom(social_weight=social_weight),
        "D": build_thermostat_D(
            initial_room_temp=initial_room_temp,
            initial_outdoor_temp=env_data["outdoor_temp"][0],
            initial_occupancy=int(env_data["occupancy"][0]),
            initial_tou=int(env_data["tou_high"][0]),
        ),
        "A_dependencies": THERMO_TOM_A_DEPS,
        "B_dependencies": THERMO_B_DEPS,
        "num_controls": THERMO_NUM_CONTROLS,
    }


# --- Battery + auditory (receives thermostat q(comfort)) ---
# A_deps[3] = [0]: auditory obs depends on SoC (battery's key state)
# Rationale: when battery has high SoC, it expects thermostat to be comfortable
# (battery has been helping via discharge). When SoC is low, expects discomfort
# (battery depleted, can't help). Receiving unexpected comfort beliefs updates
# battery's own SoC inference — genuine new info for policy selection.
BATTERY_TOM_NUM_OBS = [SOC_LEVELS, COST_LEVELS, GHG_LEVELS, COMFORT_LEVELS]
BATTERY_TOM_A_DEPS = [[0], [0, 1, 2], [0, 1, 2], [0]]  # auditory depends on soc


def build_battery_A_tom() -> list:
    """Build A matrices for battery with auditory belief modality.

    A[0..2]: identical to standard battery
    A[3]: auditory — P(o_thermo_comfort | soc), shape (5, 5)
      - high SoC: expect comfortable (battery has been helping)
      - low SoC: expect discomfort (battery depleted, can't help)

    Conditioning on SoC means the received comfort belief provides genuinely
    new information that updates SoC inference — shifting battery decisions.
    """
    A = build_battery_A()

    # Auditory A: P(received_comfort_obs | soc)
    # High SoC → battery recently charged or idle → expect thermostat comfortable
    # Low SoC → battery recently discharged heavily → expect thermostat uncomfortable
    A_aud = np.zeros((COMFORT_LEVELS, SOC_LEVELS))
    # soc=0 (depleted): expect uncomfortable
    A_aud[:, 0] = np.array([0.25, 0.30, 0.25, 0.12, 0.08])
    # soc=1: slightly uncomfortable
    A_aud[:, 1] = np.array([0.15, 0.25, 0.30, 0.18, 0.12])
    # soc=2 (mid): neutral
    A_aud[:, 2] = np.array([0.10, 0.18, 0.40, 0.20, 0.12])
    # soc=3: slightly comfortable
    A_aud[:, 3] = np.array([0.08, 0.12, 0.35, 0.28, 0.17])
    # soc=4 (full): expect comfortable
    A_aud[:, 4] = np.array([0.05, 0.08, 0.27, 0.32, 0.28])
    A.append(A_aud)

    return A


def build_battery_C_tom(social_weight: float = 1.0) -> list:
    """Build C vectors for battery with social preference on auditory obs.

    C[0..2]: identical to standard battery
    C[3]: preference for COMFY thermostat (peaked at comfort level 2)
          Scaled to compete with cost C: peak ~4.0 nats.
    """
    C = build_battery_C()

    # Social C: battery prefers observing thermostat is comfortable
    # Must compete with cost C (peak -6.0). Using ±4.0 makes it meaningful.
    c_comfort = np.array([-4.0, -1.0, 2.0, -1.0, -4.0]) * social_weight
    C.append(c_comfort)

    return C


def build_battery_model_tom(env_data: dict,
                            initial_soc: float = 0.5,
                            policy_len: int = 4,
                            social_weight: float = 1.0) -> dict:
    """Build battery generative model with auditory belief modality."""
    return {
        "A": build_battery_A_tom(),
        "B": build_battery_B(),
        "C": build_battery_C_tom(social_weight=social_weight),
        "D": build_battery_D(
            initial_soc=initial_soc,
            initial_tou=int(env_data["tou_high"][0]),
        ),
        "A_dependencies": BATTERY_TOM_A_DEPS,
        "B_dependencies": BATTERY_B_DEPS,
        "num_controls": BATTERY_NUM_CONTROLS,
    }
