"""Tests for EcoNet core modules.

Run: python -m pytest tests/test_econet.py -v
"""

import numpy as np
import pytest
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from econet.environment import (
    generate_synthetic_day, generate_multi_day, Environment,
    discretize_temp, continuous_temp, discretize_soc, continuous_soc,
    discretize_energy, discretize_cost, discretize_ghg,
    TEMP_MIN, TEMP_MAX, TEMP_STEP, TEMP_LEVELS, SOC_LEVELS, STEPS_PER_DAY,
)
from econet.generative_model import (
    build_thermostat_A, build_thermostat_B, build_thermostat_C, build_thermostat_D,
    build_thermostat_model,
    build_battery_A, build_battery_B, build_battery_C, build_battery_D,
    build_battery_model,
    THERMO_NUM_STATES, THERMO_NUM_OBS,
    THERMO_A_DEPS, THERMO_B_DEPS,
    BATTERY_NUM_STATES, BATTERY_NUM_OBS,
    BATTERY_A_DEPS, BATTERY_B_DEPS,
)
from econet.baselines import run_no_hems, run_rule_based, run_oracle
from econet.metrics import compute_metrics
from econet.climate import generate_climate_week, CLIMATE_PROFILES
from econet.validation import (
    verify_energy_balance, verify_thermodynamic_consistency,
    verify_battery_constraints,
)


# =========================================================================
# Environment tests
# =========================================================================

class TestEnvironment:
    def test_synthetic_day_length(self):
        day = generate_synthetic_day(seed=42)
        assert len(day["time_of_day"]) == 12
        assert len(day["outdoor_temp"]) == 12
        assert len(day["solar_gen"]) == 12

    def test_multi_day_length(self):
        data = generate_multi_day(num_days=3, seed=42)
        assert len(data["time_of_day"]) == 36

    def test_outdoor_temp_in_range(self):
        day = generate_synthetic_day(seed=42)
        assert all(TEMP_MIN <= t <= TEMP_MAX for t in day["outdoor_temp"])

    def test_solar_nonnegative(self):
        day = generate_synthetic_day(seed=42)
        assert all(s >= 0 for s in day["solar_gen"])

    def test_tou_binary(self):
        day = generate_synthetic_day(seed=42)
        assert set(day["tou_high"]).issubset({0, 1})

    def test_occupancy_binary(self):
        day = generate_synthetic_day(seed=42)
        assert set(day["occupancy"]).issubset({0, 1})


class TestDiscretization:
    def test_temp_roundtrip(self):
        for idx in range(TEMP_LEVELS):
            t = continuous_temp(idx)
            assert discretize_temp(t) == idx

    def test_temp_clipping(self):
        assert discretize_temp(-30.0) == 0           # below TEMP_MIN clips to 0
        assert discretize_temp(60.0) == TEMP_LEVELS - 1  # above TEMP_MAX clips to max

    def test_soc_roundtrip(self):
        for s in [0.0, 0.2, 0.4, 0.6, 0.8]:
            idx = discretize_soc(s)
            assert abs(continuous_soc(idx) - s) < 0.01

    def test_energy_range(self):
        assert 0 <= discretize_energy(-3.0) <= 9
        assert 0 <= discretize_energy(6.0) <= 9


class TestEnvironmentSim:
    def setup_method(self):
        self.env_data = generate_multi_day(num_days=1, seed=42)
        self.env = Environment(self.env_data, initial_room_temp=20.0)

    def test_thermostat_obs_keys(self):
        obs = self.env.get_thermostat_obs(0)
        assert "room_temp" in obs
        assert "outdoor_temp" in obs
        assert "occupancy" in obs
        assert "tou_high" in obs

    def test_apply_thermostat_heating(self):
        T_before = self.env.room_temp
        self.env.apply_thermostat(1, 0)  # heat
        assert self.env.room_temp > T_before - 1

    def test_apply_thermostat_off(self):
        energy = self.env.apply_thermostat(2, 0)  # off
        assert energy == 0.0

    def test_battery_soc_bounds(self):
        for _ in range(20):
            self.env.apply_thermostat(2, 0)
            self.env.apply_battery(0, 0, 0)  # charge repeatedly
        assert 0 <= self.env.soc <= 1.0

    def test_environment_reset(self):
        self.env.apply_thermostat(1, 0)
        self.env.reset(initial_room_temp=15.0, initial_soc=0.3)
        assert self.env.room_temp == 15.0
        assert self.env.soc == 0.3
        assert len(self.env.history) == 0


# =========================================================================
# Generative model tests (JAX pymdp format with A/B_dependencies)
# =========================================================================

class TestThermostatModel:
    def setup_method(self):
        self.env_data = generate_multi_day(num_days=1, seed=42)

    def test_A_shapes(self):
        A = build_thermostat_A()
        assert len(A) == 4
        # With A_dependencies=[[0],[1],[2],[3]], each A[m] only indexes its factor
        assert A[0].shape == (TEMP_LEVELS, TEMP_LEVELS)   # (o_room, s_room)
        assert A[1].shape == (TEMP_LEVELS, TEMP_LEVELS)   # (o_outdoor, s_outdoor)
        assert A[2].shape == (2, 2)                        # (o_occ, s_occ)
        assert A[3].shape == (2, 2)                        # (o_tou, s_tou)

    def test_A_normalized(self):
        A = build_thermostat_A()
        for m in range(len(A)):
            # Sum over obs dimension (axis 0) should be 1 for each state combo
            sums = A[m].sum(axis=0)
            np.testing.assert_allclose(sums, 1.0, atol=1e-5)

    def test_B_room_shape(self):
        B = build_thermostat_B()
        # With B_dependencies=[[0,1],...]: (next_room, prev_room, prev_outdoor, action)
        assert B[0].shape == (TEMP_LEVELS, TEMP_LEVELS, TEMP_LEVELS, 3)

    def test_B_room_normalized(self):
        B = build_thermostat_B()
        B_room = B[0]
        for prev_r in range(TEMP_LEVELS):
            for prev_o in range(TEMP_LEVELS):
                for a in range(3):
                    col_sum = B_room[:, prev_r, prev_o, a].sum()
                    assert abs(col_sum - 1.0) < 1e-6, \
                        f"B_room not normalized at [{prev_r},{prev_o},{a}]: sum={col_sum}"

    def test_B_room_outdoor_conditioning(self):
        """Verify that outdoor temp actually affects room temp transitions."""
        B = build_thermostat_B()
        B_room = B[0]
        # Pick a mid-range room temp, same action (off=2)
        mid_r = discretize_temp(16)  # 16°C — mid-range
        cold_o = discretize_temp(0)  # 0°C outdoor
        hot_o = discretize_temp(30)  # 30°C outdoor
        dist_cold = B_room[:, mid_r, cold_o, 2]
        dist_hot = B_room[:, mid_r, hot_o, 2]
        # With cold outdoor, next temp should be lower
        mean_cold = np.dot(np.arange(TEMP_LEVELS), dist_cold)
        mean_hot = np.dot(np.arange(TEMP_LEVELS), dist_hot)
        assert mean_cold < mean_hot, \
            f"Cold outdoor ({mean_cold:.1f}) should predict lower temp than hot ({mean_hot:.1f})"

    def test_B_exogenous_identity(self):
        B = build_thermostat_B()
        np.testing.assert_allclose(B[1].reshape(TEMP_LEVELS, TEMP_LEVELS),
                                   np.eye(TEMP_LEVELS))
        np.testing.assert_allclose(B[2].reshape(2, 2), np.eye(2))
        np.testing.assert_allclose(B[3].reshape(2, 2), np.eye(2))

    def test_C_shape(self):
        C = build_thermostat_C()
        assert len(C) == 4
        assert C[0].shape == (TEMP_LEVELS,)

    def test_C_peaked_at_target(self):
        C = build_thermostat_C()
        target_idx = discretize_temp(18)  # TARGET_TEMP_OCCUPIED = 18
        assert C[0][target_idx] == 0.0
        assert C[0][0] < 0

    def test_D_valid_distribution(self):
        D = build_thermostat_D()
        for d in D:
            assert abs(d.sum() - 1.0) < 1e-6
            assert all(d >= 0)

    def test_full_model_keys(self):
        model = build_thermostat_model(self.env_data)
        assert "A" in model
        assert "B" in model
        assert "C" in model
        assert "D" in model
        assert "num_controls" in model
        assert "A_dependencies" in model
        assert "B_dependencies" in model
        assert model["A_dependencies"] == THERMO_A_DEPS
        assert model["B_dependencies"] == THERMO_B_DEPS


class TestBatteryModel:
    def setup_method(self):
        self.env_data = generate_multi_day(num_days=1, seed=42)

    def test_A_shapes(self):
        A = build_battery_A()
        assert len(A) == 3
        # A_deps = [[0], [0,1,2], [0,1,2]]
        assert A[0].shape == (SOC_LEVELS, SOC_LEVELS)  # o_soc depends on soc only
        assert A[1].shape == (10, SOC_LEVELS, 2, 10)   # o_cost depends on all
        assert A[2].shape == (10, SOC_LEVELS, 2, 10)   # o_ghg depends on all

    def test_B_soc_shape(self):
        B = build_battery_B()
        assert B[0].shape == (SOC_LEVELS, SOC_LEVELS, 3)

    def test_B_soc_deterministic(self):
        B = build_battery_B()
        B_soc = B[0]
        for s in range(SOC_LEVELS):
            for a in range(3):
                assert abs(B_soc[:, s, a].sum() - 1.0) < 1e-6

    def test_B_soc_constraints(self):
        B = build_battery_B()
        B_soc = B[0]
        # SoC=0.2 (idx=1), discharge (a=1) should stay at 1
        assert B_soc[1, 1, 1] == 1.0
        # SoC=0.8 (idx=4), charge (a=0) should stay at 4
        assert B_soc[4, 4, 0] == 1.0

    def test_full_model_keys(self):
        model = build_battery_model(self.env_data)
        assert "A_dependencies" in model
        assert "B_dependencies" in model
        assert model["A_dependencies"] == BATTERY_A_DEPS
        assert model["B_dependencies"] == BATTERY_B_DEPS


# =========================================================================
# Baseline tests
# =========================================================================

class TestBaselines:
    def setup_method(self):
        self.env_data = generate_multi_day(num_days=1, seed=42)

    def test_no_hems_no_energy(self):
        result = run_no_hems(self.env_data)
        arr = result.to_arrays()
        np.testing.assert_array_equal(arr["hvac_energy"], 0.0)

    def test_rule_based_runs(self):
        result = run_rule_based(self.env_data)
        assert len(result.history) == STEPS_PER_DAY

    def test_oracle_runs(self):
        result = run_oracle(self.env_data, max_steps=12)
        assert len(result.history) == 12

    def test_oracle_beats_no_hems_on_comfort(self):
        no_hems = run_no_hems(self.env_data)
        oracle = run_oracle(self.env_data)
        m_no = compute_metrics(no_hems, 1)
        m_or = compute_metrics(oracle, 1)
        assert m_or.comfort_deviation_total <= m_no.comfort_deviation_total + 1.0

    def test_oracle_cost_reasonable(self):
        """Oracle with DP should not cost more than no-HEMS."""
        no_hems = run_no_hems(self.env_data)
        oracle = run_oracle(self.env_data)
        # Oracle should be at least as cheap as no-HEMS or very close
        assert oracle.total_cost <= no_hems.total_cost + 0.5


# =========================================================================
# Climate tests
# =========================================================================

class TestClimate:
    def test_climate_week_length(self):
        data = generate_climate_week("london", "summer", seed=42)
        assert len(data["time_of_day"]) == 7 * STEPS_PER_DAY

    def test_all_cities(self):
        for city in CLIMATE_PROFILES:
            for season in ["summer", "winter"]:
                data = generate_climate_week(city, season, seed=42)
                assert len(data["outdoor_temp"]) == 84

    def test_phoenix_summer_hot(self):
        data = generate_climate_week("phoenix", "summer", seed=42)
        assert data["outdoor_temp"].max() >= 25

    def test_montreal_winter_cold(self):
        data = generate_climate_week("montreal", "winter", seed=42)
        assert data["outdoor_temp"].min() <= 10


# =========================================================================
# Metrics tests
# =========================================================================

class TestMetrics:
    def test_compute_metrics(self):
        env_data = generate_multi_day(num_days=1, seed=42)
        result = run_no_hems(env_data)
        m = compute_metrics(result, num_days=1)
        assert m.total_cost >= 0
        assert m.total_ghg >= 0
        assert m.comfort_deviation_total >= 0

    def test_savings_computation(self):
        env_data = generate_multi_day(num_days=1, seed=42)
        no_hems = run_no_hems(env_data)
        rule = run_rule_based(env_data)
        m = compute_metrics(rule, num_days=1, no_hems_result=no_hems)
        assert -1.0 <= m.cost_savings_vs_no_hems <= 1.0


# =========================================================================
# Validation tests
# =========================================================================

class TestValidation:
    def setup_method(self):
        self.env_data = generate_multi_day(num_days=1, seed=42)

    def test_energy_balance_no_hems(self):
        result = run_no_hems(self.env_data)
        eb = verify_energy_balance(result)
        assert eb["passed"], f"Energy balance failed: max error = {eb['max_error']}"

    def test_battery_constraints_no_hems(self):
        result = run_no_hems(self.env_data)
        bc = verify_battery_constraints(result)
        assert bc["passed"], f"Battery constraint violation: {bc['violations']}"

    def test_energy_balance_rule_based(self):
        result = run_rule_based(self.env_data)
        eb = verify_energy_balance(result)
        assert eb["passed"], f"Energy balance failed: max error = {eb['max_error']}"


# =========================================================================
# Strategy / hierarchical tests
# =========================================================================

from econet.strategy import (
    StrategyAgent, ObservationAggregator,
    build_strategy_A, build_strategy_B, build_strategy_C, build_strategy_D,
    build_thermo_C_for_strategy, build_battery_C_for_strategy,
    apply_strategy, STRATEGY_NAMES,
    NUM_ENERGY_REGIMES, NUM_DEMAND_PHASES,
    NUM_COST_TRENDS, NUM_COMFORT_TRENDS, NUM_SOC_STATES, NUM_STRATEGIES,
    STRATEGY_A_DEPS, STRATEGY_B_DEPS,
)


class TestStrategyModel:
    def test_A_shapes(self):
        A = build_strategy_A()
        assert len(A) == 3
        # A_deps = [[0, 1], [0], [1]]
        assert A[0].shape == (NUM_COST_TRENDS, NUM_ENERGY_REGIMES, NUM_DEMAND_PHASES)
        assert A[1].shape == (NUM_COMFORT_TRENDS, NUM_ENERGY_REGIMES)
        assert A[2].shape == (NUM_SOC_STATES, NUM_DEMAND_PHASES)

    def test_A_normalized(self):
        A = build_strategy_A()
        for m in range(len(A)):
            sums = A[m].sum(axis=0)
            np.testing.assert_allclose(sums, 1.0, atol=1e-5)

    def test_B_shapes(self):
        B = build_strategy_B()
        assert len(B) == 2
        assert B[0].shape == (NUM_ENERGY_REGIMES, NUM_ENERGY_REGIMES, NUM_STRATEGIES)
        assert B[1].shape == (NUM_DEMAND_PHASES, NUM_DEMAND_PHASES, 1)

    def test_B_normalized(self):
        B = build_strategy_B()
        for prev_r in range(NUM_ENERGY_REGIMES):
            for a in range(NUM_STRATEGIES):
                col = B[0][:, prev_r, a]
                assert abs(col.sum() - 1.0) < 1e-6

    def test_B_exogenous_identity(self):
        B = build_strategy_B()
        np.testing.assert_allclose(
            B[1].reshape(NUM_DEMAND_PHASES, NUM_DEMAND_PHASES),
            np.eye(NUM_DEMAND_PHASES)
        )

    def test_C_shapes(self):
        C = build_strategy_C()
        assert len(C) == 3
        assert C[0].shape == (NUM_COST_TRENDS,)
        assert C[1].shape == (NUM_COMFORT_TRENDS,)
        assert C[2].shape == (NUM_SOC_STATES,)

    def test_C_preferences_sensible(self):
        C = build_strategy_C()
        # Cost: prefer low cost (index 0 highest)
        assert C[0][0] >= C[0][-1]
        # Comfort: prefer excellent (index 0 highest)
        assert C[1][0] >= C[1][-1]

    def test_D_valid_distribution(self):
        D = build_strategy_D()
        for d in D:
            assert abs(d.sum() - 1.0) < 1e-6
            assert all(d >= 0)

    def test_full_model_keys(self):
        """Strategy model can construct an Agent without error."""
        agent = StrategyAgent(learn_B=False)
        assert agent.agent is not None


class TestCVectorProfiles:
    def test_all_thermo_profiles_valid(self):
        for name in STRATEGY_NAMES:
            C = build_thermo_C_for_strategy(name)
            assert len(C) == 4
            assert C[0].shape == (TEMP_LEVELS,)
            # Max should be 0 (shifted)
            assert abs(C[0].max()) < 1e-6

    def test_all_battery_profiles_valid(self):
        for name in STRATEGY_NAMES:
            C = build_battery_C_for_strategy(name)
            assert len(C) == 3
            assert C[0].shape == (SOC_LEVELS,)

    def test_minimize_cost_wider_than_comfort(self):
        """MINIMIZE_COST should have a wider comfort band than MAXIMIZE_COMFORT."""
        c_comfort = build_thermo_C_for_strategy("MAXIMIZE_COMFORT")[0]
        c_cost = build_thermo_C_for_strategy("MINIMIZE_COST")[0]
        # At far-from-target states, MINIMIZE_COST should be less negative
        assert c_cost[0] > c_comfort[0]

    def test_peak_shave_soc_profile(self):
        """PEAK_SHAVE should prefer higher SoC (positive values early)."""
        C = build_battery_C_for_strategy("PEAK_SHAVE")
        assert C[0][0] > C[0][-1]  # prefer full SoC


class TestObservationAggregator:
    def test_empty_aggregator_returns_defaults(self):
        agg = ObservationAggregator()
        obs = agg.summarize()
        assert "cost_trend" in obs
        assert "comfort_trend" in obs
        assert "soc_state" in obs
        assert "demand_phase" in obs

    def test_aggregator_with_records(self):
        from econet.environment import StepRecord
        agg = ObservationAggregator()
        for i in range(6):
            rec = StepRecord(
                step=i, cost=0.1 * i, room_temp=19.0, target_temp=18.0,
                soc=0.5, tou_high=1 if i < 3 else 0,
            )
            agg.add(rec)
        obs = agg.summarize()
        assert 0 <= obs["cost_trend"] < NUM_COST_TRENDS
        assert 0 <= obs["comfort_trend"] < NUM_COMFORT_TRENDS
        assert 0 <= obs["soc_state"] < NUM_SOC_STATES

    def test_aggregator_reset(self):
        from econet.environment import StepRecord
        agg = ObservationAggregator()
        agg.add(StepRecord(cost=1.0, room_temp=20.0, target_temp=18.0, soc=0.5))
        agg.reset()
        assert len(agg.records) == 0


class TestBLearning:
    """Test that online B-learning actually updates pB concentrations."""

    def test_pB_concentrations_increase(self):
        """After several steps with learn_B=True, pB should have grown."""
        env_data = generate_multi_day(num_days=2, seed=42)
        from econet.agents import ThermostatAgent
        agent = ThermostatAgent(env_data, policy_len=4, gamma=16.0, learn_B=True)

        # Record initial pB sum for factor 0 (room_temp)
        initial_pB_sum = float(np.asarray(agent.agent.pB[0]).sum())

        # Run a few steps
        from econet.environment import Environment
        env = Environment(env_data, initial_room_temp=20.0)
        for step in range(12):
            obs = env.get_thermostat_obs(step)
            action, energy, info = agent.step(obs)
            env.apply_thermostat(action, step)

        # pB should have grown (Dirichlet concentrations increase)
        final_pB_sum = float(np.asarray(agent.agent.pB[0]).sum())
        assert final_pB_sum > initial_pB_sum, \
            f"pB sum didn't grow: {initial_pB_sum:.1f} -> {final_pB_sum:.1f}"

    def test_learn_B_false_pB_unchanged(self):
        """With learn_B=False, agent has no pB to update."""
        env_data = generate_multi_day(num_days=1, seed=42)
        from econet.agents import ThermostatAgent
        agent = ThermostatAgent(env_data, policy_len=4, gamma=16.0, learn_B=False)
        assert agent.agent.pB is None


class TestStrategyAgent:
    def test_strategy_agent_step(self):
        """Strategy agent should return a valid strategy index."""
        agent = StrategyAgent(learn_B=False)
        obs = {"cost_trend": 2, "comfort_trend": 1, "soc_state": 1, "demand_phase": 1}
        strategy_idx, info = agent.step(obs)
        assert 0 <= strategy_idx < NUM_STRATEGIES
        assert "q_pi" in info
        assert "neg_efe" in info

    def test_strategy_agent_with_learning(self):
        """Strategy agent with B-learning should not crash over multiple steps."""
        agent = StrategyAgent(learn_B=True)
        obs_sequence = [
            {"cost_trend": 3, "comfort_trend": 2, "soc_state": 0, "demand_phase": 0},
            {"cost_trend": 1, "comfort_trend": 0, "soc_state": 2, "demand_phase": 1},
            {"cost_trend": 2, "comfort_trend": 1, "soc_state": 1, "demand_phase": 2},
        ]
        for obs in obs_sequence:
            strategy_idx, info = agent.step(obs)
            assert 0 <= strategy_idx < NUM_STRATEGIES


class TestApplyStrategy:
    def test_apply_strategy_changes_C(self):
        """apply_strategy should modify agent C vectors."""
        env_data = generate_multi_day(num_days=1, seed=42)
        from econet.agents import ThermostatAgent, BatteryAgent
        thermo = ThermostatAgent(env_data, policy_len=4, gamma=16.0)
        battery = BatteryAgent(env_data, policy_len=4, gamma=16.0)

        # Get initial C
        c0_before = np.asarray(thermo.agent.C[0]).copy()

        # Apply MAXIMIZE_COMFORT (tighter C than default)
        apply_strategy(0, thermo, battery)

        c0_after = np.asarray(thermo.agent.C[0])
        # C vectors should have changed
        assert not np.allclose(c0_before, c0_after), \
            "C vectors should change after apply_strategy"


class TestHierarchicalSimulation:
    def test_hierarchical_runs(self):
        """Hierarchical simulation should complete without errors."""
        from econet.simulation import run_hierarchical_simulation
        env_data = generate_multi_day(num_days=2, seed=42)
        result = run_hierarchical_simulation(
            env_data=env_data,
            num_days=2,
            policy_len=4,
            gamma=16.0,
            learn_B=False,
            verbose=False,
        )
        assert result.total_steps == 24
        assert result.total_cost >= 0

    def test_hierarchical_with_learning(self):
        """Hierarchical simulation with B-learning should complete."""
        from econet.simulation import run_hierarchical_simulation
        env_data = generate_multi_day(num_days=2, seed=42)
        result = run_hierarchical_simulation(
            env_data=env_data,
            num_days=2,
            policy_len=4,
            gamma=16.0,
            learn_B=True,
            verbose=False,
        )
        assert result.total_steps == 24

    def test_flat_simulation_still_works(self):
        """Existing flat simulation should still run correctly."""
        from econet.simulation import run_simulation
        env_data = generate_multi_day(num_days=1, seed=42)
        result = run_simulation(
            env_data=env_data,
            num_days=1,
            policy_len=4,
            learn_B=False,
            verbose=False,
        )
        assert result.total_steps == 12
        assert result.total_cost >= 0


# =========================================================================
# Theory of Mind + Belief Sharing tests (Phase 3)
# =========================================================================

from econet.tom import (
    BatteryToM, ThermostatToM,
    belief_to_comfort, belief_to_obs,
    COMFORT_LEVELS, _near_identity, _persistence_matrix,
)
from econet.generative_model import (
    build_thermostat_A_tom, build_thermostat_C_tom,
    build_thermostat_model_tom,
    build_battery_A_tom, build_battery_C_tom,
    build_battery_model_tom,
    THERMO_TOM_NUM_OBS, THERMO_TOM_A_DEPS,
    BATTERY_TOM_NUM_OBS, BATTERY_TOM_A_DEPS,
)
from econet.agents import ToMThermostatAgent, ToMBatteryAgent
from econet.metrics import compute_communication_metrics, CommunicationMetrics


class TestBeliefProjection:
    def test_belief_to_comfort_shape(self):
        q_room = np.ones(TEMP_LEVELS) / TEMP_LEVELS
        target_idx = discretize_temp(18)  # 18°C
        q_comfort = belief_to_comfort(q_room, target_idx)
        assert q_comfort.shape == (COMFORT_LEVELS,)

    def test_belief_to_comfort_sums_to_one(self):
        q_room = np.ones(TEMP_LEVELS) / TEMP_LEVELS
        q_comfort = belief_to_comfort(q_room, target_idx=discretize_temp(18))
        np.testing.assert_allclose(q_comfort.sum(), 1.0, atol=1e-6)

    def test_belief_to_comfort_peaked_at_target(self):
        """If q(room) is peaked at target, q(comfort) should peak at COMFY (idx 2)."""
        q_room = np.zeros(TEMP_LEVELS)
        target_idx = discretize_temp(18)
        q_room[target_idx] = 1.0
        q_comfort = belief_to_comfort(q_room, target_idx)
        assert np.argmax(q_comfort) == 2  # COMFY

    def test_belief_to_comfort_cold(self):
        """If q(room) is peaked far below target, comfort should be COLD (idx 0)."""
        q_room = np.zeros(TEMP_LEVELS)
        q_room[0] = 1.0  # very cold
        q_comfort = belief_to_comfort(q_room, target_idx=discretize_temp(18))
        assert np.argmax(q_comfort) == 0  # COLD

    def test_belief_to_obs_returns_argmax(self):
        q = np.array([0.1, 0.2, 0.5, 0.15, 0.05])
        assert belief_to_obs(q) == 2

    def test_belief_to_obs_first_max(self):
        q = np.array([0.5, 0.5, 0.0, 0.0, 0.0])
        assert belief_to_obs(q) == 0  # argmax returns first


class TestToMFilters:
    def test_battery_tom_initial_uniform(self):
        tom = BatteryToM(n_states=5)
        np.testing.assert_allclose(tom.q_state, 0.2, atol=1e-6)
        assert tom.reliability == 0.5

    def test_battery_tom_converges(self):
        """After repeated consistent beliefs, ToM should converge to that state."""
        tom = BatteryToM(n_states=5)
        true_belief = np.array([0.0, 0.0, 0.0, 0.1, 0.9])
        for _ in range(20):
            tom.update(true_belief, belief_to_obs(true_belief))
        # Should converge toward high SoC states
        assert np.argmax(tom.q_state) >= 3

    def test_battery_tom_reliability_grows(self):
        """Reliability should grow when beliefs are consistent."""
        tom = BatteryToM(n_states=5)
        initial_rel = tom.reliability
        true_belief = np.array([0.0, 0.1, 0.7, 0.15, 0.05])
        for _ in range(20):
            tom.update(true_belief, belief_to_obs(true_belief))
        assert tom.reliability > initial_rel

    def test_thermostat_tom_initial_uniform(self):
        tom = ThermostatToM()
        np.testing.assert_allclose(tom.q_state, 0.2, atol=1e-6)

    def test_thermostat_tom_converges(self):
        tom = ThermostatToM()
        comfy_belief = np.array([0.0, 0.1, 0.8, 0.1, 0.0])
        for _ in range(20):
            tom.update(comfy_belief, belief_to_obs(comfy_belief))
        assert np.argmax(tom.q_state) == 2  # COMFY

    def test_gated_belief_at_zero_reliability(self):
        """At reliability=0, gated belief = prior."""
        tom = BatteryToM(n_states=5)
        tom.reliability = 0.0
        prior = np.ones(5) / 5
        result = tom.get_gated_belief(prior)
        np.testing.assert_allclose(result, prior, atol=1e-6)

    def test_gated_belief_at_full_reliability(self):
        """At reliability=1, gated belief = q_state."""
        tom = BatteryToM(n_states=5)
        tom.reliability = 1.0
        tom.q_state = np.array([0.0, 0.0, 1.0, 0.0, 0.0])
        prior = np.ones(5) / 5
        result = tom.get_gated_belief(prior)
        np.testing.assert_allclose(result, tom.q_state, atol=1e-6)

    def test_gated_belief_at_half_reliability(self):
        """At reliability=0.5, gated belief = 0.5 * q_state + 0.5 * prior."""
        tom = BatteryToM(n_states=5)
        tom.reliability = 0.5
        tom.q_state = np.array([0.0, 0.0, 1.0, 0.0, 0.0])
        prior = np.ones(5) / 5
        result = tom.get_gated_belief(prior)
        expected = 0.5 * tom.q_state + 0.5 * prior
        np.testing.assert_allclose(result, expected, atol=1e-6)

    def test_tom_reset(self):
        tom = BatteryToM(n_states=5)
        tom.update(np.array([0.0, 0.0, 0.0, 0.1, 0.9]), 4)
        tom.reset()
        np.testing.assert_allclose(tom.q_state, 0.2, atol=1e-6)
        assert tom.reliability == 0.5
        assert len(tom.history) == 0


class TestToMHelperMatrices:
    def test_near_identity_shape(self):
        A = _near_identity(5, sigma=0.5)
        assert A.shape == (5, 5)

    def test_near_identity_normalized(self):
        A = _near_identity(5, sigma=0.5)
        sums = A.sum(axis=0)
        np.testing.assert_allclose(sums, 1.0, atol=1e-6)

    def test_near_identity_peaked(self):
        A = _near_identity(5, sigma=0.5)
        for s in range(5):
            assert np.argmax(A[:, s]) == s

    def test_persistence_matrix_normalized(self):
        B = _persistence_matrix(5, stay_prob=0.7)
        sums = B.sum(axis=0)
        np.testing.assert_allclose(sums, 1.0, atol=1e-6)

    def test_persistence_matrix_peaked_diagonal(self):
        B = _persistence_matrix(5, stay_prob=0.7)
        for s in range(5):
            assert np.argmax(B[:, s]) == s


class TestToMGenerativeModels:
    def setup_method(self):
        self.env_data = generate_multi_day(num_days=1, seed=42)

    def test_thermostat_tom_A_shapes(self):
        A = build_thermostat_A_tom()
        assert len(A) == 5  # 4 original + 1 auditory
        assert A[4].shape == (SOC_LEVELS, TEMP_LEVELS)  # P(o_batt_soc | room_temp)

    def test_thermostat_tom_A_auditory_normalized(self):
        A = build_thermostat_A_tom()
        sums = A[4].sum(axis=0)
        np.testing.assert_allclose(sums, 1.0, atol=1e-6)

    def test_thermostat_tom_C_has_social(self):
        C = build_thermostat_C_tom(social_weight=1.0)
        assert len(C) == 5
        assert C[4].shape == (SOC_LEVELS,)
        # Battery having high SoC should be preferred
        assert C[4][-1] > C[4][0]

    def test_thermostat_tom_model_keys(self):
        model = build_thermostat_model_tom(self.env_data)
        assert len(model["A"]) == 5
        assert len(model["C"]) == 5
        assert model["A_dependencies"] == THERMO_TOM_A_DEPS

    def test_battery_tom_A_shapes(self):
        A = build_battery_A_tom()
        assert len(A) == 4  # 3 original + 1 auditory
        assert A[3].shape == (COMFORT_LEVELS, SOC_LEVELS)  # P(o_comfort | soc)

    def test_battery_tom_A_auditory_normalized(self):
        A = build_battery_A_tom()
        sums = A[3].sum(axis=0)
        np.testing.assert_allclose(sums, 1.0, atol=1e-6)

    def test_battery_tom_C_has_social(self):
        C = build_battery_C_tom(social_weight=1.0)
        assert len(C) == 4
        assert C[3].shape == (COMFORT_LEVELS,)
        # COMFY (idx 2) should be most preferred
        assert np.argmax(C[3]) == 2

    def test_battery_tom_model_keys(self):
        model = build_battery_model_tom(self.env_data)
        assert len(model["A"]) == 4
        assert len(model["C"]) == 4
        assert model["A_dependencies"] == BATTERY_TOM_A_DEPS

    def test_social_weight_scaling(self):
        C_w1 = build_thermostat_C_tom(social_weight=1.0)
        C_w2 = build_thermostat_C_tom(social_weight=2.0)
        # Social C scales linearly with social_weight
        np.testing.assert_allclose(C_w2[4], C_w1[4] * 2.0, atol=1e-6)


class TestToMAgents:
    def setup_method(self):
        self.env_data = generate_multi_day(num_days=1, seed=42)

    def test_tom_thermostat_creates(self):
        agent = ToMThermostatAgent(self.env_data, policy_len=4, gamma=16.0)
        assert agent.battery_tom is not None
        assert agent.battery_tom.reliability == 0.5

    def test_tom_battery_creates(self):
        agent = ToMBatteryAgent(self.env_data, policy_len=4, gamma=16.0)
        assert agent.thermo_tom is not None
        assert agent.thermo_tom.reliability == 0.5

    def test_tom_thermostat_step_without_belief(self):
        """First step with no received belief should work."""
        agent = ToMThermostatAgent(self.env_data, policy_len=4, gamma=16.0)
        from econet.environment import Environment
        env = Environment(self.env_data, initial_room_temp=20.0)
        obs = env.get_thermostat_obs(0)
        action, energy, info = agent.step(obs, received_q_soc=None)
        assert 0 <= action <= 2
        assert "q_comfort" in info
        assert info["q_comfort"].shape == (COMFORT_LEVELS,)
        assert abs(info["q_comfort"].sum() - 1.0) < 1e-5

    def test_tom_battery_step_without_belief(self):
        """First step with no received belief should work."""
        agent = ToMBatteryAgent(self.env_data, policy_len=4, gamma=16.0)
        from econet.environment import Environment
        env = Environment(self.env_data, initial_room_temp=20.0)
        obs = env.get_battery_obs(0, 0.0)
        action, info = agent.step(obs, received_q_comfort=None)
        assert 0 <= action <= 2
        assert "q_soc" in info
        assert info["q_soc"].shape == (SOC_LEVELS,)

    def test_tom_thermostat_step_with_belief(self):
        """Step with received belief should update ToM."""
        agent = ToMThermostatAgent(self.env_data, policy_len=4, gamma=16.0)
        from econet.environment import Environment
        env = Environment(self.env_data, initial_room_temp=20.0)
        obs = env.get_thermostat_obs(0)
        q_soc = np.array([0.05, 0.1, 0.7, 0.1, 0.05])
        action, energy, info = agent.step(obs, received_q_soc=q_soc)
        assert 0 <= action <= 2
        assert len(agent.battery_tom.history) == 1

    def test_tom_battery_step_with_belief(self):
        """Step with received belief should update ToM."""
        agent = ToMBatteryAgent(self.env_data, policy_len=4, gamma=16.0)
        from econet.environment import Environment
        env = Environment(self.env_data, initial_room_temp=20.0)
        obs = env.get_battery_obs(0, 0.0)
        q_comfort = np.array([0.05, 0.15, 0.60, 0.15, 0.05])
        action, info = agent.step(obs, received_q_comfort=q_comfort)
        assert 0 <= action <= 2
        assert len(agent.thermo_tom.history) == 1


class TestToMSimulation:
    def test_tom_simulation_runs(self):
        """ToM simulation should complete without errors."""
        from econet.simulation import run_tom_simulation
        env_data = generate_multi_day(num_days=2, seed=42)
        result = run_tom_simulation(
            env_data=env_data, num_days=2,
            policy_len=4, gamma=16.0,
            learn_B=False, verbose=False,
        )
        assert result.total_steps == 24
        assert result.total_cost >= 0

    def test_tom_simulation_with_learning(self):
        """ToM simulation with B-learning should complete."""
        from econet.simulation import run_tom_simulation
        env_data = generate_multi_day(num_days=2, seed=42)
        result = run_tom_simulation(
            env_data=env_data, num_days=2,
            policy_len=4, gamma=16.0,
            learn_B=True, verbose=False,
        )
        assert result.total_steps == 24

    def test_tom_simulation_has_belief_history(self):
        """ToM simulation should store belief history."""
        from econet.simulation import run_tom_simulation
        env_data = generate_multi_day(num_days=1, seed=42)
        result = run_tom_simulation(
            env_data=env_data, num_days=1,
            policy_len=4, gamma=16.0,
            learn_B=False, verbose=False,
        )
        bh = result.belief_history
        assert len(bh["q_comfort"]) == 12
        assert len(bh["q_soc"]) == 12
        assert len(bh["thermo_tom_reliability"]) == 12
        assert len(bh["battery_tom_reliability"]) == 12

    def test_tom_beliefs_are_valid_distributions(self):
        """Shared beliefs should be valid probability distributions."""
        from econet.simulation import run_tom_simulation
        env_data = generate_multi_day(num_days=1, seed=42)
        result = run_tom_simulation(
            env_data=env_data, num_days=1,
            policy_len=4, gamma=16.0,
            learn_B=False, verbose=False,
        )
        for q in result.belief_history["q_comfort"]:
            assert q.shape == (COMFORT_LEVELS,)
            assert abs(q.sum() - 1.0) < 1e-5
            assert np.all(q >= 0)
        for q in result.belief_history["q_soc"]:
            assert q.shape == (SOC_LEVELS,)
            assert abs(q.sum() - 1.0) < 1e-5
            assert np.all(q >= 0)

    def test_tom_reliability_bounded(self):
        """ToM reliability should stay in [0, 1]."""
        from econet.simulation import run_tom_simulation
        env_data = generate_multi_day(num_days=2, seed=42)
        result = run_tom_simulation(
            env_data=env_data, num_days=2,
            policy_len=4, gamma=16.0,
            learn_B=False, verbose=False,
        )
        for r in result.belief_history["thermo_tom_reliability"]:
            assert 0.0 <= r <= 1.0
        for r in result.belief_history["battery_tom_reliability"]:
            assert 0.0 <= r <= 1.0


class TestCommunicationMetrics:
    def test_communication_metrics_from_tom_sim(self):
        from econet.simulation import run_tom_simulation
        env_data = generate_multi_day(num_days=1, seed=42)
        result = run_tom_simulation(
            env_data=env_data, num_days=1,
            policy_len=4, gamma=16.0,
            learn_B=False, verbose=False,
        )
        cm = compute_communication_metrics(result)
        assert isinstance(cm, CommunicationMetrics)
        assert cm.avg_comfort_entropy >= 0
        assert cm.avg_soc_entropy >= 0
        assert 0.0 <= cm.final_thermo_tom_reliability <= 1.0
        assert 0.0 <= cm.final_battery_tom_reliability <= 1.0

    def test_communication_metrics_no_belief_history(self):
        """Non-ToM simulation result should return zero metrics."""
        env_data = generate_multi_day(num_days=1, seed=42)
        result = run_no_hems(env_data)
        cm = compute_communication_metrics(result)
        assert cm.avg_comfort_entropy == 0.0
        assert cm.avg_soc_entropy == 0.0

    def test_belief_entropy_nontrivial(self):
        """Shared comfort beliefs should have non-zero entropy."""
        from econet.simulation import run_tom_simulation
        env_data = generate_multi_day(num_days=1, seed=42)
        result = run_tom_simulation(
            env_data=env_data, num_days=1,
            policy_len=4, gamma=16.0,
            learn_B=False, verbose=False,
        )
        cm = compute_communication_metrics(result)
        # Comfort beliefs have spread (room temp A matrix is soft)
        assert cm.avg_comfort_entropy > 0.01
        # SoC entropy may be very low (deterministic B → peaked posterior)
        # but should be non-negative
        assert cm.avg_soc_entropy >= 0.0


class TestFlatSimulationRegression:
    """Ensure existing simulations still work after Phase 3 additions."""

    def test_flat_simulation_still_works(self):
        from econet.simulation import run_simulation
        env_data = generate_multi_day(num_days=1, seed=42)
        result = run_simulation(
            env_data=env_data, num_days=1, policy_len=4,
            learn_B=False, verbose=False,
        )
        assert result.total_steps == 12
        assert result.total_cost >= 0

    def test_hierarchical_still_works(self):
        from econet.simulation import run_hierarchical_simulation
        env_data = generate_multi_day(num_days=2, seed=42)
        result = run_hierarchical_simulation(
            env_data=env_data, num_days=2, policy_len=4,
            gamma=16.0, learn_B=False, verbose=False,
        )
        assert result.total_steps == 24


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
