"""Theory of Mind (ToM) module for belief sharing between agents.

Implements Friston et al. (2023) "Federated inference and belief sharing":
- Agents share posterior beliefs q(s), not raw states
- Received posteriors become observations via auditory A matrices
- ToM as meta-inference: Bayesian filter tracks other agent's reliability
- Consensus emerges from shared posteriors

Classes:
    BatteryToM — Thermostat's model of battery: tracks q(SoC) over time
    ThermostatToM — Battery's model of thermostat: tracks q(comfort) over time

Functions:
    belief_to_comfort — Project room temp posterior to 5-level comfort
    belief_to_obs — Convert posterior to discrete observation (MAP estimate)
"""

import numpy as np


# ======================================================================
# Belief projection utilities
# ======================================================================

COMFORT_LEVELS = 5  # COLD, COOL, COMFY, WARM, HOT


def belief_to_comfort(q_room: np.ndarray, target_idx: int) -> np.ndarray:
    """Project TEMP_LEVELS-level room temp posterior to 5-level comfort belief.

    Bins (relative to target, in 2°C-bin indices):
      0: COLD    (< target-2 bins = < target-4°C)
      1: COOL    (target-2 to target-1 bins = -4°C to -2°C)
      2: COMFY   (target-1 to target+1 bins = ±2°C)
      3: WARM    (target+1 to target+2 bins = +2°C to +4°C)
      4: HOT     (> target+2 bins = > target+4°C)

    Parameters
    ----------
    q_room : np.ndarray, shape (TEMP_LEVELS,)
        Posterior belief over room temperature states.
    target_idx : int
        Target temperature bin index.

    Returns
    -------
    np.ndarray, shape (5,)
        Comfort belief — proper probability distribution summing to 1.0.
    """
    bins = [target_idx - 2, target_idx - 1, target_idx + 1, target_idx + 2]
    q_comfort = np.zeros(COMFORT_LEVELS)
    for i, p in enumerate(q_room):
        bin_idx = np.digitize(i, bins)  # 0..4
        q_comfort[bin_idx] += p
    # Ensure valid distribution (handles edge cases from numerical noise)
    total = q_comfort.sum()
    if total > 0:
        q_comfort /= total
    else:
        q_comfort[:] = 1.0 / COMFORT_LEVELS
    return q_comfort


def belief_to_obs(q: np.ndarray) -> int:
    """Convert posterior to discrete observation (MAP estimate).

    Parameters
    ----------
    q : np.ndarray
        Posterior belief vector.

    Returns
    -------
    int
        Index of the most probable state (argmax).
    """
    return int(np.argmax(q))


# ======================================================================
# Helper matrices for ToM filters
# ======================================================================

def _near_identity(n: int, sigma: float = 0.5) -> np.ndarray:
    """Near-identity likelihood: peaked at diagonal with Gaussian spread.

    P(obs=o | state=s) ∝ exp(-0.5 * ((o-s)/sigma)^2)
    Normalized over obs dimension.
    """
    A = np.zeros((n, n))
    for s in range(n):
        for o in range(n):
            A[o, s] = np.exp(-0.5 * ((o - s) / sigma) ** 2)
        A[:, s] /= A[:, s].sum()
    return A


def _persistence_matrix(n: int, stay_prob: float = 0.7) -> np.ndarray:
    """Simple persistence transition model: P(s'|s).

    High probability of staying in same state, with small drift to neighbors.
    """
    B = np.zeros((n, n))
    for s in range(n):
        B[s, s] = stay_prob
        drift = (1.0 - stay_prob) / 2.0
        if s > 0:
            B[s - 1, s] = drift
        else:
            B[s, s] += drift
        if s < n - 1:
            B[s + 1, s] = drift
        else:
            B[s, s] += drift
    return B


# ======================================================================
# Theory of Mind Bayesian filters
# ======================================================================

class BatteryToM:
    """Thermostat's model of battery: tracks q(SoC) over time.

    This is the ToM layer: the thermostat doesn't just receive battery's
    belief — it MODELS the battery's belief dynamics. It asks:
    "Given what the battery told me and what I know about the world,
     what do I think the battery's SoC trajectory will be?"

    Parameters
    ----------
    n_states : int
        Number of SoC states (default 5).
    sigma : float
        Auditory likelihood spread (higher = noisier communication model).
    stay_prob : float
        Persistence probability in transition model.
    alpha : float
        Learning rate for reliability EMA.
    """

    def __init__(self, n_states: int = 5, sigma: float = 0.5,
                 stay_prob: float = 0.7, alpha: float = 0.1):
        self.n_states = n_states
        self.q_state = np.ones(n_states) / n_states  # uniform prior
        self.A_aud = _near_identity(n_states, sigma=sigma)
        self.B = _persistence_matrix(n_states, stay_prob=stay_prob)
        self.reliability = 0.5  # starts uncertain
        self.alpha = alpha
        self.history = []  # track q_state over time

    def update(self, received_belief: np.ndarray, received_obs: int):
        """Bayesian belief update using received belief as observation.

        Parameters
        ----------
        received_belief : np.ndarray, shape (n_states,)
            Full q(SoC) vector from battery.
        received_obs : int
            argmax of received_belief (for likelihood lookup).
        """
        # Predict: q(s') = B @ q(s)
        self.q_state = self.B @ self.q_state

        # Update: q(s') ∝ P(obs | s') * q(s')
        likelihood = self.A_aud[received_obs]
        self.q_state *= likelihood
        total = self.q_state.sum()
        if total > 0:
            self.q_state /= total
        else:
            self.q_state = np.ones(self.n_states) / self.n_states

        # Track reliability: how well does received match our prediction?
        prediction_error = 1.0 - np.dot(received_belief, self.q_state)
        prediction_error = np.clip(prediction_error, 0.0, 1.0)
        self.reliability = (1 - self.alpha) * self.reliability + self.alpha * (1.0 - prediction_error)
        self.reliability = np.clip(self.reliability, 0.0, 1.0)

        self.history.append(self.q_state.copy())

    def get_gated_belief(self, prior: np.ndarray) -> np.ndarray:
        """Reliability-gated belief (GatedToM pattern).

        Parameters
        ----------
        prior : np.ndarray
            Fallback prior distribution (e.g. uniform).

        Returns
        -------
        np.ndarray
            Blended belief: ρ * q_state + (1-ρ) * prior
        """
        return self.reliability * self.q_state + (1 - self.reliability) * prior

    def reset(self):
        """Reset to uniform prior."""
        self.q_state = np.ones(self.n_states) / self.n_states
        self.reliability = 0.5
        self.history = []


class ThermostatToM:
    """Battery's model of thermostat: tracks q(comfort) over time.

    Same structure as BatteryToM but with 5 comfort states
    (COLD, COOL, COMFY, WARM, HOT).

    Parameters
    ----------
    n_states : int
        Number of comfort states (default 5).
    sigma : float
        Auditory likelihood spread.
    stay_prob : float
        Persistence probability in transition model.
    alpha : float
        Learning rate for reliability EMA.
    """

    def __init__(self, n_states: int = COMFORT_LEVELS, sigma: float = 0.5,
                 stay_prob: float = 0.7, alpha: float = 0.1):
        self.n_states = n_states
        self.q_state = np.ones(n_states) / n_states  # uniform prior
        self.A_aud = _near_identity(n_states, sigma=sigma)
        self.B = _persistence_matrix(n_states, stay_prob=stay_prob)
        self.reliability = 0.5
        self.alpha = alpha
        self.history = []

    def update(self, received_belief: np.ndarray, received_obs: int):
        """Bayesian belief update using received comfort belief as observation.

        Parameters
        ----------
        received_belief : np.ndarray, shape (n_states,)
            Full q(comfort) vector from thermostat.
        received_obs : int
            argmax of received_belief.
        """
        self.q_state = self.B @ self.q_state

        likelihood = self.A_aud[received_obs]
        self.q_state *= likelihood
        total = self.q_state.sum()
        if total > 0:
            self.q_state /= total
        else:
            self.q_state = np.ones(self.n_states) / self.n_states

        prediction_error = 1.0 - np.dot(received_belief, self.q_state)
        prediction_error = np.clip(prediction_error, 0.0, 1.0)
        self.reliability = (1 - self.alpha) * self.reliability + self.alpha * (1.0 - prediction_error)
        self.reliability = np.clip(self.reliability, 0.0, 1.0)

        self.history.append(self.q_state.copy())

    def get_gated_belief(self, prior: np.ndarray) -> np.ndarray:
        """Reliability-gated belief."""
        return self.reliability * self.q_state + (1 - self.reliability) * prior

    def reset(self):
        """Reset to uniform prior."""
        self.q_state = np.ones(self.n_states) / self.n_states
        self.reliability = 0.5
        self.history = []
