"""Custom EFE computation for sophisticated inference (Pitliya et al., 2025).

Replaces pymdp's standard `infer_policies` for the battery agent by
building a step-dependent B_effective[energy] at each planning step tau,
based on the phantom thermostat's predicted HVAC probability.

Standard pymdp uses a single B matrix for all planning steps.  Here,
B[energy] varies with tau because the phantom predicts different HVAC
probabilities across the planning horizon (e.g., high HVAC at tau=0
due to hot outdoor, low HVAC at tau=2 as temp drops).

The battery's EFE for each policy is:
    G(pi) = sum_tau [ utility_tau + ambiguity_tau ]
where utility = sum_m dot(qo_m, C_m) and ambiguity = -E_qs[H[P(o|s)]].
"""

import numpy as np

from .environment import (
    HVAC_KWH_PER_STEP, ENERGY_LEVELS,
)
from .generative_model import BATTERY_A_DEPS


def _softmax(x):
    """Numerically stable softmax."""
    x = np.asarray(x, dtype=np.float64)
    x_shifted = x - x.max()
    e = np.exp(x_shifted)
    return e / e.sum()


def build_effective_B2(p_active: float, delta: int,
                       n_energy: int = ENERGY_LEVELS) -> np.ndarray:
    """Build energy transition matrix from phantom HVAC prediction.

    When HVAC is active (prob = p_active), energy demand shifts up by
    `delta` bins.  When inactive (prob = 1 - p_active), energy stays.

    Parameters
    ----------
    p_active : float
        Probability thermostat runs HVAC (cool or heat) at this step.
    delta : int
        Number of energy bins HVAC adds (round(HVAC_KWH_PER_STEP / bin_size)).
    n_energy : int
        Number of energy levels.

    Returns
    -------
    B2 : np.ndarray, shape (n_energy, n_energy)
        Transition matrix P(energy' | energy).
    """
    B2 = np.zeros((n_energy, n_energy))
    for j in range(n_energy):
        j_shift = min(j + delta, n_energy - 1)
        B2[j_shift, j] += p_active
        B2[j, j] += (1.0 - p_active)
    # Normalise columns (should already be normalised, but be safe)
    for j in range(n_energy):
        col_sum = B2[:, j].sum()
        if col_sum > 0:
            B2[:, j] /= col_sum
    return B2


def _compute_expected_obs(A, qs_list, A_deps):
    """Compute expected observations: qo_m = A[m] contracted with relevant qs factors.

    Parameters
    ----------
    A : list of np.ndarray
        A matrices (likelihood mappings).
    qs_list : list of np.ndarray
        Current state beliefs, each shape (n_states_f,) — flat 1D.
    A_deps : list of list of int
        Which state factors each modality depends on.

    Returns
    -------
    qo : list of np.ndarray
        Expected observations for each modality.
    """
    qo = []
    for m in range(len(A)):
        Am = np.asarray(A[m], dtype=np.float64)
        deps = A_deps[m]

        if len(deps) == 1:
            # Simple: qo_m = A[m] @ qs[deps[0]]
            qo_m = Am @ qs_list[deps[0]]
        else:
            # Multi-factor: contract over all dependent state factors
            # A[m] has shape (n_obs, n_s1, n_s2, ...)
            # We need to contract with outer product of relevant qs factors
            result = Am
            for ax, f_idx in enumerate(deps):
                # Contract axis (ax+1, since axis 0 is obs) with qs[f_idx]
                # After each contraction, the contracted axis disappears,
                # so we always contract axis 1
                result = np.tensordot(result, qs_list[f_idx], axes=([1], [0]))
            qo_m = result
        # Ensure normalised
        total = qo_m.sum()
        if total > 0:
            qo_m = qo_m / total
        qo.append(qo_m)
    return qo


def _compute_ambiguity(A, qs_list, A_deps):
    """Compute ambiguity: E_qs[H[P(o|s)]] for each modality.

    Ambiguity = expected entropy of the likelihood under the state posterior.
    Lower ambiguity means the agent can make more precise predictions.

    Returns
    -------
    total_ambiguity : float
        Sum of ambiguity across all modalities.
    """
    total = 0.0
    for m in range(len(A)):
        Am = np.asarray(A[m], dtype=np.float64)
        deps = A_deps[m]

        if len(deps) == 1:
            # For each state s: H[P(o|s)] = -sum_o P(o|s) log P(o|s)
            f_idx = deps[0]
            qs_f = qs_list[f_idx]
            n_states = Am.shape[1]
            for s in range(n_states):
                p_o_given_s = Am[:, s]
                H_s = -np.sum(p_o_given_s * np.log(p_o_given_s + 1e-16))
                total += qs_f[s] * H_s
        else:
            # Multi-factor: iterate over all state combinations
            # For efficiency, we reshape and compute vectorised
            n_obs = Am.shape[0]
            # Compute H[P(o|s1,s2,...)] for each state combination
            # Then weight by joint qs
            # Flatten A to (n_obs, n_joint_states) and joint qs
            shape = Am.shape[1:]  # (n_s1, n_s2, ...)
            Am_flat = Am.reshape(n_obs, -1)
            # Joint qs via outer product
            joint_qs = qs_list[deps[0]]
            for f_idx in deps[1:]:
                joint_qs = np.outer(joint_qs, qs_list[f_idx]).flatten()
            # H for each joint state
            for j in range(Am_flat.shape[1]):
                p_o = Am_flat[:, j]
                H_j = -np.sum(p_o * np.log(p_o + 1e-16))
                total += joint_qs[j] * H_j

    return total


def compute_sophisticated_efe(
    A, B, C, qs, policies,
    phantom_sequence,
    A_deps=None,
    energy_factor_idx=2,
    hvac_delta_bins=2,
    gamma=16.0,
    tou_schedule=None,
):
    """Compute EFE for battery policies with step-dependent B[energy] and TOU.

    At each planning step tau, B[energy_factor_idx] is replaced with
    B_effective_tau computed from phantom_sequence[tau] (predicted
    thermostat HVAC probability).  When tou_schedule is provided,
    B[1] (TOU transitions) is also replaced per-step with deterministic
    schedule knowledge, giving the sophisticated agent full horizon
    awareness of both HVAC activity AND tariff changes.

    Parameters
    ----------
    A : list of jnp.ndarray
        Likelihood matrices from the battery's generative model.
    B : list of jnp.ndarray
        Transition matrices from the battery's generative model.
    C : list of jnp.ndarray
        Preference vectors (with batch dim).
    qs : list of jnp.ndarray
        Current posterior beliefs, each shape (batch=1, T=1, n_states).
    policies : jnp.ndarray
        Policy array, shape (n_policies, T, n_factors).
    phantom_sequence : list of np.ndarray
        [P(hvac)_0, ..., P(hvac)_{T-1}], each shape (3,) = [P(cool), P(heat), P(off)].
    A_deps : list of list of int, optional
        A-matrix dependencies. Defaults to BATTERY_A_DEPS.
    energy_factor_idx : int
        Index of the energy factor in state factors (default 2).
    hvac_delta_bins : int
        Energy bins shifted by HVAC activity (default 2).
    gamma : float
        Policy precision (inverse temperature).
    tou_schedule : list of int, optional
        Per-step TOU state [tou_0, ..., tou_{T-1}] for the planning horizon.
        Each entry is 0 (off-peak) or 1 (peak).  When provided, B[1] is
        replaced at each tau with a deterministic transition to the known
        TOU state, enabling full-horizon tariff awareness.

    Returns
    -------
    q_pi : np.ndarray, shape (n_policies,)
        Policy posterior.
    neg_efe : np.ndarray, shape (n_policies,)
        Negative EFE (higher = better).
    """
    if A_deps is None:
        A_deps = BATTERY_A_DEPS

    # Convert from JAX/batched to plain numpy, squeezing batch dimension
    A_np = [np.asarray(a, dtype=np.float64).squeeze(0) for a in A]
    B_np = [np.asarray(b, dtype=np.float64).squeeze(0) for b in B]
    C_np = [np.asarray(c, dtype=np.float64).flatten() for c in C]
    policies_np = np.asarray(policies)

    n_policies = policies_np.shape[0]
    policy_len = policies_np.shape[1]
    G = np.zeros(n_policies)

    # Pre-build per-step TOU transition matrices if schedule provided
    tou_B1_per_step = None
    if tou_schedule is not None:
        tou_B1_per_step = []
        for tau in range(policy_len):
            t_idx = min(tau, len(tou_schedule) - 1)
            next_tou = tou_schedule[t_idx]
            B1_tau = np.zeros((2, 2))
            B1_tau[next_tou, :] = 1.0  # deterministic: next TOU is known
            tou_B1_per_step.append(B1_tau)

    for pi_idx in range(n_policies):
        policy = policies_np[pi_idx]  # shape (T, n_factors)

        # Initialise local state beliefs from current posterior
        qs_local = [np.asarray(q[0, 0, :], dtype=np.float64).copy() for q in qs]

        for tau in range(policy_len):
            # Battery's action at this planning step (controlled factor 0)
            a_f = int(policy[tau, 0])

            # Phantom prediction for this step
            p_idx = min(tau, len(phantom_sequence) - 1)
            p_phantom = phantom_sequence[p_idx]
            p_active = float(p_phantom[0] + p_phantom[1])  # P(cool) + P(heat)

            # Build B_effective for energy factor
            B2_eff = build_effective_B2(p_active, hvac_delta_bins)

            # Predict next state for each factor
            # Factor 0 (SoC): controlled by battery action
            B0 = B_np[0]  # shape (5, 5, 3)
            qs_local[0] = B0[:, :, a_f] @ qs_local[0]

            # Factor 1 (TOU): per-step schedule or static
            if tou_B1_per_step is not None:
                qs_local[1] = tou_B1_per_step[tau] @ qs_local[1]
            else:
                B1 = B_np[1]  # shape (2, 2, 1)
                qs_local[1] = B1[:, :, 0] @ qs_local[1]

            # Factor 2 (energy): SI-modified transition
            qs_local[2] = B2_eff @ qs_local[2]

            # Normalise beliefs
            for f in range(len(qs_local)):
                total = qs_local[f].sum()
                if total > 0:
                    qs_local[f] /= total

            # Expected observations
            qo = _compute_expected_obs(A_np, qs_local, A_deps)

            # Utility: sum_m dot(qo_m, C_m)
            utility = sum(
                np.dot(qo[m], C_np[m]) for m in range(len(A_np))
            )

            # Information gain = H[q(o|pi)] - E_qs[H[P(o|s)]]
            # H[q(o|pi)]: entropy of predicted observations
            H_qo = 0.0
            for m in range(len(A_np)):
                qo_m = qo[m]
                H_qo -= np.sum(qo_m * np.log(qo_m + 1e-16))

            # E_qs[H[P(o|s)]]: expected ambiguity under state beliefs
            ambiguity = _compute_ambiguity(A_np, qs_local, A_deps)

            info_gain = H_qo - ambiguity

            # neg_G = utility + info_gain (matches pymdp EFE decomposition)
            G[pi_idx] += utility + info_gain

    # G is now "neg EFE" (higher = better policy)
    q_pi = _softmax(gamma * G)
    return q_pi, G


