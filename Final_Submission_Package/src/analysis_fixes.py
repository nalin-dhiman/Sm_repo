"""analysis_fixes.py

Small, battle-tested helper functions to prevent "pretty but wrong" figures.

These are designed to be copy/pasted into your pipeline.

Key ideas:
- Deterministic seeding that actually affects default_rng users.
- Sanity checks: detect flat traces, all-zeros transfer functions, etc.
- Input decomposition from spikes using the same synaptic decay rule as AdExNetwork.

Assumes:
- res['t'] is in ms, res['spikes']=(t_spikes_ms, neuron_idx)
- W is post x pre sparse matrix (counts or weights)
- alpha scales W into synaptic conductance increments (nS per synapse)
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Tuple, Optional

import numpy as np

try:
    from scipy import sparse
    from scipy.signal import correlate
except Exception:  # pragma: no cover
    sparse = None
    correlate = None


def seed_everything(seed: int) -> np.random.Generator:
    """Return a dedicated Generator and also seed NumPy's legacy RNG.

    Important: if your simulation uses np.random.default_rng(seed=...), pass this seed
    directly into that object; calling np.random.seed(...) is NOT enough.
    """
    seed = int(seed)
    np.random.seed(seed)
    return np.random.default_rng(seed)


def assert_nontrivial(name: str, x: np.ndarray, *, min_std: float = 1e-6, min_range: float = 1e-4) -> None:
    x = np.asarray(x)
    if not np.all(np.isfinite(x)):
        raise ValueError(f"{name} contains NaN/Inf")
    if float(np.std(x)) < min_std:
        raise ValueError(f"{name} is (near) constant: std={np.std(x):.3g}")
    if float(np.max(x) - np.min(x)) < min_range:
        raise ValueError(f"{name} has (near) zero range: range={(np.max(x)-np.min(x)):.3g}")


def assert_transfer_function_ok(I_grid: np.ndarray, rates_hz: np.ndarray, *, min_max_rate: float = 5.0) -> None:
    I_grid = np.asarray(I_grid)
    rates_hz = np.asarray(rates_hz)
    if len(I_grid) != len(rates_hz):
        raise ValueError("I_grid and rates must match")
    if np.max(rates_hz) < min_max_rate:
        raise ValueError(
            f"Transfer function looks dead: max rate={np.max(rates_hz):.3g} Hz. "
            "This is usually an input-unit mismatch or insufficient I range."
        )


@dataclass
class DecompositionResult:
    t_ms: np.ndarray
    rate_sub_hz: np.ndarray
    g_ext_ns: np.ndarray
    g_rec_ns: np.ndarray


def _spike_event_to_bins(t_spikes_ms: np.ndarray, dt_ms: float, T: int) -> np.ndarray:
    idx = np.floor(np.asarray(t_spikes_ms) / float(dt_ms)).astype(int)
    idx = idx[(idx >= 0) & (idx < T)]
    return idx


def estimate_sub_input_conductances(
    res: dict,
    W_post_pre,
    sub_nodes: np.ndarray,
    *,
    alpha: float,
    dt_ms: float,
    tau_syn_ms: float,
) -> DecompositionResult:
    """Estimate mean synaptic conductance onto a subcircuit from spikes.

    Uses the same update rule as AdExNetwork:
        g[t] = decay*g[t-1] + inc[t]
    where inc[t] is the (mean over sub neurons) conductance increment produced by spikes at t.

    Returns conductances in "nS-equivalent" units (whatever alpha produces). If alpha is in nS/synapse,
    these are nS.
    """
    if sparse is None:
        raise ImportError("scipy is required (sparse + correlate)")

    t = np.asarray(res["t"], dtype=float)
    T = len(t)
    t = np.asarray(res["t"], dtype=float)
    T = len(t)
    t_spikes_ms, spike_i = extract_spike_arrays(res)
    t_spikes_ms = np.asarray(t_spikes_ms, dtype=float)
    spike_i = np.asarray(spike_i, dtype=int)

    sub_nodes = np.asarray(sub_nodes, dtype=int)
    sub_mask = np.zeros(W_post_pre.shape[0], dtype=bool)
    sub_mask[sub_nodes] = True

    # Mean incoming weight into sub from each presyn neuron: mean_i W[i, j] for i in sub.
    # W is post x pre.
    W_sub = W_post_pre[sub_nodes, :]
    if sparse.issparse(W_sub):
        w_mean = np.asarray(W_sub.sum(axis=0)).ravel() / float(len(sub_nodes))
    else:
        w_mean = np.sum(W_sub, axis=0).ravel() / float(len(sub_nodes))

    w_mean = w_mean * float(alpha)

    # Bin spikes to integer timesteps.
    spike_t_idx = _spike_event_to_bins(t_spikes_ms, dt_ms, T)
    spike_i = spike_i[: len(spike_t_idx)]  # conservative alignment

    # Conductance increments per timestep.
    inc_ext = np.zeros(T, dtype=float)
    inc_rec = np.zeros(T, dtype=float)

    # Classify spikes into presyn groups.
    # Presyn in sub => recurrent contribution; otherwise => external (rest of network).
    presyn_is_sub = sub_mask[spike_i]

    # Accumulate increments (event-based, O(#spikes)).
    for ti, ni, is_sub in zip(spike_t_idx, spike_i, presyn_is_sub):
        if is_sub:
            inc_rec[ti] += w_mean[ni]
        else:
            inc_ext[ti] += w_mean[ni]

    decay = float(np.exp(-float(dt_ms) / float(tau_syn_ms)))

    g_ext = np.zeros(T, dtype=float)
    g_rec = np.zeros(T, dtype=float)
    for k in range(1, T):
        g_ext[k] = decay * g_ext[k - 1] + inc_ext[k]
        g_rec[k] = decay * g_rec[k - 1] + inc_rec[k]

    # Subcircuit firing rate in Hz from spikes (10 ms bins by default)
    # (If you already compute rate elsewhere, ignore this and supply your own.)
    bin_ms = 5.0
    bin_steps = max(1, int(round(bin_ms / dt_ms)))
    bins = np.arange(0, T + 1, bin_steps)

    sub_spike_idx = spike_t_idx[presyn_is_sub]
    counts, _ = np.histogram(sub_spike_idx, bins=bins)
    rate_sub = counts / (len(sub_nodes) * (bin_ms / 1000.0))

    # Expand rate_sub back to per-timestep by repetition for convenience.
    rate_sub_ts = np.repeat(rate_sub, bin_steps)[:T]

    return DecompositionResult(t_ms=t, rate_sub_hz=rate_sub_ts, g_ext_ns=g_ext, g_rec_ns=g_rec)


def xcorr_peak(x: np.ndarray, y: np.ndarray, *, dt_ms: float, max_lag_ms: float = 200.0) -> Tuple[np.ndarray, np.ndarray, float, float]:
    """Normalized cross-correlation of x vs y; returns (lags_ms, corr, best_lag_ms, best_r)."""
    if correlate is None:
        raise ImportError("scipy is required (signal.correlate)")

    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)

    x = (x - np.mean(x)) / (np.std(x) + 1e-12)
    y = (y - np.mean(y)) / (np.std(y) + 1e-12)

    corr = correlate(y, x, mode="full") / float(len(x))
    lags = np.arange(-len(x) + 1, len(x)) * float(dt_ms)

    mask = np.abs(lags) <= float(max_lag_ms)
    lags = lags[mask]
    corr = corr[mask]

    # Use minimum (strongest negative) by default; swap to argmax for positive.
    k = int(np.argmin(corr))
    return lags, corr, float(lags[k]), float(corr[k])

def extract_spike_arrays(res: dict) -> Tuple[np.ndarray, np.ndarray]:
    """Extract spikes into (t_ms, node_id) arrays."""
    if 'spikes' in res:
        spikes = res['spikes']
        if len(spikes) == 0:
            return np.array([]), np.array([])
            
        # Check if it's a tuple of arrays (t, i)
        if isinstance(spikes, tuple) and len(spikes) == 2:
             # Assume (times, indices) structure if elements are arrays
             if isinstance(spikes[0], (list, np.ndarray)) and isinstance(spikes[1], (list, np.ndarray)):
                  return np.asarray(spikes[0], dtype=float), np.asarray(spikes[1], dtype=int)
        
        # Determine if it's list of tuples or something else
        # AdExNetwork standard: list of (t, [indices])
        if isinstance(spikes, list):
             # Check first element structure
             if len(spikes) > 0 and isinstance(spikes[0], (list, tuple)) and len(spikes[0]) == 2:
                  # It is list of (t_val, idxs)
                  t_flat = []
                  i_flat = []
                  for (t_v, idxs) in spikes:
                       count = len(idxs) if isinstance(idxs, (list, np.ndarray)) else 1
                       if count > 0:
                           idx_list = idxs if isinstance(idxs, (list, np.ndarray)) else [idxs]
                           t_flat.extend([t_v] * count)
                           i_flat.extend(idx_list)
                  return np.array(t_flat), np.array(i_flat)
        
        # If we got here, format is unknown or empty?
        # Try brute force unpacking if it looks like (t_arr, i_arr) but wasn't tuple
        try:
             times, indices = spikes
             return np.asarray(times), np.asarray(indices)
        except:
             pass
             
    return np.array([]), np.array([])

def estimate_population_ei_currents(
    res: dict,
    W_post_pre,
    sub_nodes: np.ndarray,
    *,
    alpha: float,
    dt_ms: float,
    tau_syn_ms: float
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Estimate Excitatory and Inhibitory synaptic currents (rec) + External.
    Returns (t, I_exc, I_inh).
    Assumes W_post_pre is signed.
    Exc Current > 0 (Inward), Inh Current < 0 (Outward)? 
    Or Standard AdEx: I_syn = g * (V - E).
    Here we calculate 'Effective Current' I = g * weight.
    Since we don't track V here, we approximate or just sum "weighted spikes" (Current Injection Model).
    
    If alpha scales W to pA (Current based), then we just sum W.
    """
    t = np.asarray(res['t'])
    T = len(t)
    t_spikes, spike_i = extract_spike_arrays(res)
    
    # 1. Identify E and I inputs
    # We need to know if presynaptic neuron is E or I.
    # We infer from W signs.
    # Row sum? No. Column sum.
    # Signs are in W itself.
    
    # W is post x pre. W[:, j] is output of neuron j.
    # Check sign of output weights.
    # W_post_pre is sparse CSR/CSC.
    # Let's get sign of each neuron.
    N = W_post_pre.shape[1]
    
    # Inferred Signs
    # Sum of outgoing weights
    out_sum = np.array(W_post_pre.sum(axis=0)).flatten()
    # Or sample non-zeros.
    # If mixed mode (Dale violation), we treat each synapse.
    
    # We iterate spikes. For each spike from 'i', we add W[:, i] to targets.
    # We care about currents into 'sub_nodes'.
    # Mean current into sub_nodes.
    
    # Pre-calculate Mean Weight from `j` to `sub_nodes`.
    # W_sub = W[sub_nodes, :]
    W_sub = W_post_pre[sub_nodes, :]
    
    # mean_w_j = mean(W_sub[:, j]) for all j.
    # This is column sum of W_sub / N_sub.
    mean_w = np.array(W_sub.sum(axis=0)).flatten() / len(sub_nodes)
    mean_w *= float(alpha) # in nS or pA
    
    # Bin spikes
    spike_t_idx = _spike_event_to_bins(t_spikes, dt_ms, T)
    spike_i = spike_i[:len(spike_t_idx)]
    
    inc_exc = np.zeros(T)
    inc_inh = np.zeros(T)
    
    # Accumulate
    # We assume 'mean_w' contains the sign info.
    # Positive w -> Exc, Negative w -> Inh.
    
    # Optimized accumulation?
    # spike_w = mean_w[spike_i]
    # But we need to separate E and I.
    # E_spikes: spike_w > 0
    # I_spikes: spike_w < 0
    
    current_increments = mean_w[spike_i]
    is_exc = current_increments > 0
    
    # Add Exc
    np.add.at(inc_exc, spike_t_idx[is_exc], current_increments[is_exc])
    # Add Inh
    np.add.at(inc_inh, spike_t_idx[~is_exc], current_increments[~is_exc])
    
    # Exponential Decay
    decay = np.exp(-dt_ms / tau_syn_ms)
    I_exc = np.zeros(T)
    I_inh = np.zeros(T)
    
    # Filter
    # import scipy.signal.lfilter?
    # I[t] = d*I[t-1] + inc[t]
    # Filter coeff: b=[1], a=[1, -decay] ? 
    # y[n] = x[n] + decay*y[n-1] -> generic IIR.
    # scipy.signal.lfilter([1], [1, -decay], inc)
    
    if correlate is not None:
        from scipy.signal import lfilter
        I_exc = lfilter([1], [1, -decay], inc_exc)
        I_inh = lfilter([1], [1, -decay], inc_inh)
    else:
        for k in range(1, T):
            I_exc[k] = decay * I_exc[k - 1] + inc_exc[k]
            I_inh[k] = decay * I_inh[k - 1] + inc_inh[k]
            
    return t, I_exc, I_inh
