
import numpy as np
import os
import sys
import scipy.sparse
import scipy.signal
from scipy.ndimage import gaussian_filter1d
from sklearn.linear_model import RidgeCV
from tqdm import tqdm
import argparse

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from src.data_loader import FlyWireClient
from src.simulation import AdExNetwork
from src.statistics import NetworkStatistics
from src.linear_id import LinearSystemID
from src.analysis_fixes import (
    assert_transfer_function_ok, 
    estimate_sub_input_conductances, 
    xcorr_peak
)
from src.meanfield import create_safe_transfer_function
from src.utils import log, save_provenance, get_file_checksum, atomic_json_dump
from src.validate import validate_rate_range, validate_psd_kwargs
import src.config as cfg

DATA_DIR = "outputs/data"
os.makedirs(DATA_DIR, exist_ok=True)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _get_sub_rate_trace(res, sub_nodes, dt=cfg.DT, smooth_sigma_ms=20.0):
    t_len = len(res['t'])
    dt_sec = dt / 1000.0
    
    if 'spikes' in res:
        spikes = res['spikes']
        rate_raw = np.zeros(t_len)
        nodes_set = set(sub_nodes)
        
        for (t_val, idxs) in spikes:
            i = int(round(t_val / dt))
            if i < t_len:
                count = len(nodes_set.intersection(idxs))
                rate_raw[i] += count
        rate_raw = rate_raw / (len(sub_nodes) * dt_sec)
    else:
        rate_raw = res['rate_raw']

    if smooth_sigma_ms > 0:
        sigma_samples = smooth_sigma_ms / dt
        rate = gaussian_filter1d(rate_raw, sigma_samples)
    else:
        rate = rate_raw
    return rate, rate_raw

def _compute_psd(trace, fs=10000.0, nperseg=2048):
    trace_detrend = trace - np.mean(trace)
    freqs, psd = scipy.signal.welch(trace_detrend, fs=fs, nperseg=int(nperseg))
    return freqs, psd

# ---------------------------------------------------------------------------
# Phases
# ---------------------------------------------------------------------------

def run_fig1_transfer_phi():
    log("Running Fig 1: Transfer Function...")
    bbox = AdExNetwork(n_neurons=1, **cfg.ADEX_PARAMS)
    I_grid = np.linspace(0, 2000, 41) 
    F_means = []
    F_sems = []
    
    seeds = cfg.N_SEEDS
    for I_in in tqdm(I_grid, desc="Phi(I)"):
        rates = []
        for s in range(seeds):
            bbox.reset_state(seed=s)
            res = bbox.run(300, cfg.DT, lambda t: I_in)
            rates.append(np.mean(res['rate_raw'][1000:])) # Last 200 ms
        F_means.append(np.mean(rates))
        F_sems.append(np.std(rates, ddof=1)/np.sqrt(seeds))
    
    assert_transfer_function_ok(I_grid, F_means)
    np.savez(f"{DATA_DIR}/fig1_data.npz", I_grid=I_grid, F_means=F_means, F_sems=F_sems)

def run_fig2_alpha_calibration(W):
    log("Running Fig 2: Alpha Calibration...")
    alphas = np.linspace(2, 10, 9)
    seeds = cfg.SEEDS
    
    res_rate_mean = []
    res_rate_sem = []
    res_sync = []
    
    N = W.shape[0]
    W_base = W
    seeds = cfg.SEEDS[:10] # Reduced for speed (Calibration only)
    
    for a_val in tqdm(alphas, desc="Alpha Sweep"):
        rates = []
        syncs = []
        W_eff = W_base * a_val
        for s in seeds:
            adex = AdExNetwork(n_neurons=N, **cfg.ADEX_PARAMS)
            adex.reset_state(seed=s)
            res = adex.run(1000, cfg.DT, lambda t: cfg.BIAS_BASE, W_matrix=W_eff)
            r_tr = res['rate_smooth']
            mean_r = np.mean(r_tr[5000:])
            rates.append(mean_r)
            if np.mean(r_tr) > 1.0:
                 cv = np.std(r_tr) / np.mean(r_tr)
            else:
                 cv = 0.0
            syncs.append(cv)
        res_rate_mean.append(np.mean(rates))
        res_rate_sem.append(np.std(rates, ddof=1)/np.sqrt(len(seeds)))
        res_sync.append(np.mean(syncs))
        
    np.savez(f"{DATA_DIR}/alpha_sweep_with_ci.npz", alphas=alphas, 
             rate_mean=res_rate_mean, rate_sem=res_rate_sem, sync_mean=res_sync)

def run_fig3_attractor(W, sub_nodes):
    log("Running Fig 3: Attractor Test...")
    N = W.shape[0]
    W_pA = W * cfg.ALPHA
    mask = np.zeros(N); mask[sub_nodes] = 1.0
    
    seeds = cfg.SEEDS
    end_states_pulse = []
    end_states_ctrl = []
    t_axis = None
    example_pulse = None
    example_ctrl = None
    
    for seed in tqdm(seeds, desc="Attractor"):
        # Pulse: Boosted to 100.0 pA for Robust Bistability
        adex = AdExNetwork(n_neurons=N, **cfg.ADEX_PARAMS)
        adex.reset_state(seed=seed)
        def inp_p(t): return cfg.BIAS_BASE + (100.0 * mask if 500<=t<700 else 0)
        def sig_s(t): return 0.0 if t > 1500 else cfg.ADEX_PARAMS['ou_sigma']
        res_p = adex.run(2000, cfg.DT, inp_p, W_matrix=W_pA, sigma_schedule=sig_s, record_per_neuron=True)
        r_p, _ = _get_sub_rate_trace(res_p, sub_nodes)
        
        # Control
        adex.reset_state(seed=seed)
        def inp_c(t): return cfg.BIAS_BASE
        res_c = adex.run(2000, cfg.DT, inp_c, W_matrix=W_pA, sigma_schedule=sig_s, record_per_neuron=True)
        r_c, _ = _get_sub_rate_trace(res_c, sub_nodes)
        
        # End State (1750-2000 ms)
        start_idx = int(1750/cfg.DT)
        end_states_pulse.append(np.mean(r_p[start_idx:]))
        end_states_ctrl.append(np.mean(r_c[start_idx:]))
        
        if seed == seeds[0]:
             example_pulse = r_p
             example_ctrl = r_c
             t_axis = res_p['t']

    np.savez(f"{DATA_DIR}/fig3_attractor.npz", 
             seeds=seeds, end_pulse=end_states_pulse, end_ctrl=end_states_ctrl,
             t=t_axis, ex_pulse=example_pulse, ex_ctrl=example_ctrl)

def run_embedding_analysis(W, sub_nodes):
    log("Running Fig 4: Embedding...")
    seeds = cfg.SEEDS[:10] # Reduced, effect was clearly significant with 20
    results = {'iso_power': [], 'full_power': [], 'iso_peak': [], 'full_peak': [], 'examples': {}}
    N = W.shape[0]; W_pA = W * cfg.ALPHA
    W_sub = W_pA[sub_nodes, :][:, sub_nodes]
    
    for seed in tqdm(seeds, desc="Embedding"):
        # Full
        adex = AdExNetwork(n_neurons=N, **cfg.ADEX_PARAMS)
        adex.reset_state(seed=seed)
        res_full = adex.run(cfg.DURATION_MS, cfg.DT, lambda t: cfg.BIAS_BASE, W_matrix=W_pA, record_per_neuron=True)
        r_full_smooth, r_full_raw = _get_sub_rate_trace(res_full, sub_nodes)
        
        # Isolated
        adex_iso = AdExNetwork(n_neurons=len(sub_nodes), **cfg.ADEX_PARAMS)
        adex_iso.reset_state(seed=seed)
        res_iso = adex_iso.run(cfg.DURATION_MS, cfg.DT, lambda t: cfg.BIAS_BASE, W_matrix=W_sub, record_per_neuron=True)
        r_iso_smooth, r_iso_raw = _get_sub_rate_trace(res_iso, list(range(len(sub_nodes))))
        
        # PSD (500-1500)
        start = int(cfg.WINDOW_START_MS/cfg.DT); end = int(cfg.WINDOW_END_MS/cfg.DT)
        f_f, p_f = _compute_psd(r_full_raw[start:end])
        f_i, p_i = _compute_psd(r_iso_raw[start:end])
        
        # Metrics (30-80Hz)
        mask = (f_f >= 30) & (f_f <= 80)
        results['full_power'].append(np.trapz(p_f[mask], f_f[mask]))
        results['iso_power'].append(np.trapz(p_i[mask], f_i[mask]))
        
        if seed == seeds[0]:
            results['examples'] = {'t': res_full['t'], 'full_trace': r_full_smooth, 'iso_trace': r_iso_smooth, 
                                   'psd_freqs': f_f, 'psd_full': p_f, 'psd_iso': p_i}
    np.savez(f"{DATA_DIR}/embedding_analysis.npz", **results)
    

def run_input_decomposition(W, sub_nodes):
    log("Running Fig 5: Input Decomp (Multi-seed)...")
    N = W.shape[0]; W_pA = W * cfg.ALPHA
    seeds = 20
    
    # Storage for multi-seed metrics
    all_lags = []
    all_peaks = []
    example_dat = None
    
    from src.analysis_fixes import estimate_population_ei_currents

    for seed in tqdm(range(seeds), desc="Input Seeds"):
        adex = AdExNetwork(n_neurons=N, seed=seed, **cfg.ADEX_PARAMS)
        res = adex.run(cfg.DURATION_MS, cfg.DT, lambda t: cfg.BIAS_BASE, W_matrix=W_pA, record_per_neuron=True)
        
        # Flatten spikes for analysis_fixes
        # (This is now handled by extract_spike_arrays inside estimate_*, but we keep it for estimate_sub_input_conductances for now if needed)
        # Actually estimate_sub_input_conductances expects raw res or fixed spikes.
        # Let's fix spikes just in case.
        # But wait, we modified analysis_fixes to have extract_spike_arrays.
        
        dec = estimate_sub_input_conductances(
            res,
            W_post_pre=W,
            sub_nodes=sub_nodes,
            alpha=cfg.ALPHA,
            dt_ms=cfg.DT,
            tau_syn_ms=cfg.ADEX_PARAMS['tau_syn']
        )
        
        lags, val_corr, best_lag, best_r = xcorr_peak(dec.rate_sub_hz, dec.g_ext_ns, dt_ms=cfg.DT)
        
        all_lags.append(best_lag)
        all_peaks.append(best_r)
        
        # Save first seed as example trace AND Phase Space Data
        if seed == 0:
            example_dat = {
                't': dec.t_ms,
                'g_ext': dec.g_ext_ns,
                'g_rec': dec.g_rec_ns,
                'rate': dec.rate_sub_hz,
                'lags': lags,
                'corr': val_corr
            }
            
            # Additional Phase Space Logic
            t_ei, I_exc, I_inh = estimate_population_ei_currents(
                res,
                W_post_pre=W,
                sub_nodes=sub_nodes,
                alpha=cfg.ALPHA,
                dt_ms=cfg.DT,
                tau_syn_ms=cfg.ADEX_PARAMS['tau_syn']
            )
            
            # Save ei_phase_space.npz
            np.savez(f"{DATA_DIR}/ei_phase_space.npz", t=t_ei, I_exc=I_exc, I_inh=I_inh)
            log(f"Saved {DATA_DIR}/ei_phase_space.npz")

    np.savez(f"{DATA_DIR}/input_decomposition.npz", 
             t=example_dat['t'], 
             g_ext=example_dat['g_ext'], 
             g_rec=example_dat['g_rec'], 
             rate=example_dat['rate'],
             lags=example_dat['lags'], 
             val_corr=example_dat['corr'], # Example correlation
             all_lags=np.array(all_lags),
             all_peaks=np.array(all_peaks),
             best_lag=np.mean(all_lags), # Mean lag across seeds
             best_r=np.mean(all_peaks)   # Mean r across seeds
             )

def run_linear_baseline(W, sub_nodes):
    log("Running Fig 6: Linear Baseline...")
    N = W.shape[0]; W_sub = W[sub_nodes, :][:, sub_nodes] * cfg.ALPHA
    
    # Train
    AdExNetwork(n_neurons=len(sub_nodes), seed=42, **cfg.ADEX_PARAMS) # Just warm up RNG if needed
    adex = AdExNetwork(n_neurons=len(sub_nodes), seed=42, **cfg.ADEX_PARAMS)
    steps = int(2000/cfg.DT)
    U_train = np.zeros(steps)
    seg_len = int(200/cfg.DT)
    val = 0
    rng = np.random.default_rng(42)
    for i in range(0, steps, seg_len):
        val = rng.uniform(0, 1) * 10.0
        U_train[i:i+seg_len] = val
        
    def inp_train(t): 
        idx = int(t/cfg.DT)
        u = U_train[idx] if idx < len(U_train) else 0
        return cfg.BIAS_BASE + u
        
    res = adex.run(2000, cfg.DT, inp_train, W_matrix=W_sub)
    r_train, _ = _get_sub_rate_trace(res, list(range(len(sub_nodes))))
    
    lid = LinearSystemID(dt_ms=cfg.DT, enforce_stability=True, max_rho=0.95)
    burn = int(300/cfg.DT)
    lid.fit(r_train, U_train, burn_in=burn)
    
    # Test
    adex.reset_state(seed=43)
    U_test = np.zeros(steps); U_test[int(500/cfg.DT):int(1500/cfg.DT)] = 10.0
    def inp_test(t):
        idx = int(t/cfg.DT)
        u = U_test[idx] if idx < len(U_test) else 0
        return cfg.BIAS_BASE + u
    res_test = adex.run(2000, cfg.DT, inp_test, W_matrix=W_sub)
    r_test, _ = _get_sub_rate_trace(res_test, list(range(len(sub_nodes))))
    
    pred_step = lid.predict_one_step(r_test, U_test)
    pred_open = lid.predict_open_loop(r_test[0], U_test) # Using correct method name
    
    np.savez(f"{DATA_DIR}/linear_baseline.npz", t=res_test['t'], r_test=r_test, pred_step=pred_step, pred_open=pred_open)

def run_sensitivity(W, sub_nodes):
    log("Running Fig 7: Sensitivity...")
    W_pA = W * cfg.ALPHA
    diffs = []
    seeds = 20; draws = 10 # 200 samples total
    ex_base = None; ex_pert = None; t_ax = None
    
    for seed in tqdm(range(seeds), desc="Sens Seeds"):
        adex = AdExNetwork(n_neurons=W.shape[0], **cfg.ADEX_PARAMS)
        adex.reset_state(seed=seed)
        res_b = adex.run(1500, cfg.DT, lambda t: cfg.BIAS_BASE+10.0, W_matrix=W_pA, record_per_neuron=True)
        r_b, _ = _get_sub_rate_trace(res_b, sub_nodes)
        
        for d in range(draws):
            W_pert = W_pA.copy()
            dat = W_pert.data
            nnz = len(dat)
            n_flip = int(0.3 * nnz)
            rng = np.random.default_rng(seed*1000 + d)
            idx = rng.choice(nnz, n_flip, replace=False)
            W_pert.data[idx] *= -1.0
            
            adex.reset_state(seed=seed)
            res_p = adex.run(1500, cfg.DT, lambda t: cfg.BIAS_BASE+10.0, W_matrix=W_pert, record_per_neuron=True)
            r_p, _ = _get_sub_rate_trace(res_p, sub_nodes)
            
            diff = np.mean(np.abs(r_b[5000:] - r_p[5000:]))
            diffs.append(diff)
            
            if seed==0 and d==0:
                ex_base=r_b; ex_pert=r_p; t_ax=res_b['t']
                
    np.savez(f"{DATA_DIR}/sensitivity.npz", diffs=diffs, ex_base=ex_base, ex_pert=ex_pert, t=t_ax)

def main():
    log("Forensic V6: Final Refinement Phase...")
    
    # run_fig1_transfer_phi()
    
    try:
        client = FlyWireClient(cache_dir=DATA_DIR)
        ids = client.get_cell_ids("Sm.*")
        W = client.get_connectivity_matrix(ids, ids, sparse_output=True)
    except:
        log("Loading W from local cache...", verbose=True)
        dat = np.load(f"{DATA_DIR}/W_counts_783_4463x4463.npz")
        W = scipy.sparse.csr_matrix((dat['data'], dat['indices'], dat['indptr']), shape=dat['shape'])
        
    # run_fig2_alpha_calibration(W)
    
    # Find Subcircuit
    candidates = NetworkStatistics.find_candidate_subcircuits(W, top_k=1)
    sub_nodes = candidates[0]['nodes']
    
    # run_fig3_attractor(W, sub_nodes)
    # run_embedding_analysis(W, sub_nodes)
    run_input_decomposition(W, sub_nodes)
    run_linear_baseline(W, sub_nodes)
    run_sensitivity(W, sub_nodes)
    
    # Provenance with Checksums
    files = [
        "fig1_data.npz", "alpha_sweep_with_ci.npz", "fig3_attractor.npz",
        "embedding_analysis.npz", "input_decomposition.npz",
        "linear_baseline.npz", "sensitivity.npz", "ei_phase_space.npz"
    ]
    checksums = {f: get_file_checksum(f"{DATA_DIR}/{f}") for f in files}
    
    save_provenance(f"outputs/reports/provenance.json", {
        "status": "Generated-V6",
        "phases": ["3-Attr-Refined", "4-Analysis-Patched"],
        "checksums": checksums,
        "config": str(cfg.__dict__)
    })
    log("Data generation complete.")

if __name__ == "__main__":
    main()
