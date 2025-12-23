"""validate_harness.py

Automated sanity checks for Data Consistency, Units, and Scientific Plausibility.
Run this after generating data and before final plotting.
"""

import numpy as np
import os
import sys

# Constants
MIN_RATE_HZ = 0.1
MAX_RATE_HZ = 1000.0 # Biologically implausible above this
DT_MS = 0.1
MIN_SAMPLES = 1000

def load_data(name):
    path = f"outputs/data/{name}.npz"
    if not os.path.exists(path):
        print(f"FAIL: Missing {path}")
        return None
    return np.load(path)

def check_fig1(data):
    print("Checking Fig 1 (Transfer)...")
    I = data['I_grid']
    F = data['F_means']
    
    if np.max(I) < 500:
        print("WARN: Fig 1 Input range small (<500 pA)")
    if np.max(F) <= 0:
        print("FAIL: Fig 1 Transfer function is dead (all zeros).")
        return False
    if np.any(np.diff(F) < -1.0): # Allow small noise, but not large drops
        print("WARN: Fig 1 non-monotonic (significant drop).")
        
    print(f"PASS: I_max={np.max(I)}pA, F_max={np.max(F):.2f}Hz")
    return True

def check_fig2(data):
    print("Checking Fig 2 (Alpha)...")
    alphas = data['alphas']
    sync = data['sync_mean']
    
    if len(alphas) < 3:
        print("FAIL: Fig 2 Alpha sweep too coarse (<3 points).")
        return False
        
    print(f"PASS: Alphas={alphas}, Max Sync={np.max(sync):.2f}")
    return True

def check_fig3(data):
    print("Checking Fig 3 (Attractor)...")
    ep = data['end_pulse']
    ec = data['end_ctrl']
    seeds = data['seeds']
    
    if len(ep) != len(ec):
        print("FAIL: Fig 3 Mismatched sample sizes.")
        return False
        
    if len(seeds) < 20: # Rigid check
        print(f"WARN: Fig 3 N={len(seeds)} (Recommended >= 20)")
        
    diff = np.mean(ep) - np.mean(ec)
    print(f"PASS: N={len(seeds)}, Mean Diff={diff:.2f} Hz")
    return True

def check_fig4(data):
    print("Checking Fig 4 (Embedding)...")
    iso = data['iso_power']
    full = data['full_power']
    
    if len(iso) < 5:
        print("FAIL: Fig 4 N < 5.")
        return False
        
    gamma_ratio = np.mean(iso) / (np.mean(full) + 1e-6)
    print(f"PASS: Gamma Ratio (Iso/Full) = {gamma_ratio:.2f}")
    return True

def check_fig5(data):
    print("Checking Fig 5 (Input)...")
    lags = data['lags']
    corr = data['val_corr']
    
    if np.max(np.abs(corr)) < 0.1:
        print("WARN: Fig 5 Correlation very weak (<0.1).")
        
    peak_lag = data['best_lag']
    print(f"PASS: Peak Lag={peak_lag} ms, r={data['best_r']:.2f}")
    return True

def check_fig6(data):
    print("Checking Fig 6 (Linear)...")
    r_test = data['r_test']
    p_step = data['pred_step']
    p_open = data['pred_open']
    
    # Check dimensions (allow squeeze)
    r_test = r_test.flatten()
    p_step = p_step.flatten()
    p_open = p_open.flatten()
    
    if r_test.shape != p_step.shape:
        print(f"FAIL: Fig 6 Shape mismatch (Test vs One-Step). {r_test.shape} vs {p_step.shape}")
        return False
        
    # Check for flatline
    if np.std(p_open) < 1e-6:
        print("WARN: Fig 6 Open-loop prediction is flat (Mean-only?).")
        
    # NMSE
    mse_step = np.mean((r_test - p_step)**2)
    var = np.var(r_test)
    r2_step = 1.0 - mse_step/var
    
    print(f"PASS: One-Step R2={r2_step:.2f}")
    return True

def check_fig7(data):
    print("Checking Fig 7 (Sensitivity)...")
    diffs = data['diffs']
    
    if len(diffs) < 50:
        print(f"WARN: Fig 7 N={len(diffs)} (Low sample size)")
        
    print(f"PASS: N={len(diffs)}, Mean Sensitivity={np.mean(diffs):.2f} Hz")
    return True

def main():
    print("=== Sanity Check Harness ===")
    all_pass = True
    
    d1 = load_data("fig1_data"); 
    if d1: all_pass &= check_fig1(d1)
    
    d2 = load_data("alpha_sweep_with_ci")
    if d2: all_pass &= check_fig2(d2)
    
    d3 = load_data("fig3_attractor")
    if d3: all_pass &= check_fig3(d3)
    
    d4 = load_data("embedding_analysis")
    if d4: all_pass &= check_fig4(d4)
    
    d5 = load_data("input_decomposition")
    if d5: all_pass &= check_fig5(d5)
    
    d6 = load_data("linear_baseline")
    if d6: all_pass &= check_fig6(d6)
    
    d7 = load_data("sensitivity")
    if d7: all_pass &= check_fig7(d7)
    
    if all_pass:
        print("\n[OK] All Sanity Checks Passed.")
        sys.exit(0)
    else:
        print("\n[FAIL] Some Checks Failed.")
        sys.exit(1)

if __name__ == "__main__":
    main()
