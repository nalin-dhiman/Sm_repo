
import numpy as np
import matplotlib.pyplot as plt
import os
import sys
import json
from scipy.ndimage import gaussian_filter1d
from scipy import stats
import scipy
from tqdm import tqdm

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from src.plotting.paper_style import make_figure, save_figure, add_panel_label
from src.statistics import cohens_dz, paired_signed_rank, bootstrap_ci_paired
import src.config as cfg

DATA_DIR = "outputs/data"
FIG_DIR = "outputs/figures/paper"
os.makedirs(FIG_DIR, exist_ok=True)

def load_data(name):
    try:
        return np.load(f"{DATA_DIR}/{name}.npz", allow_pickle=True)
    except FileNotFoundError:
        print(f"Error: {name}.npz not found.")
        return None

def fig1_transfer_phi():
    print("Generating Fig 1 (Phi)...")
    d = load_data("fig1_data")
    if not d: return
    I_grid = d['I_grid']
    F_means = d['F_means']
    F_sems = d['F_sems']
    
    fig, ax = make_figure('single')
    ax.errorbar(I_grid, F_means, yerr=F_sems, fmt='ko', capsize=2, label=f'Data (N={cfg.N_SEEDS})', markersize=3)
    ax.plot(I_grid, F_means, 'r-', alpha=0.5, label='Interpolation')
    
    ax.set_xlabel("Input Current (pA)")
    ax.set_ylabel("Freq (Hz)")
    ax.legend(loc='upper left', frameon=False, fontsize=7)
    
    # Params Text - Move up to avoid line
    txt = f"C={cfg.ADEX_PARAMS['C']}pF\ngL={cfg.ADEX_PARAMS['gL']}nS"
    ax.text(0.95, 0.15, txt, transform=ax.transAxes, ha='right', va='bottom', fontsize=6)
    
    save_figure(fig, "Fig1_transfer_phi")

    save_figure(fig, "Fig1_transfer_phi")

    save_figure(fig, "Fig1_transfer_phi")

def fig2_connectivity_and_ei_phase():
    print("Generating Fig 2 (Connectivity + Phase)...")
    try:
        fname = f"{DATA_DIR}/W_counts_783_4463x4463.npz"
        if os.path.exists(fname):
             dat = np.load(fname)
             W = scipy.sparse.csr_matrix((dat['data'], dat['indices'], dat['indptr']), shape=dat['shape'])
        else:
             print("W cache not found, skipping Fig 2 Connectivity part.")
             W = None
             
        d_ei = load_data("ei_phase_space")
    except Exception as e:
        print(f"Error loading data for Fig 2: {e}")
        return

    # Horizontal Layout as requested
    # Double column width (183mm) for 3 panels.
    fig, axes = make_figure('double', height_mm=60, nrows=1, ncols=3)
    ax1, ax2, ax3 = axes
    
    # Define sparse ticks helper
    def set_sparse_ticks(ax, max_val):
        ticks = [0, int(max_val/2), int(max_val)]
        ax.set_xticks(ticks)
        ax.set_yticks(ticks)
        
    # A: Spy
    if W is not None:
        N = W.shape[0]
        ax1.spy(W, markersize=0.1, color='k', alpha=0.5)
        ax1.set_title(f"Adjacency (N={N})", fontsize=8)
        ax1.set_xlabel("Source")
        ax1.set_ylabel("Target")
        set_sparse_ticks(ax1, N)
        add_panel_label(ax1, "A")
        
        # B: Sorted
        indeg = np.array(W.sum(axis=1)).flatten()
        sort_idx = np.argsort(indeg)[::-1]
        W_sorted = W[sort_idx, :][:, sort_idx]
        
        ax2.spy(W_sorted, markersize=0.1, color='b', alpha=0.5)
        ax2.set_title("Sorted (In-Degree)", fontsize=8)
        ax2.set_xlabel("Source (Sorted)")
        set_sparse_ticks(ax2, N)
        ax2.set_yticklabels([]) # Hide Y labels for middle panel to save space? Or keep sparse.
        # Let's keep sparse y ticks but maybe remove label to avoid clutter if adjacent
        # ax2.set_ylabel("Target (Sorted)") 
        add_panel_label(ax2, "B")
    
    # C: Phase Space
    if d_ei:
        t = d_ei['t']
        I_exc = d_ei['I_exc']
        I_inh = d_ei['I_inh'] # Typically negative
        
        ax3.plot(-I_inh, I_exc, 'k-', lw=0.5, alpha=0.7)
        ax3.plot(-I_inh[0], I_exc[0], 'go', markersize=3, label='Start')
        ax3.plot(-I_inh[-1], I_exc[-1], 'ro', markersize=3, label='End')
        
        ax3.set_xlabel("Inh Current (pA)")
        ax3.set_ylabel("Exc Current (pA)")
        ax3.set_title("E-I Phase Space", fontsize=8)
        
        # Sparse ticks for Phase Space too?
        # Let matplotlib handle it but constrained layout helps.
        
        ax3.legend(fontsize=6, frameon=False, loc='lower right')
        add_panel_label(ax3, "C")
        
    save_figure(fig, "Fig2_connectivity_phase")

def fig3_alpha_calibration():
    print("Generating Fig 3 (Alpha)...")
    d = load_data("alpha_sweep_with_ci")
    if not d: return
    alphas = d['alphas']
    
    fig, axes = make_figure('double', height_mm=60, ncols=2)
    ax_rate, ax_sync = axes
    
    # A
    ax_rate.errorbar(alphas, d['rate_mean'], yerr=np.array(d['rate_sem'])*1.96, fmt='ko-', label='Rate')
    ax_rate.axvline(cfg.ALPHA, color='gray', linestyle='--', label=f'Target')
    ax_rate.set_xlabel(r"$\alpha$")
    ax_rate.set_ylabel("Rate (Hz)")
    ax_rate.legend(fontsize=6, frameon=False)
    add_panel_label(ax_rate, "A")
    
    # B
    ax_sync.plot(alphas, d['sync_mean'], 'bs-', label='CV')
    ax_sync.set_xlabel(r"$\alpha$")
    ax_sync.set_ylabel("CV (Sync)")
    add_panel_label(ax_sync, "B")
    
    save_figure(fig, "Fig3_alpha_calibration")

def fig4_attractor_strict():
    print("Generating Fig 4 (Strict)...")
    d = load_data("fig3_attractor")
    if not d: return
    
    t = d['t']
    ex_pulse = d['ex_pulse']
    ex_ctrl = d['ex_ctrl']
    end_pulse = d['end_pulse']
    end_ctrl = d['end_ctrl']
    
    fig, axes = make_figure('double', height_mm=70, ncols=2)
    ax_trace, ax_box = axes
    
    # A
    ax_trace.plot(t, ex_pulse, 'r', label='Pulse')
    ax_trace.plot(t, ex_ctrl, 'k', label='Control')
    ax_trace.axvspan(500, 700, color='gray', alpha=0.1)
    ax_trace.axvline(1500, color='b', linestyle='--', alpha=0.5)
    ax_trace.set_xlabel("Time (ms)")
    ax_trace.set_ylabel("Rate (Hz)")
    ax_trace.legend(fontsize=6, frameon=False, loc='upper right')
    add_panel_label(ax_trace, "A")
    
    # B
    stat, p_val = paired_signed_rank(end_pulse, end_ctrl)
    dz = cohens_dz(end_pulse, end_ctrl)
    
    ax_box.boxplot([end_ctrl, end_pulse], positions=[1, 2], widths=0.4, patch_artist=True, boxprops=dict(facecolor='white'))
    # Connect dots
    for i in range(len(end_ctrl)):
        ax_box.plot([1, 2], [end_ctrl[i], end_pulse[i]], 'k-', alpha=0.1, linewidth=0.5)
        
    ax_box.set_xticks([1, 2])
    ax_box.set_xticklabels(['Control', 'Pulse'])
    ax_box.set_ylabel("End Rate (Hz)")
    
    # Title Stats
    ax_box.set_title(f"p={p_val:.2e}, dz={dz:.2f}", fontsize=7)
    add_panel_label(ax_box, "B")
    
    save_figure(fig, "Fig4_attractor_strict")

def fig5_embedding():
    print("Generating Fig 5 (Embedding)...")
    d = load_data("embedding_analysis")
    if not d: return
    
    fig, axes = make_figure('double', height_mm=80, ncols=3)
    ax1, ax2, ax3 = axes
    
    ex = d['examples'].item()
    
    # A
    ax1.plot(ex['t'], ex['full_trace'], 'k', label='Emb')
    ax1.plot(ex['t'], ex['iso_trace'], 'r', label='Iso')
    ax1.set_xlim(500, 1500)
    ax1.set_xlabel("Time")
    ax1.set_ylabel("Hz")
    ax1.legend(fontsize=6, frameon=False)
    add_panel_label(ax1, "A")
    
    # B
    ax2.plot(ex['psd_freqs'], ex['psd_full'], 'k')
    ax2.plot(ex['psd_freqs'], ex['psd_iso'], 'r')
    ax2.set_xlim(0, 100)
    ax2.set_xlabel("Freq")
    ax2.axvspan(30, 80, color='y', alpha=0.1)
    add_panel_label(ax2, "B")
    
    # C
    iso = d['iso_power']
    full = d['full_power']
    stat, p = paired_signed_rank(iso, full)
    dz = cohens_dz(full, iso)
    
    ax3.boxplot([iso, full], positions=[1, 2], widths=0.4, patch_artist=True, boxprops=dict(facecolor='white'))
    for i in range(len(iso)):
        ax3.plot([1, 2], [iso[i], full[i]], 'k-', alpha=0.1, linewidth=0.5)
        
    ax3.set_xticklabels(['Iso', 'Emb'])
    ax3.set_ylabel("Gamma Power")
    ax3.set_title(f"p={p:.2e}, dz={dz:.2f}", fontsize=7)
    add_panel_label(ax3, "C")
    
    save_figure(fig, "Fig5_embedding")

def fig6_input():
    print("Generating Fig 6...")
    d = load_data("input_decomposition")
    if not d: return
    
    fig, axes = make_figure('double', height_mm=60, ncols=3)
    ax1, ax2, ax3 = axes
    
    t = d['t']
    mask = (t > 500) & (t < 1500)
    
    ax1.plot(t[mask], d['g_ext'][mask], 'tab:blue', label='Ext', linewidth=0.8)
    ax1.plot(t[mask], d['g_rec'][mask], 'tab:orange', label='Rec', linewidth=0.8)
    ax1.legend(fontsize=6, frameon=False, loc='best')
    ax1.set_ylabel("Conductance (nS)")
    add_panel_label(ax1, "A")
    
    # B: Cross-corr (N=20 mean)
    if 'val_corr' in d and d['val_corr'].ndim > 0:
        ax2.plot(d['lags'], d['val_corr'], 'k', label='Example')
        ax2.set_xlabel("Lag (ms)")
        ax2.set_title(f"Peak Lag: {d['best_lag']:.1f}ms", fontsize=8)
    
    # Adding Distribution Panel (C) if available
    if 'all_peaks' in d:
        ax3.hist(d['all_peaks'], bins=10, color='gray', edgecolor='k')
        ax3.set_xlabel("Peak Correlation")
        ax3.set_ylabel("Count")
        mean_r = np.mean(d['all_peaks'])
        ax3.axvline(mean_r, color='r', linestyle='--')
        ax3.set_title(f"Mean r={mean_r:.2f}", fontsize=8)
        add_panel_label(ax3, "C")

    save_figure(fig, "Fig6_input_decomposition")

def fig7_linear():
    print("Generating Fig 7...")
    d = load_data("linear_baseline")
    if not d: return
    
    fig, axes = make_figure('double', height_mm=60, ncols=2)
    ax1, ax2 = axes
    
    r_test = d['r_test']
    t = d['t'][:len(r_test)]
    pred_step = d['pred_step']
    pred_open = d['pred_open']
    mask = (t > 500) & (t < 1500)
    
    # A
    ax1.plot(t[mask], r_test[mask], 'k', label='True')
    ax1.plot(t[mask], pred_step[mask], 'r--', label='Model')
    ax1.set_title("One-Step Prediction", fontsize=8)
    ax1.legend(fontsize=6, frameon=False)
    add_panel_label(ax1, "A")
    
    # B
    ax2.plot(t[mask], r_test[mask], 'k')
    ax2.plot(t[mask], pred_open[mask], 'r--')
    ax2.set_title("Open-Loop (Model Failure)", fontsize=8)
    # Annotation
    ax2.text(0.5, 0.5, "Trajectory Divergence", transform=ax2.transAxes, ha='center', color='red', fontsize=8)
    add_panel_label(ax2, "B")
    
    save_figure(fig, "Fig7_linear_baseline")

def fig8_sensitivity():
    print("Generating Fig 8...")
    d = load_data("sensitivity")
    if not d: return
    
    fig, axes = make_figure('double', height_mm=60, ncols=2)
    ax1, ax2 = axes
    
    ax1.plot(d['t'], d['ex_base'], 'k', label='Base')
    ax1.plot(d['t'], d['ex_pert'], 'r', label='Pert')
    ax1.legend(fontsize=6, frameon=False)
    add_panel_label(ax1, "A")
    
    diffs = d['diffs']
    mu = np.mean(diffs)
    sem = np.std(diffs, ddof=1)/np.sqrt(len(diffs))
    
    ax2.hist(diffs, bins=15, color='gray')
    ax2.axvline(mu, color='r', linestyle='--')
    ax2.set_title(f"Mean Diff: {mu:.2f} $\pm$ {sem:.2f} Hz", fontsize=8)
    add_panel_label(ax2, "B")
    
    save_figure(fig, "Fig8_sign_sensitivity")

def main():
    fig1_transfer_phi()
    fig2_connectivity_and_ei_phase()
    fig3_alpha_calibration()
    fig4_attractor_strict()
    fig5_embedding()
    fig6_input()
    fig7_linear()
    fig8_sensitivity()

if __name__ == "__main__":
    main()
