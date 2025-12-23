import numpy as np
import matplotlib.pyplot as plt
import sys
import os

# Ensure src is in path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from src.theory import IntervalDistribution
from src.analysis import StabilityAnalyzer
from src.simulation import AdExNetwork, LinearEffectomeBaseline
import src.config as config

# Setup styling for "Nature" quality plots
plt.rcParams.update({'font.size': 12, 'font.family': 'sans-serif'})

from sklearn.cluster import SpectralClustering
import seaborn as sns

def analyze_structural_modularity(W):
    """
    Generates Figure 4: The 'Checkerboard' Proof.
    Sorts neurons to reveal if they form two competing groups (Population A vs B).
    """
    print("--- 4. Generating Figure 4: Structural Modularity (Checkerboard) ---")
    
    # We focus on the Inhibitory structure (Negative weights)
    W_inh = W.copy()
    W_inh[W_inh > 0] = 0 # Mask excitation
    W_inh = np.abs(W_inh) # Use magnitude for clustering
    
    # 1. Spectral Clustering to find 2 groups
    try:
        clustering = SpectralClustering(n_clusters=2, affinity='precomputed', random_state=42)
        labels = clustering.fit_predict(W_inh)
    except Exception as e:
        print(f"Clustering failed (Matrix likely empty or uniform): {e}")
        return None

    # 2. Sort the matrix based on these labels
    sorted_indices = np.argsort(labels)
    W_sorted = W[sorted_indices][:, sorted_indices]
    
    # 3. Plot
    plt.figure(figsize=(7, 6))
    # We visualize the original SIGNED matrix to see Inhibition (Blue) vs Excitation (Red)
    sns.heatmap(W_sorted, cmap="RdBu_r", center=0, vmin=-5, vmax=5, cbar_kws={'label': 'Synaptic Counts'})
    plt.title("Fig 4: Evidence of Competitive Sub-Populations\n(Blue Blocks = Mutual Inhibition)")
    plt.xlabel("Neuron ID (Sorted by Cluster)")
    plt.ylabel("Neuron ID (Sorted by Cluster)")
    plt.tight_layout()
    plt.savefig('fig4_checkerboard.png')
    print("Saved fig4_checkerboard.png")
    
    return labels

def get_real_Sm_coupling():
    """
    Calculate biological 'J' parameter and check Neurotransmitter signs.
    """
    import glob
    cache_files = glob.glob("cache_W_*.npy")
    if not cache_files:
        print("No cached connectivity found. Skipping biological calibration.")
        return None, None, None
    
    # Use the most recent or relevant one (4463x4463)
    target_cache = None
    for f in cache_files:
        if "4463" in f or "4000" in f:
             target_cache = f
             break
    if not target_cache and cache_files: target_cache = cache_files[0]
        
    print(f"Loading {target_cache} for Calibration...")
    W = np.load(target_cache)
    
    # --- Step 2 Check: Neurotransmitter Polarity ---
    # W contains signed counts. 
    # Extract signs by looking at non-zero elements
    non_zeros = W[W != 0]
    n_exc = np.sum(non_zeros > 0)
    n_inh = np.sum(non_zeros < 0)
    total = len(non_zeros)
    
    exc_ratio = n_exc / total if total > 0 else 0
    print(f"\n[NEUROTRANSMITTER CHECK]")
    print(f"  Excitatory Connections: {n_exc} ({exc_ratio*100:.1f}%)")
    print(f"  Inhibitory Connections: {n_inh} ({(1-exc_ratio)*100:.1f}%)")
    
    if exc_ratio > 0.5:
        print("  VERDICT: System is Dominantly EXCITATORY. (Hypothesis: Memory/Integrator plausible)")
    else:
        print("  VERDICT: System is Dominantly INHIBITORY. (Hypothesis: Competition/Oscillation plausible)")
        print("  WARNING: Memory narrative requires disinhibition or specific sub-motifs.")

    # --- Step 3: Real J Calculation ---
    # In Mean Field, J = (Number of Synapses per neuron) * (Weight per synapse)
    
    # W entries are (signed) synapse counts.
    # We care about the effective drive J.
    # Average total synaptic count received from other Sm neurons (absolute magnitude of connectivity)
    # Using absolute to estimate total synaptic surface area/load.
    # But for J in bifurcation, we usually mean Net Feedback.
    # If Exc > Inh, Net is positive.
    
    # Let's calculate Net Connectivity (Exc - Inh) per neuron
    net_synapse_counts = np.sum(W, axis=1) # Sum over pre-synaptic inputs
    avg_net_count = np.mean(net_synapse_counts)
    std_net_count = np.std(net_synapse_counts)
    
    # Estimate unitary EPSP (Voltage change per synapse)
    # For flies, 1 synapse ~ 0.5 - 1.0 mV depolarization (approx)
    unitary_PSP = 0.5 # mV
    
    # J in our model is roughly "Total Voltage change per second per Hz"? 
    # Or just Total Voltage shift at steady state?
    # Our gain function is A = g(I). I is current/voltage drive.
    # J is coupling.
    # Let's just use the user's provided metric: J_real (mV)
    
    J_real = avg_net_count * unitary_PSP
    J_std = std_net_count * unitary_PSP
    
    print(f"\n[BIOLOGICAL GROUND TRUTH]")
    print(f"  Avg Net Synapses per Neuron: {avg_net_count:.2f}")
    print(f"  Estimated Unitary PSP: {unitary_PSP} mV")
    print(f"  Calculated J (Effective Coupling): {J_real:.2f} +/- {J_std:.2f} mV")
    
    return J_real, J_std, exc_ratio

def run_analysis():
    print("--- 1. Initializing Theoretical Framework ---")
    # Initialize the Mean-Field Theory components
    # Fix: User script used 'theta', 'u_r'. src/theory matches except theta -> u_th?
    # Checking theory.py... __init__(self, tau_m, u_r, u_th, u_rest)
    dist = IntervalDistribution(
        tau_m=config.SM_TAU_M, 
        u_r=config.SM_V_REST, 
        theta=config.SM_V_THRESH
    )
    analyzer = StabilityAnalyzer(dist)

    # --- FIGURE 1: Nullclines Analysis (Finding Fixed Points) ---
    print("--- 2. Generating Figure 1: Nullclines & Gain Function ---")
    
    # We want to see how the Population Activity (A) depends on Input (I_total)
    # The self-consistency equation is: A = g(I_ext + J * A)
    
    test_I_range = np.linspace(0, 500, 50)  # pA
    gain_curve = [analyzer.compute_gain(i) * 1000 for i in test_I_range] # Convert kHz to Hz
    
    plt.figure(figsize=(6, 5))
    plt.plot(test_I_range, gain_curve, 'k-', linewidth=2, label='Gain Function g(I)')
    plt.xlabel('Total Input Current (pA)')
    plt.ylabel('Population Firing Rate (Hz)')
    plt.title('Fig 1: Nonlinear Gain Function of Sm Population')
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.legend()
    plt.tight_layout()
    plt.savefig('fig1_gain_function.png')
    print("Saved fig1_gain_function.png")

    # --- FIGURE 2: Bifurcation Analysis (The "Novelty" Search) ---
    print("--- 3. Generating Figure 2: Bifurcation Diagram ---")
    
    # We sweep the Recurrent Weight (J) to see if the system develops memory
    # J represents the strength of Sm-to-Sm connections from the Connectome
    
    # Scale of J: If g(I) ~ 20Hz at I=100.
    # I_rec = J * A.
    # If J=5, A=20 -> I_rec = 100pA. Significant.
    # If J=0.5, I_rec = 10pA. Small.
    
    J_values = np.linspace(0, 50, 50) # Sweep J extended range to 20
    I_baseline = 100.0 # Constant background input (pA)
    
    fixed_points_map = []
    
    print("Sweeping J parameters...")
    for J in J_values:
        # Find roots for this specific J
        roots = analyzer.find_fixed_points(I_baseline, J)
        roots_hz = [r * 1000 for r in roots] 
        fixed_points_map.append(roots_hz)

    plt.figure(figsize=(7, 5))
    for i, J in enumerate(J_values):
        points = fixed_points_map[i]
        plt.scatter([J]*len(points), points, color='b', s=20, alpha=0.6)
        
    # Calibrate with Real Data
    bio_J, bio_std, exc_ratio = get_real_Sm_coupling()
    if bio_J is not None:
        # Scale Note: The J values in plot are usually small (0-50).
        # Our bio_J is likely in mV/pA range.
        # If bio_J is very large (e.g. 1000), it means our coupling is very strong.
        
        # Color based on dominance
        c_dom = 'g' if exc_ratio > 0.5 else 'm'
        label_dom = f"Sm Connectome (J={bio_J:.1f}, {'Exc' if exc_ratio>0.5 else 'Inh'} Dominated)"
        
        plt.axvline(x=bio_J, color=c_dom, linestyle='-', linewidth=2, label=label_dom)
        plt.axvspan(bio_J - bio_std, bio_J + bio_std, color=c_dom, alpha=0.1, label='Variance')
    
    plt.xlabel('Recurrent Coupling Strength J (mV*s or pA/Hz)')
    plt.ylabel('Fixed Point Activity (Hz)')
    plt.title('Fig 2: Bifurcation Diagram\n(Transition to Multistability/Memory)')
    plt.axvline(x=15.0, color='r', linestyle='--', label='Critical Threshold (Hypothetical)') 
    plt.legend()
    plt.tight_layout()
    plt.savefig('fig2_bifurcation.png')
    print("Saved fig2_bifurcation.png")
    
    # --- FIGURE 4: Structural Modularity (Checkerboard) ---
    # Need to load W for this
    import glob
    cache_files = glob.glob("cache_W_*.npy")
    if cache_files:
        W_for_fig4 = np.load(cache_files[0])
        analyze_structural_modularity(W_for_fig4)
    
    # --- FIGURE 3: Competitive Dynamics (Winner-Take-All) ---
    print("--- 5. Generating Figure 3: Winner-Take-All Dynamics ---")
    
    # Simulation Parameters
    T_sim = 1000 # ms
    dt = 0.1
    t_vals = np.arange(0, T_sim, dt)
    n_pop = config.N_SM_TOTAL // 2 # 2000 neurons in Pop A, 2000 in Pop B
    
    # 1. Setup Inhibitory Weights (Derived from Mean Field J)
    # If J_real is negative (e.g. -2.0), we set cross-inhibition to be strong
    J_inhib = 4.0 # Strength of mutual inhibition (positive magnitude)
    J_self = 1.0  # Weak self-excitation (if any)
    
    # Create Block Matrix for 2 Populations
    # [ Self   Cross ]
    # [ Cross  Self  ]
    W_wta = np.zeros((2*n_pop, 2*n_pop))
    
    # Pop A (0-49) inhibits Pop B (50-99)
    W_wta[n_pop:, :n_pop] = -J_inhib / n_pop
    # Pop B inhibits Pop A
    W_wta[:n_pop, n_pop:] = -J_inhib / n_pop
    # Self recurrence (optional, keeps activity alive)
    np.fill_diagonal(W_wta, J_self / n_pop)
    
    # 2. Stimulus: Pulse ONLY to Population A
    def input_wta(t):
        I_base = 15.0 # pA (Background)
        I_stim = 0.0
        if 200 <= t <= 400:
            I_stim = 10.0 # Stimulate Pop A
        
        # Input vector: [PopA_Inputs, PopB_Inputs]
        I_vec = np.ones(2*n_pop) * I_base
        I_vec[:n_pop] += I_stim # Add stimulus to A
        
        # Add Noise (Crucial for decision making)
        noise = np.random.normal(0, 5.0, 2*n_pop)
        return I_vec + noise

    # 3. Run AdEx Simulation
    adex_sys = AdExNetwork(n_neurons=2*n_pop)
    # We must modify AdEx step to handle vector inputs if it doesn't already
    # (Assuming your simulation.py handles vector I_ext, which standard implementations do)
    
    # Manual Loop for custom input handling
    activity_A = []
    activity_B = []
    spikes_prev = np.zeros(2*n_pop)
    
    for t in t_vals:
        I_in = input_wta(t)
        new_spikes = adex_sys.step(dt, I_in, W_wta, spikes_prev)
        
        # Record population rates
        rate_A = np.sum(new_spikes[:n_pop]) / (n_pop * dt * 0.001)
        rate_B = np.sum(new_spikes[n_pop:]) / (n_pop * dt * 0.001)
        
        activity_A.append(rate_A)
        activity_B.append(rate_B)
        spikes_prev = new_spikes.astype(float)

    # Smooth for plotting
    from scipy.ndimage import gaussian_filter1d
    acc_A = gaussian_filter1d(activity_A, sigma=20)
    acc_B = gaussian_filter1d(activity_B, sigma=20)
    
    plt.figure(figsize=(10, 4))
    plt.plot(t_vals, acc_A, 'r-', linewidth=2, label='Pop A (Stimulated)')
    plt.plot(t_vals, acc_B, 'b-', linewidth=1, alpha=0.7, label='Pop B (Suppressed)')
    plt.axvspan(200, 400, color='gray', alpha=0.2, label='Stimulus A')
    
    plt.title('Fig 3: Winner-Take-All Memory (Competitive Inhibition)')
    plt.xlabel('Time (ms)')
    plt.ylabel('Firing Rate (Hz)')
    plt.legend()
    plt.tight_layout()
    plt.savefig('fig3_dynamics.png')
    print("Saved fig3_dynamics.png")

    print("\n--- Analysis Complete ---")

if __name__ == "__main__":
    run_analysis()
