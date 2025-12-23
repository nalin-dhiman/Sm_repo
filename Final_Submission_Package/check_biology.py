import numpy as np
import sys
import os

# Ensure src is in path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))
import src.config as config

def check_biology():
    print("--- CHECKING BIOLOGICAL REALISM ---")
    
    # 1. The "Magic Resistor" Check
    # Fly interneurons: Rin ~ 1 GOhm (1000 MOhm).
    # Current params:
    # tau = 20ms, C = 10pF => gL = C/tau = 10/20 = 0.5 nS.
    # Rin = 1/gL = 1 / 0.5 nS = 2 GigaOhms.
    # So actually, my previous update to config (C=10, gL=0.5) implicitly set R = 2 GOHM!
    # This means a 10 pA input -> V = I*R = 10pA * 2GOhm = 20 mV.
    # This is massive. 
    # Rheobase (Current to reach threshold):
    # V_th - V_rest = -45 - (-60) = 15 mV.
    # I_rheo = 15mV / 2GOhm = 7.5 pA.
    
    print(f"[1] Input Resistance & Rheobase Check")
    print(f"    C = {config.SM_C} pF")
    print(f"    gL = {config.SM_GL} nS")
    R_in_GOhm = 1.0 / config.SM_GL
    print(f"    => R_in = {R_in_GOhm:.2f} GOhm (Fly Reality: ~1-2 GOhm)")
    print(f"    Verdict: The 'Magic Resistor' critique is PARTIALLY SOLVED by prior config update.")
    print(f"             However, code using 'u = -70 + I' assumes R=1 if units are consistent.")
    print(f"             In theory.py, 'mu = -60 + I_ext' implies I_ext is in mV (Voltage Drive).")
    print(f"             If I_ext is passed in pA, it must be multiplied by R.")
    
    # 2. The "Sign" Check
    print(f"\n[2] Connectivity Sign Check")
    import glob
    cache_files = glob.glob("cache_W_*.npy")
    if cache_files:
        W = np.load(cache_files[0])
        non_zeros = W[W != 0]
        prop_inh = np.sum(non_zeros < 0) / len(non_zeros)
        print(f"    Proportion Inhibitory: {prop_inh*100:.1f}%")
        
        if prop_inh > 0.5:
             print("    Verdict: SCENARIO B (Inhibitory Dominant).")
             print("             Pivot required: 'Integrator' -> 'Winner-Take-All'.")
        else:
             print("    Verdict: SCENARIO A (Excitatory Dominant).")
    else:
        print("    No cache found.")

if __name__ == "__main__":
    check_biology()
