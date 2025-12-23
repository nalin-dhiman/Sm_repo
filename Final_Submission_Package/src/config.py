import os

# API Configuration
FLYWIRE_API_TOKEN = os.getenv("FLYWIRE_API_TOKEN", "35475f2a738d87bc752c75a9e08ec27b") # Defaulting to user provided token, but best practice is env var
FLYWIRE_API_URL = "https://api.flywire.ai/api/v1" # Example URL, needs verification if used

# Simulation Parameters
DT = 0.1  # ms, integration time step
SIMULATION_DURATION = 1000.0  # ms

# AdEx Neuron Parameters (Drosophila Scale)
# Source: Gouwens & Wilson (2009) or similar fly interneuron data
# C ~ 10-20 pF, R_in ~ 1-2 GOhm (gL ~ 0.5-1 nS)
SM_TAU_M = 20.0       # Membrane time constant (ms)
SM_V_REST = -60.0     # Resting potential (mV)
SM_V_THRESH = -45.0   # Spike threshold (mV)
SM_C = 10.0           # Capacitance (pF)
SM_GL = 0.5           # Leak conductance (nS) -> tau = C/gL = 10/0.5 = 20ms

# Network Parameters
N_SM_TOTAL = 4000    # Total number of Sm neurons to simulate for biological realism
