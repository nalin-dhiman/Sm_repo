import numpy as np
from typing import Tuple, Optional

class AdExNetwork:
    """
    Simulates a population of Adaptive Exponential Integrate-and-Fire neurons.
    """
    def __init__(self, n_neurons: int = 100, 
                 C: float = 10.0,    # pF 
                 gL: float = 0.5,    # nS
                 EL: float = -60.0,  # mV
                 VT: float = -45.0,  # mV
                 DeltaT: float = 2.0,# mV 
                 tau_w: float = 100.0, # ms 
                 a: float = 0.0,     # nS 
                 b: float = 5.0,     # pA 
                 Vr: float = -65.0,   # mV
                 tau_syn: float = 5.0 # ms (Synaptic decay time)
                 ):
        self.n_neurons = n_neurons
        # Parameters
        self.C = C
        self.gL = gL
        self.EL = EL
        self.VT = VT
        self.DeltaT = DeltaT
        self.a = a
        self.tau_w = tau_w
        self.b = b
        self.Vr = Vr
        self.tau_syn = tau_syn
        
        # State
        self.v = np.ones(n_neurons) * self.EL
        self.w = np.zeros(n_neurons)
        self.I_syn = np.zeros(n_neurons) # Synaptic Current State
        
    def step(self, dt: float, I_ext: np.ndarray, J_matrix: Optional[np.ndarray] = None, spikes_prev: Optional[np.ndarray] = None):
        """
        Integration step with Exponential Synapses.
        dI_syn/dt = -I_syn/tau_syn + Inputs
        """
        # 1. Synaptic Dynamics
        # If spikes arrived, add them to I_syn (instant rise)
        # We model this as a "kick" to the current variable.
        # kick = Sum(Weights) for firing presynaptic neurons.
        kick = 0.0
        if J_matrix is not None and spikes_prev is not None:
             # Spikes is bool array.
             # J_matrix is (N_post, N_pre).
             # We want W * spikes.
             kick = np.dot(J_matrix, spikes_prev) 
             # Note: standard "current based" synapse: current jumps by W, then decays.
             # Kick units: pA.
        
        # Euler for I_syn
        # dI = (-I/tau) * dt
        # But we also add the kick instantenously (or in the step)
        self.I_syn += (-self.I_syn / self.tau_syn) * dt + kick
        
        # 2. V and w Dynamics
        exp_term = self.gL * self.DeltaT * np.exp((self.v - self.VT) / self.DeltaT)
        
        # Total Current = External + Synaptic
        # I_ext is usually constant or noise, I_syn is the recurrent part
        dv = ( -self.gL * (self.v - self.EL) + exp_term - self.w + I_ext + self.I_syn ) / self.C
        
        dw = (self.a * (self.v - self.EL) - self.w) / self.tau_w
        
        self.v += dv * dt
        self.w += dw * dt
        
        # 3. Spike detection & Reset
        spikes = self.v > 0.0 # Peak threshold
        
        if np.any(spikes):
            self.v[spikes] = self.Vr
            self.w[spikes] += self.b
            
        return spikes

    def run(self, duration: float, dt: float, I_ext_func: callable, J_val: float = 0.0, W_matrix: Optional[np.ndarray] = None) -> Tuple[np.ndarray, np.ndarray]:
        """
        Run simulation.
        I_ext_func: function(t) -> scalar current.
        J_val: Global coupling strength (Mean field equivalent) if W_matrix is None.
        W_matrix: Explicit connectivity matrix (n x n). If provided, J_val scalar is IGNORED for connectivity construction.
        """
        steps = int(duration / dt)
        t_vals = np.arange(steps) * dt
        activity = np.zeros(steps)
        
        # Mean field connectivity matrix proxy
        if W_matrix is not None:
            # Ensure shape matches
            assert W_matrix.shape == (self.n_neurons, self.n_neurons), f"Shape mismatch: {W_matrix.shape} vs ({self.n_neurons},{self.n_neurons})"
            W = W_matrix
        else:
            # All-to-all mean field proxy
            W = np.ones((self.n_neurons, self.n_neurons)) * (J_val / self.n_neurons)
        
        spikes_prev = np.zeros(self.n_neurons)
        
        for i, t in enumerate(t_vals):
            I_in = I_ext_func(t)
            new_spikes = self.step(dt, I_in, W, spikes_prev)
            
            activity[i] = np.sum(new_spikes) / (self.n_neurons * dt * 0.001) # Hz
            spikes_prev = new_spikes.astype(float)
            
        return t_vals, activity

class LinearEffectomeBaseline:
    """
    The 'Effectome' model (Pospisil et al) assumes linear dynamics:
    dx/dt = Ax + Bu
    """
    def __init__(self, tau: float = 20.0, gain: float = 1.0):
        self.tau = tau
        self.gain = gain
        
    def run(self, duration: float, dt: float, input_val: np.ndarray, eigenvalue: float = -0.5) -> np.ndarray:
        """
        eigenvalue: Re(lambda) of the mode. 
        If stable: < 0.
        """
        steps = len(input_val)
        x = np.zeros(steps)
        
        # dx/dt = lambda*x + input
        # Explicit Euler
        # lambda_eff = -1/tau + connection_strength ??
        # Let's map eigenvalue directly to the linear term.
        
        for i in range(1, steps):
            curr_in = input_val[i-1]
            dx = (eigenvalue * x[i-1] + curr_in)
            x[i] = x[i-1] + dx * dt / self.tau # Scaling dt by tau for canonical form
            
        return x
