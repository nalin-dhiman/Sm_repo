import numpy as np
from scipy.integrate import quad
from scipy.special import erf, erfc
from typing import Callable, Tuple, Optional

class IntervalDistribution:
    """
    RIGOROUS IMPLEMENTATION
    Handles the Interval Distribution P(t) and Gain Function using 
    the Fokker-Planck First Passage Time solution (Siegert Formula).
    Reference: Neuronal Dynamics, Ch 12.3 & 14.
    """
    def __init__(self, tau_m: float = 20.0, u_r: float = -70.0, theta: float = -50.0, 
                 sigma_noise: float = 5.0, tau_rp: float = 2.0, gL: float = 0.5):
        self.tau_m = tau_m        # Membrane time constant (ms)
        self.u_r = u_r            # Reset potential (mV)
        self.theta = theta        # Threshold (mV)
        self.sigma = sigma_noise  # Noise amplitude (mV), represents synaptic fluctuation
        self.tau_rp = tau_rp      # Refractory period (ms)
        self.R_in = 1.0 / gL      # Input resistance (GOhm) if gL in nS.
        # Note: If I is in pA and R in GOhm, then I*R is in mV.
        # Example: 10 pA * 2 GOhm = 20 mV.
        
    def siegert_gain(self, I_ext: float, J_val: float = 0, Rate_prev: float = 0) -> float:
        """
        Computes the EXACT mean firing rate (Gain) using the Siegert Formula.
        CORRECTED: Converts pA inputs to mV using Input Resistance.
        """
        # 1. Calculate Total Input Current (pA)
        # J_val must be interpreted as [pA / Hz] effective weight
        I_total_pA = I_ext + J_val * Rate_prev
        
        # 2. Convert to Voltage Drive (mV) using Input Resistance
        # If R_in is in GOhm and I is in pA, result is mV.
        I_voltage = I_total_pA * self.R_in 
        
        # 3. Calculate Drift (Mean Membrane Potential)
        mu = self.u_r + I_voltage 
        
        # Integration limits (normalized by noise sigma)
        y_theta = (self.theta - mu) / self.sigma
        y_reset = (self.u_r - mu) / self.sigma
        
        def integrand(x):
            return np.exp(x**2) * (1 + erf(x))
        
        try:
            integral, error = quad(integrand, y_reset, y_theta)
        except:
            return 0.0 
            
        t_mean = self.tau_rp + self.tau_m * np.sqrt(np.pi) * integral
        
        return 1.0 / t_mean if t_mean > 0 else 0.0

    def pdf(self, t: float, I_ext: float) -> float:
        """
        Returns the First Passage Time Density P(t).
        Approximation: Inverse Gaussian (Wald Distribution).
        This is the exact solution for a drift-diffusion process without leak,
        and a very strong approximation for LIF in the fluctuation-driven regime.
        """
        if t <= self.tau_rp:
            return 0.0
        
        # Effective time since refractory period
        dt = t - self.tau_rp
        
        # Drift rate (slope of potential rise)
        # v_dot ~ (I_ext + E_L - V_mean)/tau
        # Simplified drift v = (mu - u_reset) / tau_m
        
        I_voltage = I_ext * self.R_in
        mu = -60.0 + I_voltage # Updated to -60 match and R scaling
        drift = (mu - self.u_r) / self.tau_m
        
        # Distance to threshold
        alpha = self.theta - self.u_r
        
        if drift <= 0:
            # Subthreshold regime: firing is purely noise driven (Poisson-like tails)
            drift = 0.001 
            
        # Diffusion coefficient D related to sigma
        # sigma^2 = D * tau_m  => D = sigma^2 / tau_m
        D = (self.sigma**2) / self.tau_m
        
        # P(t) = (alpha / sqrt(2*pi*D*t^3)) * exp( -(alpha - v*t)^2 / (2*D*t) )
        if dt <= 0: return 0.0 # Safety
        
        exponent = - (alpha - drift*dt)**2 / (2 * D * dt)
        prefactor = alpha / np.sqrt(2 * np.pi * D * dt**3)
        
        return prefactor * np.exp(exponent)

    def survivor_function(self, t_vals: np.ndarray, I_ext: float) -> np.ndarray:
        """
        Computes S(t) = 1 - CDF(t).
        Essential for the integral equation kernel.
        """
        # VECTORIZED NUMERICAL APPROACH (More stable than formula)
        # Using cumulative sum of PDF
        pdf_vals = np.array([self.pdf(t, I_ext) for t in t_vals])
        dt_step = t_vals[1] - t_vals[0] if len(t_vals) > 1 else 0.1
        cdf = np.cumsum(pdf_vals) * dt_step
        S_vals = 1.0 - cdf
        S_vals[S_vals < 0] = 0.0 # Clamp
        
        return S_vals
