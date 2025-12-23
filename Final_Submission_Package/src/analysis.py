import numpy as np
from scipy.optimize import fsolve, root
from typing import List, Tuple, Callable, Dict
from .theory import IntervalDistribution

class StabilityAnalyzer:
    """
    Analyzes the stability of the population dynamics.
    Focuses on finding fixed points A0 such that A0 = g(I_ext + J*A0).
    """
    def __init__(self, interval_dist: IntervalDistribution):
        self.dist = interval_dist
        
    def compute_gain(self, I_val: float) -> float:
        """
        Computes the steady state firing rate A0 using the Siegert Formula.
        """
        # Checks if dist supports siegert_gain (it does now)
        if hasattr(self.dist, 'siegert_gain'):
            return self.dist.siegert_gain(I_val)
            
        # Fallback to integration (legacy)
        limit = 200.0
        s_vals = np.linspace(0, limit, 1000)
        dt = s_vals[1] - s_vals[0]
        S_vals = self.dist.survivor_function(s_vals, I_val)
        mean_isi = np.sum(S_vals) * dt
        if mean_isi == 0: return 0.0
        return 1.0 / mean_isi

    def get_gain_function_curve(self, I_range: np.ndarray) -> np.ndarray:
        return np.array([self.compute_gain(i) for i in I_range])

    def find_fixed_points(self, I_ext: float, J: float) -> List[float]:
        """
        Finds solutions A* to A* = g(I_ext + J * A*).
        """
        # Define the function F(A) = A - g(I_ext + J*A)
        # Roots of F(A) = 0 are fixed points.
        
        def func(A):
            I_total = I_ext + J * A
            # Clip negative A? A should be >= 0
            if A < 0: return A # simple linear return to guide solver back to 0
            return A - self.compute_gain(I_total)
        
        # We scan a range of A to find good initial guesses
        # Max rate is usually 1/refractory. Say 0 to 0.2 (200Hz in ms^-1)
        a_guesses = np.linspace(0, 0.2, 10) 
        roots = set()
        
        for ag in a_guesses:
            sol = fsolve(func, ag)
            if sol[0] >= 0:
                # verify it's a root
                if abs(func(sol[0])) < 1e-4:
                    roots.add(round(sol[0], 5)) # dedup
                    
        return sorted(list(roots))

    def check_stability(self, A_fixed: float, I_ext: float, J: float) -> str:
        """
        Checks linear stability at the fixed point.
        Condition: J * g'(I_total) < 1 for stability (in simple rate model).
        """
        I_total = I_ext + J * A_fixed
        
        # Numerical derivative of gain function
        delta = 0.01
        g_plus = self.compute_gain(I_total + delta)
        g_minus = self.compute_gain(I_total - delta)
        g_prime = (g_plus - g_minus) / (2 * delta)
        
        eig = J * g_prime
        
        if eig > 1.0:
            return "Unstable"
        else:
            return "Stable"

    def detect_bifurcation(self, I_range: np.ndarray, J: float) -> List[Dict]:
        """
        Scans I_ext to find where number of fixed points changes.
        """
        points = []
        for I_ext in I_range:
            fps = self.find_fixed_points(I_ext, J)
            points.append({"I": I_ext, "FixedPoints": fps})
            
        # Analyze transitions...
        # Just returning the map for visualization for now
        return points
