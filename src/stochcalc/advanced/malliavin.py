from typing import Callable, Tuple, Dict
import numpy as np

class MalliavinGreeks:
    def __init__(self, S0: float, r: float, sigma: float, T: float):
        self.S0 = S0
        self.r = r
        self.sigma = sigma
        self.T = T

    def compute_greeks(self, 
                       payoff_fn: Callable[[np.ndarray], np.ndarray], 
                       num_paths: int = 500_000) -> Tuple[float, float, Dict[str, float]]:
        W_T = np.random.normal(loc=0.0, scale=np.sqrt(self.T), size=num_paths)
        drift = (self.r - 0.5 * self.sigma**2) * self.T
        S_T = self.S0 * np.exp(drift + self.sigma * W_T)
        
        payoffs = payoff_fn(S_T)
        discount = np.exp(-self.r * self.T)
        
        weight_delta = W_T / (self.S0 * self.sigma * self.T)
        weight_gamma = (W_T**2 - self.sigma * self.T * W_T - self.T) / (self.S0**2 * self.sigma**2 * self.T**2)
        
        delta_estimates = discount * payoffs * weight_delta
        gamma_estimates = discount * payoffs * weight_gamma
        
        delta = float(np.mean(delta_estimates))
        gamma = float(np.mean(gamma_estimates))
        
        delta_se = float(np.std(delta_estimates) / np.sqrt(num_paths))
        gamma_se = float(np.std(gamma_estimates) / np.sqrt(num_paths))
        
        return delta, gamma, {
            "delta_se": delta_se,
            "gamma_se": gamma_se,
            "delta_95_lower": delta - 1.96 * delta_se,
            "delta_95_upper": delta + 1.96 * delta_se,
            "gamma_95_lower": gamma - 1.96 * gamma_se,
            "gamma_95_upper": gamma + 1.96 * gamma_se
        }
