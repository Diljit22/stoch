from typing import Callable
import numpy as np

class FourierCOS:
    @staticmethod
    def _chi(k: np.ndarray, a: float, b: float, c: float, d: float) -> np.ndarray:
        bma = b - a
        omega = k * np.pi / bma
        exp_d = np.exp(d)
        exp_c = np.exp(c)
        
        cos_d = np.cos(omega * (d - a))
        cos_c = np.cos(omega * (c - a))
        sin_d = np.sin(omega * (d - a))
        sin_c = np.sin(omega * (c - a))
        
        term1 = cos_d * exp_d - cos_c * exp_c
        term2 = omega * (sin_d * exp_d - sin_c * exp_c)
        return (term1 + term2) / (1.0 + omega**2)

    @staticmethod
    def _psi(k: np.ndarray, a: float, b: float, c: float, d: float) -> np.ndarray:
        bma = b - a
        omega = k * np.pi / bma
        psi = np.zeros_like(k, dtype=np.float64)
        
        mask = k != 0
        psi[mask] = (np.sin(omega[mask] * (d - a)) - np.sin(omega[mask] * (c - a))) / omega[mask]
        psi[~mask] = d - c
        return psi

    def price_european(self, 
                       char_func: Callable[[np.ndarray], np.ndarray], 
                       S0: float, 
                       K: float, 
                       r: float, 
                       T: float, 
                       option_type: str = "call", 
                       N: int = 128, 
                       L: float = 10.0,
                       sigma_cumulant: float = 0.2) -> float:
        c1 = (r - 0.5 * sigma_cumulant**2) * T
        c2 = (sigma_cumulant**2) * T
        a = c1 - L * np.sqrt(c2)
        b = c1 + L * np.sqrt(c2)
        
        x = np.log(S0 / K)
        k = np.arange(N)
        
        bma = b - a
        if option_type.lower() == "call":
            U = (2.0 / bma) * (self._chi(k, a, b, 0.0, b) - self._psi(k, a, b, 0.0, b))
        elif option_type.lower() == "put":
            U = (2.0 / bma) * (-self._chi(k, a, b, a, 0.0) + self._psi(k, a, b, a, 0.0))
        else:
            raise ValueError("Option type must be 'call' or 'put'.")
            
        omega = k * np.pi / bma
        cf_vals = char_func(omega)
        U[0] = 0.5 * U[0]
        
        option_value = K * np.exp(-r * T) * np.sum(np.real(cf_vals * np.exp(1j * omega * (x - a))) * U)
        return float(option_value)