import pytest
import numpy as np
from stochcalc import FourierCOS, MalliavinGreeks

def test_fourier_cos():
    # Test European Call price against typical parameters
    S0, K, rate, T, vol = 100.0, 100.0, 0.05, 1.0, 0.2
    def bs_cf(u): return np.exp(1j * u * (rate - 0.5 * vol**2) * T - 0.5 * vol**2 * u**2 * T)
    
    cos_engine = FourierCOS()
    price = cos_engine.price_european(bs_cf, S0, K, rate, T, option_type="call")
    # Black-Scholes analytical target is ~10.45058
    assert np.allclose(price, 10.45058, atol=1e-3)

def test_malliavin_greeks():
    engine = MalliavinGreeks(S0=100.0, r=0.05, sigma=0.2, T=1.0)
    def call_payoff(S): return np.maximum(S - 100.0, 0.0)
    
    delta, gamma, stats = engine.compute_greeks(call_payoff, num_paths=5000)
    assert 0.0 < delta < 1.0
    assert gamma > 0.0
    assert "delta_se" in stats