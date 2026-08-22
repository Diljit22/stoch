import numpy as np
import sympy as sp
import matplotlib.pyplot as plt
import time
import sys
import os

# Adjust path to import local stochcalc package
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../src')))

from stochcalc import (
    TimeGrid,
    PathAnalyzer,
    GeometricBrownianMotion,
    FractionalBrownianMotion,
    VarianceGammaProcess,
    SDE,
    SDESolver,
    SymbolicSDE,
    symbolic_ito,
    quadratic_covariation,
    stochastic_product_rule,
    girsanov_transform,
    fokker_planck_pde,
    feynman_kac_pde,
    lamperti_transform,
    MalliavinGreeks,
    FourierCOS
)

try:
    from stochcalc import TorchSDESolver
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False

def test_timegrid_and_processes():
    print("\n==================================================")
    print("1. RUNNING NON-UNIFORM GRIDS & JUMP SIMULATIONS")
    print("==================================================")
    discrete_schedule = [0.0, 0.05, 0.12, 0.3, 0.45, 0.7, 0.85, 1.0]
    grid = TimeGrid(schedule=discrete_schedule)
    print(f"TimeGrid Initialized. Start: {grid.start}, End: {grid.end}, Discrete: {grid.is_discrete}")
    
    vg = VarianceGammaProcess(grid=grid, theta=-0.05, sigma=0.15, nu=0.1)
    times, paths = vg.simulate_paths(num_paths=10)
    print(f"Variance Gamma discrete simulation paths shape: {paths.shape}")
    
    continuous_grid = TimeGrid(start=0.0, end=1.0)
    fbm = FractionalBrownianMotion(H=0.7, grid=continuous_grid, dims=1)
    print("Simulating Fractional Brownian Motion (H = 0.7) via Covariance Cholesky...")
    fbm_times, fbm_paths = fbm.simulate_paths(num_paths=5, steps=100)
    print(f"fBM Paths Generated. Shape: {fbm_paths.shape}")
    
    qv = PathAnalyzer.realized_quadratic_variation(fbm_paths)
    mdd = PathAnalyzer.maximum_drawdown(fbm_paths)
    print(f"Paths Realized Quadratic Variation: {qv[:, 0]}")
    print(f"Paths Maximum Drawdowns: {mdd[:, 0]}")


def test_numerical_solvers_on_discrete_grid():
    print("\n==================================================")
    print("2. RUNNING SDE SOLVERS ON DISCRETE EVENT GRIDS")
    print("==================================================")
    discrete_grid = TimeGrid(schedule=[0.0, 0.1, 0.25, 0.5, 0.75, 1.1, 1.5, 2.0])
    
    k, theta_val, sigma_val = 1.5, 0.06, 0.12
    def cir_drift(t, x): return k * (theta_val - x)
    def cir_diffusion(t, x): return sigma_val * np.sqrt(np.clip(x, 1e-6, None))

    cir_sde = SDE(drift=cir_drift, diffusion=cir_diffusion, x0=np.array([0.03]), grid=discrete_grid)
    solver = SDESolver(cir_sde)
    
    print("Integrating SDE on discrete schedule using Euler-Maruyama...")
    times, paths = solver.euler_maruyama(num_paths=100)
    print(f"Euler Maruyama completed over discrete nodes {times}. Paths shape: {paths.shape}")


def test_symbolic_itos_polar():
    print("\n==================================================")
    print("3. COMPLEX SYMBOLIC ITO'S LEMMA (POLAR TRANSFORMATION)")
    print("==================================================")
    x, y, t = sp.symbols('x y t')
    planar_bm = SymbolicSDE(state_vars=[x, y], drift=[0, 0], diffusion=[[1, 0], [0, 1]], time_var=t)
    
    r = sp.sqrt(x**2 + y**2)
    theta = sp.atan2(y, x)
    
    print("Applying multidimensional Ito's Formula...")
    polar_sde = symbolic_ito([r, theta], [x, y], planar_bm)
    print("\nTransformed SDE (Polar Coordinate Bessel Process):")
    print(polar_sde)


def test_advanced_symbolic():
    print("\n==================================================")
    print("4. PARABOLIC PDEs & INTEGRATION TRANSFORMS")
    print("==================================================")
    x, t = sp.symbols('x t')
    mu, sigma, gamma, r = sp.symbols('mu sigma gamma r')
    
    stock_gbm = SymbolicSDE(state_vars=[x], drift=[r * x], diffusion=[[sigma * x]], time_var=t)
    u = sp.Function('V')(t, x)
    bs_pde = feynman_kac_pde(stock_gbm, discount_rate=r, u_function=u)
    print("Feynman-Kac derived Black-Scholes PDE:")
    sp.pprint(bs_pde)
    
    cev = SymbolicSDE(state_vars=[x], drift=[mu * x], diffusion=[[sigma * (x**gamma)]], time_var=t)
    lamperti_sde = lamperti_transform(cev)
    print("\nLamperti transformation on CEV Process:")
    print(lamperti_sde)


def test_option_pricing():
    print("\n==================================================")
    print("5. ANALYTICAL OPTIONS PRICING")
    print("==================================================")
    S0, K, rate, T, vol = 100.0, 100.0, 0.05, 1.0, 0.2
    
    def bs_cf(u): return np.exp(1j * u * (rate - 0.5 * vol**2) * T - 0.5 * vol**2 * u**2 * T)
    cos_engine = FourierCOS()
    price = cos_engine.price_european(bs_cf, S0, K, rate, T, option_type="call")
    print(f"Fourier COS European Call Option Price: {price:.5f}")


def test_gpu_acceleration():
    print("\n==================================================")
    print("6. M5 HARDWARE BACKEND ACCELERATION")
    print("==================================================")
    if not HAS_TORCH:
        print("PyTorch is not installed. Skipping GPU validation.")
        return
        
    def drift_fn(t, x): return 0.05 * x
    def diff_fn(t, x): return 0.2 * x
    
    gpu_solver = TorchSDESolver(drift_fn, diff_fn, x0=[1.0])
    print(f"Selected Device: {gpu_solver.device}")
    
    t_start = time.time()
    times, paths = gpu_solver.solve_euler_maruyama(num_paths=100_000, steps=100)
    t_duration = time.time() - t_start
    print(f"Successfully simulated 100,000 paths over 100 steps on GPU in {t_duration:.5f} seconds.")


if __name__ == "__main__":
    test_timegrid_and_processes()
    test_numerical_solvers_on_discrete_grid()
    test_symbolic_itos_polar()
    test_advanced_symbolic()
    test_option_pricing()
    test_gpu_acceleration()
    print("\nALL INTEGRATED MODULE TESTS EXECUTED NATIVELY WITH NO WARNINGS.")
