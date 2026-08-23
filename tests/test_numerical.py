import pytest
import numpy as np
from stochcalc import TimeGrid, SDE, SDESolver

@pytest.fixture
def custom_sde():
    grid = TimeGrid(start=0.0, end=1.0)
    # Simple Ornstein-Uhlenbeck Process
    def drift(t, x): return 1.0 * (0.05 - x)
    def diff(t, x): return 0.2 * np.ones_like(x)
    return SDE(drift=drift, diffusion=diff, x0=np.array([0.05]), grid=grid)

def test_euler_maruyama_boundaries(custom_sde):
    solver = SDESolver(custom_sde)
    
    # Test Reflective boundary
    _, paths_reflect = solver.euler_maruyama(num_paths=10, steps=10, boundary="reflect")
    assert np.all(paths_reflect >= 0.0)
    
    # Test Absorbing boundary
    _, paths_absorb = solver.euler_maruyama(num_paths=10, steps=10, boundary="absorb")
    assert np.all(paths_absorb >= 0.0)

def test_milstein_and_platen(custom_sde):
    solver = SDESolver(custom_sde)
    _, paths_mil = solver.milstein(num_paths=5, steps=10)
    assert paths_mil.shape == (5, 11, 1)
    
    _, paths_plat = solver.platen_rk10(num_paths=5, steps=10)
    assert paths_plat.shape == (5, 11, 1)

def test_adaptive_solve(custom_sde):
    solver = SDESolver(custom_sde)
    times, paths = solver.adaptive_solve(num_paths=5, tol=1e-2)
    assert len(times) > 1
    assert paths.shape[0] == 5
    assert paths.shape[1] == len(times)