import pytest
import numpy as np
from stochcalc import (
    TimeGrid, BrownianMotion, GeometricBrownianMotion, 
    FractionalBrownianMotion, GammaProcess, VarianceGammaProcess, 
    CompoundPoissonProcess, MertonJumpDiffusion
)

@pytest.fixture
def test_grid():
    return TimeGrid(start=0.0, end=1.0)

def test_brownian_motion(test_grid):
    bm = BrownianMotion(grid=test_grid, dims=2)
    times, paths = bm.simulate_paths(num_paths=5, steps=10)
    assert paths.shape == (5, 11, 2)
    assert np.all(paths[:, 0, :] == 0.0)

def test_geometric_brownian_motion(test_grid):
    gbm = GeometricBrownianMotion(grid=test_grid, mu=np.array([0.05]), sigma=np.array([0.2]), S0=np.array([100.0]))
    times, paths = gbm.simulate_paths(num_paths=5, steps=10)
    assert paths.shape == (5, 11, 1)
    assert np.allclose(paths[:, 0, 0], 100.0)

def test_fractional_brownian_motion(test_grid):
    fbm = FractionalBrownianMotion(H=0.6, grid=test_grid)
    times, paths = fbm.simulate_paths(num_paths=3, steps=10)
    assert paths.shape == (3, 11, 1)

def test_gamma_process(test_grid):
    gp = GammaProcess(grid=test_grid, mu=1.0, nu=0.2)
    times, paths = gp.simulate_paths(num_paths=3, steps=10)
    assert paths.shape == (3, 11, 1)
    assert np.all(paths[:, 1:, :] >= 0.0)

def test_variance_gamma_process(test_grid):
    vg = VarianceGammaProcess(grid=test_grid, theta=-0.1, sigma=0.2, nu=0.1)
    times, paths = vg.simulate_paths(num_paths=3, steps=10)
    assert paths.shape == (3, 11, 1)

def test_compound_poisson_process(test_grid):
    cpp = CompoundPoissonProcess(grid=test_grid, lam=2.0, jump_mean=0.0, jump_std=1.0)
    times, paths = cpp.simulate_paths(num_paths=3, steps=10)
    assert paths.shape == (3, 11, 1)

def test_merton_jump_diffusion(test_grid):
    mjd = MertonJumpDiffusion(grid=test_grid, mu=0.05, sigma=0.2, S0=100.0, lam=1.0, jump_mean=-0.05, jump_std=0.1)
    times, paths = mjd.simulate_paths(num_paths=3, steps=10)
    assert paths.shape == (3, 11, 1)
    assert np.allclose(paths[:, 0, 0], 100.0)