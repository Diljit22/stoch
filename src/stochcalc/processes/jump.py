from typing import Tuple, Optional
import numpy as np
from .base import StochasticProcess
from ..core.index import TimeGrid

class GammaProcess(StochasticProcess):
    def __init__(self, grid: TimeGrid, mu: float, nu: float):
        super().__init__(grid, dims=1)
        self.mu = mu
        self.nu = nu

    def sample_marginal(self, t: float, num_paths: int) -> np.ndarray:
        dt = t - self.grid.start
        shape = np.broadcast_to(dt / self.nu, (num_paths, 1))
        scale = self.mu * self.nu
        return np.random.gamma(shape, scale)

    def simulate_paths(self, num_paths: int, steps: Optional[int] = None) -> Tuple[np.ndarray, np.ndarray]:
        times, dts = self.grid.generate_steps(steps)
        n_steps = len(dts)
        
        paths = np.zeros((num_paths, n_steps + 1, self.dims))
        shape_raw = dts[np.newaxis, :, np.newaxis] / self.nu
        shape = np.broadcast_to(shape_raw, (num_paths, n_steps, self.dims))
        scale = self.mu * self.nu
        
        increments = np.random.gamma(shape, scale)
        paths[:, 1:, :] = np.cumsum(increments, axis=1)
        return times, paths


class VarianceGammaProcess(StochasticProcess):
    def __init__(self, grid: TimeGrid, theta: float, sigma: float, nu: float):
        super().__init__(grid, dims=1)
        self.theta = theta
        self.sigma = sigma
        self.nu = nu

    def sample_marginal(self, t: float, num_paths: int) -> np.ndarray:
        dt = t - self.grid.start
        shape = np.broadcast_to(dt / self.nu, (num_paths, 1))
        g = np.random.gamma(shape, self.nu)
        Z = np.random.normal(size=(num_paths, 1))
        return self.theta * g + self.sigma * np.sqrt(g) * Z

    def simulate_paths(self, num_paths: int, steps: Optional[int] = None) -> Tuple[np.ndarray, np.ndarray]:
        times, dts = self.grid.generate_steps(steps)
        n_steps = len(dts)
        
        paths = np.zeros((num_paths, n_steps + 1, self.dims))
        dt_col = dts[np.newaxis, :, np.newaxis]
        shape = np.broadcast_to(dt_col / self.nu, (num_paths, n_steps, self.dims))
        dg = np.random.gamma(shape, self.nu)
        Z = np.random.normal(size=(num_paths, n_steps, self.dims))
        
        increments = self.theta * dg + self.sigma * np.sqrt(dg) * Z
        paths[:, 1:, :] = np.cumsum(increments, axis=1)
        return times, paths


class CompoundPoissonProcess(StochasticProcess):
    def __init__(self, grid: TimeGrid, lam: float, jump_mean: float, jump_std: float):
        super().__init__(grid, dims=1)
        self.lam = lam
        self.jump_mean = jump_mean
        self.jump_std = jump_std

    def sample_marginal(self, t: float, num_paths: int) -> np.ndarray:
        dt = t - self.grid.start
        num_jumps = np.random.poisson(self.lam * dt, size=num_paths)
        samples = np.zeros((num_paths, 1))
        for p in range(num_paths):
            n = num_jumps[p]
            if n > 0:
                samples[p, 0] = np.sum(np.random.normal(self.jump_mean, self.jump_std, size=n))
        return samples

    def simulate_paths(self, num_paths: int, steps: Optional[int] = None) -> Tuple[np.ndarray, np.ndarray]:
        times, dts = self.grid.generate_steps(steps)
        n_steps = len(dts)
        
        paths = np.zeros((num_paths, n_steps + 1, self.dims))
        
        for n in range(n_steps):
            dt = dts[n]
            num_jumps = np.random.poisson(self.lam * dt, size=num_paths)
            increments = np.zeros(num_paths)
            
            active_paths = np.where(num_jumps > 0)[0]
            for p in active_paths:
                j_count = num_jumps[p]
                increments[p] = np.sum(np.random.normal(self.jump_mean, self.jump_std, size=j_count))
                
            paths[:, n+1, 0] = paths[:, n, 0] + increments
            
        return times, paths


class MertonJumpDiffusion(StochasticProcess):
    def __init__(self, grid: TimeGrid, mu: float, sigma: float, S0: float, 
                 lam: float, jump_mean: float, jump_std: float):
        super().__init__(grid, dims=1)
        self.mu = mu
        self.sigma = sigma
        self.S0 = S0
        self.lam = lam
        self.jump_mean = jump_mean
        self.jump_std = jump_std

    def sample_marginal(self, t: float, num_paths: int) -> np.ndarray:
        dt = t - self.grid.start
        Z = np.random.normal(size=(num_paths, 1))
        continuous = (self.mu - 0.5 * self.sigma**2) * dt + self.sigma * np.sqrt(dt) * Z
        
        num_jumps = np.random.poisson(self.lam * dt, size=num_paths)
        jumps = np.zeros((num_paths, 1))
        for p in range(num_paths):
            n = num_jumps[p]
            if n > 0:
                jumps[p, 0] = np.sum(np.random.normal(self.jump_mean, self.jump_std, size=n))
                
        return self.S0 * np.exp(continuous + jumps)

    def simulate_paths(self, num_paths: int, steps: Optional[int] = None) -> Tuple[np.ndarray, np.ndarray]:
        times, dts = self.grid.generate_steps(steps)
        n_steps = len(dts)
        
        paths = np.zeros((num_paths, n_steps + 1, self.dims))
        paths[:, 0, :] = self.S0
        
        for n in range(n_steps):
            dt = dts[n]
            Z = np.random.normal(size=(num_paths, 1))
            continuous = (self.mu - 0.5 * self.sigma**2) * dt + self.sigma * np.sqrt(dt) * Z
            
            num_jumps = np.random.poisson(self.lam * dt, size=num_paths)
            jumps = np.zeros((num_paths, 1))
            active_paths = np.where(num_jumps > 0)[0]
            for p in active_paths:
                jumps[p, 0] = np.sum(np.random.normal(self.jump_mean, self.jump_std, size=num_jumps[p]))
                
            paths[:, n+1, :] = paths[:, n, :] * np.exp(continuous + jumps)
            
        return times, paths