from typing import Tuple, Optional
import numpy as np
from .base import StochasticProcess
from ..core.index import TimeGrid

class BrownianMotion(StochasticProcess):
    def __init__(self, 
                 grid: TimeGrid,
                 drift: Optional[np.ndarray] = None, 
                 scale: Optional[np.ndarray] = None, 
                 correlation_matrix: Optional[np.ndarray] = None, 
                 dims: int = 1):
        super().__init__(grid, dims)
        self.drift = np.zeros(dims) if drift is None else np.array(drift, dtype=np.float64)
        self.scale = np.ones(dims) if scale is None else np.array(scale, dtype=np.float64)
        
        if len(self.drift) != dims or len(self.scale) != dims:
            raise ValueError("Drift and Scale dimensions must match process dimensions.")
            
        if correlation_matrix is not None:
            self.L = np.linalg.cholesky(correlation_matrix)
        else:
            self.L = np.eye(dims)

    def sample_marginal(self, t: float, num_paths: int) -> np.ndarray:
        dt = t - self.grid.start
        Z = np.random.normal(size=(num_paths, self.dims))
        correlated_Z = Z @ self.L.T
        return self.drift * dt + self.scale * np.sqrt(dt) * correlated_Z

    def simulate_paths(self, num_paths: int, steps: Optional[int] = None) -> Tuple[np.ndarray, np.ndarray]:
        times, dts = self.grid.generate_steps(steps)
        n_steps = len(dts)
        
        paths = np.zeros((num_paths, n_steps + 1, self.dims))
        Z = np.random.normal(size=(num_paths, n_steps, self.dims))
        correlated_Z = np.einsum('psd,kd->psk', Z, self.L)
        
        dt_col = dts[np.newaxis, :, np.newaxis]
        increments = self.drift * dt_col + self.scale * np.sqrt(dt_col) * correlated_Z
        paths[:, 1:, :] = np.cumsum(increments, axis=1)
        return times, paths


class GeometricBrownianMotion(StochasticProcess):
    def __init__(self, 
                 grid: TimeGrid,
                 mu: np.ndarray, 
                 sigma: np.ndarray, 
                 S0: np.ndarray, 
                 correlation_matrix: Optional[np.ndarray] = None):
        super().__init__(grid, dims=len(S0))
        self.mu = np.array(mu, dtype=np.float64)
        self.sigma = np.array(sigma, dtype=np.float64)
        self.S0 = np.array(S0, dtype=np.float64)
        
        if correlation_matrix is not None:
            self.L = np.linalg.cholesky(correlation_matrix)
        else:
            self.L = np.eye(self.dims)

    def sample_marginal(self, t: float, num_paths: int) -> np.ndarray:
        dt = t - self.grid.start
        Z = np.random.normal(size=(num_paths, self.dims))
        correlated_Z = Z @ self.L.T
        drift_term = (self.mu - 0.5 * self.sigma**2) * dt
        diffusion_term = self.sigma * np.sqrt(dt) * correlated_Z
        return self.S0 * np.exp(drift_term + diffusion_term)

    def simulate_paths(self, num_paths: int, steps: Optional[int] = None) -> Tuple[np.ndarray, np.ndarray]:
        times, dts = self.grid.generate_steps(steps)
        n_steps = len(dts)
        
        paths = np.zeros((num_paths, n_steps + 1, self.dims))
        paths[:, 0, :] = self.S0
        
        Z = np.random.normal(size=(num_paths, n_steps, self.dims))
        correlated_Z = np.einsum('psd,kd->psk', Z, self.L)
        
        dt_col = dts[np.newaxis, :, np.newaxis]
        drift = (self.mu - 0.5 * self.sigma**2) * dt_col
        log_increments = drift + self.sigma * np.sqrt(dt_col) * correlated_Z
        paths[:, 1:, :] = self.S0 * np.exp(np.cumsum(log_increments, axis=1))
        return times, paths


class FractionalBrownianMotion(StochasticProcess):
    def __init__(self, H: float, grid: TimeGrid, dims: int = 1):
        super().__init__(grid, dims)
        if not (0.0 < H < 1.0):
            raise ValueError("Hurst parameter H must be in the open interval (0, 1).")
        self.H = H

    def _covariance_matrix(self, times: np.ndarray) -> np.ndarray:
        N = len(times)
        C = np.zeros((N, N))
        H2 = 2.0 * self.H
        for i in range(N):
            for j in range(N):
                t_i = times[i]
                t_j = times[j]
                C[i, j] = 0.5 * (t_i**H2 + t_j**H2 - np.abs(t_i - t_j)**H2)
        return C

    def sample_marginal(self, t: float, num_paths: int) -> np.ndarray:
        std = np.sqrt(t**(2.0 * self.H))
        return np.random.normal(0.0, std, size=(num_paths, self.dims))

    def simulate_paths(self, num_paths: int, steps: Optional[int] = None) -> Tuple[np.ndarray, np.ndarray]:
        times, _ = self.grid.generate_steps(steps)
        n_steps = len(times) - 1
        
        paths = np.zeros((num_paths, n_steps + 1, self.dims))
        active_times = times[1:]
        C = self._covariance_matrix(active_times)
        C += np.eye(n_steps) * 1e-12
        L = np.linalg.cholesky(C)
        
        Z = np.random.normal(size=(num_paths, n_steps, self.dims))
        paths[:, 1:, :] = np.einsum('ij,pjd->pid', L, Z)
        
        return times, paths