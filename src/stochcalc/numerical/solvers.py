from typing import Tuple, List, Optional
import numpy as np
from .sde import SDE

class SDESolver:
    def __init__(self, sde: SDE):
        self.sde = sde

    def euler_maruyama(self, 
                       num_paths: int, 
                       steps: Optional[int] = None, 
                       boundary: Optional[str] = None) -> Tuple[np.ndarray, np.ndarray]:
        times, dts = self.sde.grid.generate_steps(steps)
        n_steps = len(dts)
        
        paths = np.zeros((num_paths, n_steps + 1, self.sde.dims))
        paths[:, 0, :] = self.sde.x0
        
        for n in range(n_steps):
            t = times[n]
            dt = dts[n]
            X = paths[:, n, :]
            
            dW = np.random.normal(loc=0.0, scale=np.sqrt(dt), size=(num_paths, self.sde.noise_dims))
            
            a_val = self.sde.drift(t, X)
            b_val = self.sde.diffusion(t, X)
            
            if b_val.ndim == 2:
                X_next = X + a_val * dt + b_val * dW
            else:
                X_next = X + a_val * dt + np.einsum('pdm,pm->pd', b_val, dW)
            
            if boundary == "reflect":
                X_next = np.abs(X_next)
            elif boundary == "absorb":
                X_next = np.where(X_next < 0, 0.0, X_next)
                
            paths[:, n+1, :] = X_next
            
        return times, paths

    def milstein(self, 
                 num_paths: int, 
                 steps: Optional[int] = None, 
                 h: float = 1e-5, 
                 boundary: Optional[str] = None) -> Tuple[np.ndarray, np.ndarray]:
        if self.sde.noise_dims != self.sde.dims:
            raise NotImplementedError("Milstein solver supports diagonal noise configurations.")
            
        times, dts = self.sde.grid.generate_steps(steps)
        n_steps = len(dts)
        
        paths = np.zeros((num_paths, n_steps + 1, self.sde.dims))
        paths[:, 0, :] = self.sde.x0
        
        for n in range(n_steps):
            t = times[n]
            dt = dts[n]
            X = paths[:, n, :]
            
            dW = np.random.normal(loc=0.0, scale=np.sqrt(dt), size=(num_paths, self.sde.dims))
            
            a_val = self.sde.drift(t, X)
            b_val = self.sde.diffusion(t, X)
            
            db_dx = np.zeros_like(b_val)
            for i in range(self.sde.dims):
                perturb = np.zeros_like(X)
                perturb[:, i] = h
                b_high = self.sde.diffusion(t, X + perturb)
                b_low = self.sde.diffusion(t, X - perturb)
                db_dx[:, i] = (b_high[:, i] - b_low[:, i]) / (2 * h)
            
            milstein_corr = 0.5 * b_val * db_dx * (dW**2 - dt)
            X_next = X + a_val * dt + b_val * dW + milstein_corr
            
            if boundary == "reflect":
                X_next = np.abs(X_next)
            elif boundary == "absorb":
                X_next = np.where(X_next < 0, 0.0, X_next)
                
            paths[:, n+1, :] = X_next
            
        return times, paths

    def platen_rk10(self, 
                    num_paths: int, 
                    steps: Optional[int] = None, 
                    boundary: Optional[str] = None) -> Tuple[np.ndarray, np.ndarray]:
        if self.sde.noise_dims != self.sde.dims:
            raise NotImplementedError("Platen RK 1.0 supports diagonal noise configurations.")
            
        times, dts = self.sde.grid.generate_steps(steps)
        n_steps = len(dts)
        
        paths = np.zeros((num_paths, n_steps + 1, self.sde.dims))
        paths[:, 0, :] = self.sde.x0
        
        for n in range(n_steps):
            t = times[n]
            dt = dts[n]
            X = paths[:, n, :]
            
            dW = np.random.normal(loc=0.0, scale=np.sqrt(dt), size=(num_paths, self.sde.dims))
            
            a_val = self.sde.drift(t, X)
            b_val = self.sde.diffusion(t, X)
            
            X_support = X + a_val * dt + b_val * np.sqrt(dt)
            b_support = self.sde.diffusion(t, X_support)
            
            platen_corr = (0.5 / np.sqrt(dt)) * (b_support - b_val) * (dW**2 - dt)
            X_next = X + a_val * dt + b_val * dW + platen_corr
            
            if boundary == "reflect":
                X_next = np.abs(X_next)
            elif boundary == "absorb":
                X_next = np.where(X_next < 0, 0.0, X_next)
                
            paths[:, n+1, :] = X_next
            
        return times, paths

    def adaptive_solve(self, 
                       num_paths: int, 
                       tol: float = 1e-3, 
                       dt_min: float = 1e-6, 
                       dt_max: float = 0.1, 
                       boundary: Optional[str] = None) -> Tuple[np.ndarray, np.ndarray]:
        t0 = self.sde.grid.start
        te = self.sde.grid.end
        t = t0
        dt = dt_max
        
        times: List[float] = [t]
        paths: List[np.ndarray] = [np.repeat(self.sde.x0[np.newaxis, :], num_paths, axis=0)]

        while t < te:
            if t + dt > te:
                dt = te - t
                
            Z = np.random.normal(size=(num_paths, self.sde.dims))
            dW = Z * np.sqrt(dt)
            
            X = paths[-1]
            a_val = self.sde.drift(t, X)
            b_val = self.sde.diffusion(t, X)
            
            X_em = X + a_val * dt + b_val * dW
            
            X_support = X + a_val * dt + b_val * np.sqrt(dt)
            b_support = self.sde.diffusion(t, X_support)
            corr = (0.5 / np.sqrt(dt)) * (b_support - b_val) * (dW**2 - dt)
            X_rk = X + a_val * dt + b_val * dW + corr
            
            error = np.linalg.norm(X_rk - X_em, axis=1)
            max_error = np.max(error)
            
            if max_error <= tol or dt <= dt_min:
                t += dt
                times.append(t)
                
                if boundary == "reflect":
                    X_rk = np.abs(X_rk)
                elif boundary == "absorb":
                    X_rk = np.where(X_rk < 0, 0.0, X_rk)
                    
                paths.append(X_rk)
                
                factor = 0.85 * (tol / (max_error + 1e-16))**0.25
                dt = np.clip(dt * factor, dt_min, dt_max)
            else:
                dt = max(dt * 0.5, dt_min)
                
        return np.array(times), np.swapaxes(np.array(paths), 0, 1)