from typing import Callable, Tuple
import torch
import numpy as np

class TorchSDESolver:
    def __init__(self, 
                 drift_fn: Callable[[torch.Tensor, torch.Tensor], torch.Tensor], 
                 diff_fn: Callable[[torch.Tensor, torch.Tensor], torch.Tensor], 
                 x0: torch.Tensor,
                 t_span: Tuple[float, float] = (0.0, 1.0)):
        if torch.backends.mps.is_available():
            self.device = torch.device("mps")
        elif torch.cuda.is_available():
            self.device = torch.device("cuda")
        else:
            self.device = torch.device("cpu")
            
        self.drift = drift_fn
        self.diff = diff_fn
        
        if isinstance(x0, torch.Tensor):
            self.x0 = x0.clone().detach().to(dtype=torch.float32, device=self.device)
        else:
            self.x0 = torch.tensor(x0, dtype=torch.float32, device=self.device)
            
        self.t_span = t_span
        self.dims = len(self.x0)

    def solve_euler_maruyama(self, num_paths: int, steps: int) -> Tuple[torch.Tensor, torch.Tensor]:
        t0, te = self.t_span
        dt = (te - t0) / steps
        times = torch.linspace(t0, te, steps + 1, device=self.device)
        
        paths = torch.zeros((num_paths, steps + 1, self.dims), device=self.device)
        paths[:, 0, :] = self.x0
        
        dW = torch.normal(mean=0.0, std=np.sqrt(dt), size=(num_paths, steps, self.dims), device=self.device)
        dt_tensor = torch.tensor(dt, device=self.device)
        
        for n in range(steps):
            t = times[n]
            X = paths[:, n, :]
            
            a_val = self.drift(t, X)
            b_val = self.diff(t, X)
            
            paths[:, n+1, :] = X + a_val * dt_tensor + b_val * dW[:, n, :]
            
        if self.device.type == "mps":
            torch.mps.synchronize()
        elif self.device.type == "cuda":
            torch.cuda.synchronize()
            
        return times.cpu(), paths.cpu()
