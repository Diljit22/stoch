from typing import Callable, Optional
import numpy as np
from ..core.index import TimeGrid

class SDE:
    def __init__(self, 
                 drift: Callable[[float, np.ndarray], np.ndarray], 
                 diffusion: Callable[[float, np.ndarray], np.ndarray], 
                 x0: np.ndarray, 
                 grid: TimeGrid,
                 noise_dims: Optional[int] = None):
        self.drift = drift
        self.diffusion = diffusion
        self.x0 = np.array(x0, dtype=np.float64)
        self.grid = grid
        self.dims = len(self.x0)
        self.noise_dims = noise_dims if noise_dims is not None else self.dims
