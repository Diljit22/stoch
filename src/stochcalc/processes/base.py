from abc import ABC, abstractmethod
from typing import Tuple, Any, Optional
import numpy as np
import matplotlib.pyplot as plt
from ..core.index import TimeGrid

class StochasticProcess(ABC):
    def __init__(self, grid: TimeGrid, dims: int = 1):
        self.grid = grid
        self.dims = dims

    @abstractmethod
    def sample_marginal(self, t: float, num_paths: int) -> np.ndarray:
        pass

    @abstractmethod
    def simulate_paths(self, num_paths: int, steps: Optional[int] = None) -> Tuple[np.ndarray, np.ndarray]:
        pass

    def plot(self, num_paths: int = 5, steps: Optional[int] = None, **kwargs: Any) -> None:
        times, paths = self.simulate_paths(num_paths=num_paths, steps=steps)
        
        plt.figure(figsize=(10, 5))
        for d in range(self.dims):
            for p in range(num_paths):
                label = f"Path {p+1} (Dim {d+1})" if (num_paths <= 5 and self.dims <= 2) else ""
                plt.plot(times, paths[p, :, d], lw=1.2, alpha=0.8, label=label)
        
        plt.title(f"Sample Paths: {self.__class__.__name__}")
        plt.xlabel("Time (t)")
        plt.ylabel("State Vector X_t")
        plt.grid(True, linestyle="--", alpha=0.5)
        if num_paths <= 5 and self.dims <= 2:
            plt.legend()
        plt.show()
