from typing import Union, Tuple, Optional
import numpy as np

class TimeGrid:
    def __init__(self, 
                 start: float = 0.0, 
                 end: Optional[float] = None, 
                 schedule: Optional[Union[np.ndarray, list]] = None):
        if schedule is not None:
            self.grid = np.unique(np.array(schedule, dtype=np.float64))
            self.is_discrete = True
            if len(self.grid) < 2:
                raise ValueError("Discrete schedules must contain at least 2 coordinate points.")
        else:
            if end is None:
                raise ValueError("Must provide an 'end' boundary for continuous time grids.")
            if start >= end:
                raise ValueError("Start boundary must be strictly less than end boundary.")
            self.grid = np.array([start, end], dtype=np.float64)
            self.is_discrete = False

    @property
    def start(self) -> float:
        return float(self.grid[0])

    @property
    def end(self) -> float:
        return float(self.grid[-1])

    @property
    def span(self) -> float:
        return self.end - self.start

    def generate_steps(self, steps: Optional[int] = None) -> Tuple[np.ndarray, np.ndarray]:
        if self.is_discrete:
            times = self.grid
        else:
            if steps is None:
                raise ValueError("Must specify 'steps' to discretize a continuous TimeGrid.")
            times = np.linspace(self.start, self.end, steps + 1)
            
        dts = np.diff(times)
        return times, dts

    def restrict(self, min_t: Optional[float] = None, max_t: Optional[float] = None) -> 'TimeGrid':
        t_start = max(self.start, min_t) if min_t is not None else self.start
        t_end = min(self.end, max_t) if max_t is not None else self.end
        
        if t_start >= t_end:
            raise ValueError("Restricted boundaries result in an empty or inverted interval.")
            
        if self.is_discrete:
            subset = self.grid[(self.grid >= t_start) & (self.grid <= t_end)]
            return TimeGrid(schedule=subset)
        else:
            return TimeGrid(start=t_start, end=t_end)

    def __repr__(self) -> str:
        return f"TimeGrid(start={self.start}, end={self.end}, discrete={self.is_discrete})"
