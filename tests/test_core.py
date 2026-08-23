import pytest
import numpy as np
from stochcalc import TimeGrid, PathAnalyzer

def test_time_grid_continuous():
    grid = TimeGrid(start=0.0, end=2.0)
    assert grid.start == 0.0
    assert grid.end == 2.0
    assert grid.span == 2.0
    
    times, dts = grid.generate_steps(steps=10)
    assert len(times) == 11
    assert len(dts) == 10
    assert np.allclose(dts, 0.2)

def test_time_grid_discrete():
    grid = TimeGrid(schedule=[0.0, 0.5, 1.2, 2.0])
    assert grid.is_discrete
    assert grid.end == 2.0
    
    times, dts = grid.generate_steps()
    assert np.array_equal(times, [0.0, 0.5, 1.2, 2.0])

def test_path_analyzer():
    times = np.array([0.0, 1.0, 2.0])
    paths = np.array([[[1.0], [1.5], [2.1]]]) # shape: (1, 3, 1)
    
    # First passage time
    fpt = PathAnalyzer.first_passage_time(times, paths, barrier=1.4)
    assert fpt[0, 0] == 1.0
    
    # Realized quadratic variation
    qv = PathAnalyzer.realized_quadratic_variation(paths)
    assert np.allclose(qv[0, 0], 0.25 + 0.36)
    
    # Maximum Drawdown (positive path)
    mdd = PathAnalyzer.maximum_drawdown(paths)
    assert mdd[0, 0] == 0.0 # No drop