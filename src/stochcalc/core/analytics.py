import numpy as np

class PathAnalyzer:
    @staticmethod
    def first_passage_time(times: np.ndarray, paths: np.ndarray, barrier: float) -> np.ndarray:
        num_paths, _, dims = paths.shape
        fpt = np.full((num_paths, dims), np.nan)
        
        for p in range(num_paths):
            for d in range(dims):
                x = paths[p, :, d]
                x0 = x[0]
                if barrier >= x0:
                    crossings = np.where(x >= barrier)[0]
                else:
                    crossings = np.where(x <= barrier)[0]
                
                if crossings.size > 0:
                    fpt[p, d] = times[crossings[0]]
                    
        return fpt

    @staticmethod
    def maximum_drawdown(paths: np.ndarray) -> np.ndarray:
        num_paths, _, dims = paths.shape
        mdd = np.zeros((num_paths, dims))
        
        for p in range(num_paths):
            for d in range(dims):
                x = paths[p, :, d]
                peaks = np.maximum.accumulate(x)
                if np.any(x <= 0):
                    # Absolute drawdown for processes that can be zero or negative (e.g. FBM)
                    drawdowns = x - peaks
                else:
                    # Relative drawdown for strictly positive processes (e.g. GBM)
                    drawdowns = (x - peaks) / peaks
                mdd[p, d] = np.min(drawdowns)
                
        return mdd

    @staticmethod
    def realized_quadratic_variation(paths: np.ndarray) -> np.ndarray:
        differences = np.diff(paths, axis=1)
        return np.sum(differences**2, axis=1)
