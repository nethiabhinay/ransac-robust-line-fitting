"""Reproducible synthetic 2D line data with labelled inliers and outliers."""
from dataclasses import dataclass
from typing import Optional, Tuple
import numpy as np

@dataclass
class Dataset:
    points: np.ndarray
    inlier_mask: np.ndarray
    true_slope: float
    true_intercept: float
    x_range: Tuple[float, float]

    @property
    def inliers(self): return self.points[self.inlier_mask]
    @property
    def outliers(self): return self.points[~self.inlier_mask]
    @property
    def n_points(self): return len(self.points)
    @property
    def n_inliers(self): return int(self.inlier_mask.sum())
    @property
    def n_outliers(self): return int((~self.inlier_mask).sum())
    @property
    def outlier_percentage(self):
        return 100.0 * self.n_outliers / self.n_points if self.n_points else 0.0

def generate_line_data(n_inliers=100, n_outliers=40, slope=2.0, intercept=5.0,
                       noise_std=1.0, x_range=(0.0, 20.0), y_margin=15.0,
                       seed: Optional[int]=42, enforce_separation=True):
    """Generate shuffled points; ground-truth labels are evaluation-only."""
    if isinstance(n_inliers, bool) or int(n_inliers) != n_inliers or n_inliers < 2:
        raise ValueError("n_inliers must be an integer >= 2.")
    if isinstance(n_outliers, bool) or int(n_outliers) != n_outliers or n_outliers < 0:
        raise ValueError("n_outliers must be an integer >= 0.")
    if not np.isfinite([slope, intercept, noise_std, y_margin]).all():
        raise ValueError("Numeric parameters must be finite.")
    if noise_std < 0 or y_margin <= 0:
        raise ValueError("noise_std must be >= 0 and y_margin must be > 0.")
    if len(x_range) != 2 or not np.isfinite(x_range).all() or x_range[0] >= x_range[1]:
        raise ValueError("x_range must contain two finite values with min < max.")
    rng = np.random.default_rng(seed)
    xmin, xmax = map(float, x_range)
    x = rng.uniform(xmin, xmax, int(n_inliers))
    y = slope*x + intercept + rng.normal(0, noise_std, int(n_inliers))
    inlier_points = np.column_stack((x, y))
    if n_outliers:
        ya, yb = slope*xmin + intercept, slope*xmax + intercept
        ymin, ymax = min(ya, yb)-y_margin, max(ya, yb)+y_margin
        xo = rng.uniform(xmin, xmax, int(n_outliers))
        yo = rng.uniform(ymin, ymax, int(n_outliers))
        if enforce_separation:
            threshold = max(3.0*noise_std, 2.0)
            close = np.abs(yo - (slope*xo + intercept)) < threshold
            signs = np.where(rng.random(close.sum()) > .5, 1.0, -1.0)
            yo[close] += signs*(threshold+5.0)
        points = np.vstack((inlier_points, np.column_stack((xo, yo))))
        mask = np.r_[np.ones(int(n_inliers), dtype=bool), np.zeros(int(n_outliers), dtype=bool)]
    else:
        points, mask = inlier_points, np.ones(int(n_inliers), dtype=bool)
    order = rng.permutation(len(points))
    return Dataset(points[order], mask[order], float(slope), float(intercept), (xmin, xmax))
