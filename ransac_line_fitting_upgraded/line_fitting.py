"""OLS baseline and a manual, confidence-adaptive RANSAC line fitter."""
from dataclasses import dataclass
from typing import Optional, Tuple
import math
import numpy as np

@dataclass
class LineModel:
    slope: float
    intercept: float
    inlier_mask: Optional[np.ndarray] = None
    n_iterations: Optional[int] = None
    inlier_count: Optional[int] = None
    def predict(self, x):
        return self.slope*np.asarray(x, dtype=np.float64)+self.intercept
    def perpendicular_distance(self, points):
        pts = _points(points)
        return np.abs(self.slope*pts[:, 0]-pts[:, 1]+self.intercept)/math.sqrt(self.slope**2+1.0)

def _points(points):
    pts = np.asarray(points, dtype=np.float64)
    if pts.ndim != 2 or pts.shape[1] != 2 or len(pts) < 2:
        raise ValueError("points must be a finite array with shape (N, 2), N >= 2.")
    if not np.isfinite(pts).all():
        raise ValueError("points must contain only finite values.")
    return pts

def fit_ols(points):
    pts = _points(points)
    x, y = pts[:, 0], pts[:, 1]
    xd, yd = x-x.mean(), y-y.mean()
    den = float(xd @ xd)
    if den < 1e-12:
        raise ValueError("Cannot represent a vertical line as y = m*x + c (all x values are equal).")
    m = float((xd @ yd)/den)
    c = float(y.mean()-m*x.mean())
    return LineModel(m, c, np.ones(len(pts), dtype=bool), 0, len(pts))

def compute_line_from_two_points(p1, p2) -> Optional[Tuple[float, float, float]]:
    p1, p2 = np.asarray(p1, dtype=float), np.asarray(p2, dtype=float)
    dx, dy = p2-p1
    if dx*dx+dy*dy < 1e-12: return None
    a, b, c = dy, -dx, p2[0]*p1[1]-p1[0]*p2[1]
    norm = math.hypot(a, b)
    if norm < 1e-12: return None
    return float(a/norm), float(b/norm), float(c/norm)

def required_ransac_iterations(inlier_ratio, confidence=0.99, sample_size=2):
    """Return theoretical iteration bound; returns infinity if success is impossible."""
    if not 0 < confidence < 1: raise ValueError("confidence must be between 0 and 1.")
    if not 0 <= inlier_ratio <= 1: raise ValueError("inlier_ratio must be in [0, 1].")
    if sample_size < 1: raise ValueError("sample_size must be >= 1.")
    good = inlier_ratio**sample_size
    if good <= 0: return math.inf
    if good >= 1: return 1
    return int(math.ceil(math.log1p(-confidence)/math.log1p(-good)))

def fit_ransac(points, distance_threshold=2.5, max_iterations=100, min_inliers=2,
               seed=42, confidence=0.99, adaptive=True):
    """Fit a line using random 2-point hypotheses and orthogonal-distance consensus."""
    pts = _points(points)
    if not np.isfinite(distance_threshold) or distance_threshold <= 0:
        raise ValueError("distance_threshold must be finite and > 0.")
    if int(max_iterations) != max_iterations or max_iterations < 1:
        raise ValueError("max_iterations must be an integer >= 1.")
    if int(min_inliers) != min_inliers or min_inliers < 2 or min_inliers > len(pts):
        raise ValueError("min_inliers must be between 2 and the number of points.")
    if not 0 < confidence < 1: raise ValueError("confidence must be between 0 and 1.")
    rng = np.random.default_rng(seed)
    best_mask, best_count, best_model = None, -1, None
    limit, executed = int(max_iterations), 0
    x, y = pts[:, 0], pts[:, 1]
    while executed < limit:
        executed += 1
        i, j = rng.choice(len(pts), 2, replace=False)
        coeff = compute_line_from_two_points(pts[i], pts[j])
        if coeff is None: continue
        a, b, c = coeff
        # Normalized coefficients make this the true perpendicular distance.
        mask = np.abs(a*x+b*y+c) <= distance_threshold
        count = int(mask.sum())
        if count > best_count:
            best_count, best_mask = count, mask
            if abs(b) > 1e-10:
                best_model = (-a/b, -c/b)
            if adaptive and count >= min_inliers:
                ratio = count/len(pts)
                bound = required_ransac_iterations(ratio, confidence, 2)
                if math.isfinite(bound):
                    limit = min(limit, max(executed, int(bound)))
    if best_mask is None or best_count < min_inliers:
        raise ValueError("RANSAC could not find a valid consensus model; increase iterations or threshold.")
    selected = pts[best_mask]
    try:
        refined = fit_ols(selected)
        slope, intercept = refined.slope, refined.intercept
    except ValueError:
        if best_model is None: raise
        slope, intercept = best_model
    # Reclassify once against the refined line for a consistent returned mask.
    model = LineModel(slope, intercept)
    final_mask = model.perpendicular_distance(pts) <= distance_threshold
    if final_mask.sum() >= 2 and not np.allclose(pts[final_mask, 0], pts[final_mask, 0][0]):
        refined = fit_ols(pts[final_mask])
        slope, intercept = refined.slope, refined.intercept
        model = LineModel(slope, intercept)
        final_mask = model.perpendicular_distance(pts) <= distance_threshold
    return LineModel(float(slope), float(intercept), final_mask, executed, int(final_mask.sum()))
