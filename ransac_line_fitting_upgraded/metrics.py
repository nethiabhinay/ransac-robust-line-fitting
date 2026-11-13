"""Evaluation metrics for fitted lines and inlier classification."""
from dataclasses import dataclass
from typing import Optional, Dict, Any, Tuple
import numpy as np
from data_generator import Dataset
from line_fitting import LineModel

@dataclass
class LineFittingMetrics:
    model_name: str
    estimated_slope: float
    estimated_intercept: float
    slope_error: float
    intercept_error: float
    mae: float
    rmse: float
    detected_inliers: int
    detected_outliers: int
    inlier_precision: Optional[float] = None
    inlier_recall: Optional[float] = None
    inlier_f1: Optional[float] = None
    classification_accuracy: Optional[float] = None
    execution_time_ms: Optional[float] = None
    def to_dict(self) -> Dict[str, Any]:
        return {k: (round(v*100, 2) if k in {"inlier_precision","inlier_recall","inlier_f1","classification_accuracy"} and v is not None
                    else round(v, 4) if isinstance(v, (float, np.floating)) else v)
                for k, v in self.__dict__.items()}

def evaluate_model(model, dataset, model_name="Model", eval_samples=500, execution_time_ms=None):
    if eval_samples < 2: raise ValueError("eval_samples must be >= 2.")
    x = np.linspace(*dataset.x_range, eval_samples)
    residual = model.predict(x)-(dataset.true_slope*x+dataset.true_intercept)
    mae, rmse = float(np.mean(np.abs(residual))), float(np.sqrt(np.mean(residual**2)))
    precision = recall = f1 = accuracy = None
    if model.inlier_mask is not None:
        pred = np.asarray(model.inlier_mask, dtype=bool)
        if pred.shape != dataset.inlier_mask.shape: raise ValueError("Model mask shape does not match dataset.")
        truth = dataset.inlier_mask
        tp, fp = int((truth & pred).sum()), int((~truth & pred).sum())
        fn, tn = int((truth & ~pred).sum()), int((~truth & ~pred).sum())
        precision = tp/(tp+fp) if tp+fp else 0.0
        recall = tp/(tp+fn) if tp+fn else 0.0
        f1 = 2*precision*recall/(precision+recall) if precision+recall else 0.0
        accuracy = (tp+tn)/len(truth)
    detected = int(model.inlier_mask.sum()) if model.inlier_mask is not None else dataset.n_points
    return LineFittingMetrics(model_name, model.slope, model.intercept,
        abs(model.slope-dataset.true_slope), abs(model.intercept-dataset.true_intercept),
        mae, rmse, detected, dataset.n_points-detected, precision, recall, f1, accuracy, execution_time_ms)

def compare_models(ols_model, ransac_model, dataset, ols_time_ms=None, ransac_time_ms=None):
    return (evaluate_model(ols_model, dataset, "Ordinary Least Squares", execution_time_ms=ols_time_ms),
            evaluate_model(ransac_model, dataset, "RANSAC", execution_time_ms=ransac_time_ms))
