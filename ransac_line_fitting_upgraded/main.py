"""Command-line experiment runner for robust line fitting."""
import argparse, os, time
from data_generator import generate_line_data
from line_fitting import fit_ols, fit_ransac, required_ransac_iterations
from metrics import compare_models
from visualization import plot_line_fitting, plot_comparison_summary

def parse_arguments():
    p=argparse.ArgumentParser(description="Compare OLS and manual RANSAC on synthetic 2D points.")
    p.add_argument("--inliers",type=int,default=100)
    p.add_argument("--outliers",type=int,default=40)
    p.add_argument("--slope",type=float,default=2.0)
    p.add_argument("--intercept",type=float,default=5.0)
    p.add_argument("--noise",type=float,default=1.0)
    p.add_argument("--threshold",type=float,default=2.5)
    p.add_argument("--iterations",type=int,default=500)
    p.add_argument("--confidence",type=float,default=.99)
    p.add_argument("--seed",type=int,default=42)
    p.add_argument("--output-dir",default="outputs")
    p.add_argument("--no-plots",action="store_true")
    return p.parse_args()

def main():
    a=parse_arguments()
    if a.inliers<2 or a.outliers<0 or a.iterations<1: raise SystemExit("Counts and iterations are outside valid ranges.")
    ds=generate_line_data(a.inliers,a.outliers,a.slope,a.intercept,a.noise,seed=a.seed)
    t=time.perf_counter(); ols=fit_ols(ds.points); ols_ms=(time.perf_counter()-t)*1000
    t=time.perf_counter(); ran=fit_ransac(ds.points,a.threshold,a.iterations,seed=a.seed,confidence=a.confidence); ran_ms=(time.perf_counter()-t)*1000
    om,rm=compare_models(ols,ran,ds,ols_ms,ran_ms)
    print("\nROBUST LINE FITTING — EXPERIMENT REPORT")
    print("="*82)
    print(f"Ground truth: y = {ds.true_slope:.3f}x + {ds.true_intercept:.3f} | Points: {ds.n_points} | Outliers: {ds.outlier_percentage:.1f}%")
    print(f"{'Metric':<27}{'OLS':>22}{'RANSAC':>22}")
    print("-"*71)
    rows=[("Slope",om.estimated_slope,rm.estimated_slope),("Intercept",om.estimated_intercept,rm.estimated_intercept),
          ("Slope absolute error",om.slope_error,rm.slope_error),("Intercept absolute error",om.intercept_error,rm.intercept_error),
          ("Prediction MAE",om.mae,rm.mae),("Prediction RMSE",om.rmse,rm.rmse),
          ("Detected inliers",om.detected_inliers,rm.detected_inliers),("Execution time (ms)",ols_ms,ran_ms),
          ("Precision (%)",None if om.inlier_precision is None else om.inlier_precision*100,rm.inlier_precision*100),
          ("Recall (%)",None if om.inlier_recall is None else om.inlier_recall*100,rm.inlier_recall*100),
          ("F1 score (%)",None if om.inlier_f1 is None else om.inlier_f1*100,rm.inlier_f1*100),
          ("Classification accuracy (%)",None if om.classification_accuracy is None else om.classification_accuracy*100,rm.classification_accuracy*100)]
    for name,x,y in rows:
        print(f"{name:<27}{'N/A' if x is None else f'{x:.4f}':>22}{y:>22.4f}")
    theoretical=required_ransac_iterations(ds.n_inliers/ds.n_points,a.confidence)
    print(f"\\nRANSAC iterations used: {ran.n_iterations}/{a.iterations}; theoretical bound from known demo ratio: {theoretical}")
    print("Note: the ground-truth ratio is used here only for experiment reporting, never by the fitter.")
    if not a.no_plots:
        os.makedirs(a.output_dir,exist_ok=True)
        print("\nCharts:")
        print(plot_line_fitting(ds,ols,ran,os.path.join(a.output_dir,"ransac_vs_ols_fitting.png")))
        print(plot_comparison_summary(ds,om,rm,os.path.join(a.output_dir,"comparison_summary.png")))
    print("\nDone.")
    return 0
if __name__=="__main__": raise SystemExit(main())
