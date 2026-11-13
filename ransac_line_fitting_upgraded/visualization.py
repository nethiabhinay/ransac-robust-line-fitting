"""Clean, high-resolution charts for RANSAC/OLS experiments."""
import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

def _style():
    plt.rcParams.update({"font.family":"DejaVu Sans","font.size":10,
        "axes.titleweight":"bold","axes.spines.top":False,"axes.spines.right":False,
        "figure.facecolor":"#f8fafc","axes.facecolor":"#ffffff","savefig.facecolor":"#f8fafc"})

def plot_line_fitting(dataset, ols_model, ransac_model, output_path="ransac_vs_ols_fitting.png", show_plot=False):
    _style()
    fig, ax = plt.subplots(figsize=(12, 7), dpi=170)
    ax.scatter(*dataset.inliers.T, s=34, color="#2563eb", alpha=.78, label=f"Ground-truth inliers ({dataset.n_inliers})", zorder=3)
    if dataset.n_outliers:
        ax.scatter(*dataset.outliers.T, s=55, marker="x", linewidth=1.7, color="#ef4444", alpha=.82,
                   label=f"Ground-truth outliers ({dataset.n_outliers})", zorder=4)
    xmin, xmax = dataset.x_range
    xx = np.linspace(xmin-(xmax-xmin)*.04, xmax+(xmax-xmin)*.04, 300)
    ax.plot(xx, dataset.true_slope*xx+dataset.true_intercept, "--", color="#111827", lw=2.3,
            label=f"Ground truth  y={dataset.true_slope:.2f}x+{dataset.true_intercept:.2f}")
    ax.plot(xx, ols_model.predict(xx), "-.", color="#f97316", lw=2.6,
            label=f"OLS  y={ols_model.slope:.2f}x+{ols_model.intercept:.2f}")
    ax.plot(xx, ransac_model.predict(xx), "-", color="#16a34a", lw=2.8,
            label=f"RANSAC  y={ransac_model.slope:.2f}x+{ransac_model.intercept:.2f}")
    ax.set(title="Robust Line Fitting | RANSAC vs OLS", xlabel="X coordinate", ylabel="Y coordinate")
    ax.grid(True, linestyle=":", alpha=.35)
    ax.legend(loc="upper left", frameon=True, framealpha=.96, fontsize=9)
    ax.text(.985,.035,f"RANSAC consensus: {ransac_model.inlier_count}/{dataset.n_points}\n"
            f"Outlier contamination: {dataset.outlier_percentage:.1f}%\n"
            f"Iterations used: {ransac_model.n_iterations}",
            transform=ax.transAxes,ha="right",va="bottom",
            bbox={"boxstyle":"round,pad=.55","facecolor":"#eff6ff","edgecolor":"#bfdbfe"})
    fig.tight_layout()
    path=os.path.abspath(output_path); os.makedirs(os.path.dirname(path),exist_ok=True)
    fig.savefig(path,dpi=220,bbox_inches="tight")
    if show_plot: plt.show()
    plt.close(fig); return path

def plot_comparison_summary(dataset, ols_metrics, ransac_metrics, output_path="comparison_summary.png", show_plot=False):
    _style()
    fig,(left,right)=plt.subplots(1,2,figsize=(13,5.6),dpi=160)
    xx=np.linspace(*dataset.x_range,250)
    left.plot(xx,dataset.true_slope*xx+dataset.true_intercept,"--",color="#111827",lw=2,label="Ground truth")
    left.plot(xx,ols_metrics.estimated_slope*xx+ols_metrics.estimated_intercept,"-.",color="#f97316",lw=2.5,label="OLS")
    left.plot(xx,ransac_metrics.estimated_slope*xx+ransac_metrics.estimated_intercept,"-",color="#16a34a",lw=2.5,label="RANSAC")
    left.set(title="Estimated line comparison",xlabel="X coordinate",ylabel="Y coordinate"); left.grid(alpha=.3,linestyle=":"); left.legend()
    labels=["Slope error","Intercept error","Prediction RMSE"]
    a=[ols_metrics.slope_error,ols_metrics.intercept_error,ols_metrics.rmse]
    b=[ransac_metrics.slope_error,ransac_metrics.intercept_error,ransac_metrics.rmse]
    x=np.arange(3); w=.34
    bars1=right.bar(x-w/2,a,w,label="OLS",color="#f97316")
    bars2=right.bar(x+w/2,b,w,label="RANSAC",color="#16a34a")
    right.set_xticks(x,labels,rotation=10); right.set_title("Error metrics (lower is better)"); right.grid(axis="y",alpha=.3,linestyle=":")
    right.legend()
    for bars in (bars1,bars2):
        for bar in bars:
            right.annotate(f"{bar.get_height():.3f}",(bar.get_x()+bar.get_width()/2,bar.get_height()),
                           xytext=(0,3),textcoords="offset points",ha="center",fontsize=8)
    fig.tight_layout()
    path=os.path.abspath(output_path); os.makedirs(os.path.dirname(path),exist_ok=True)
    fig.savefig(path,dpi=220,bbox_inches="tight")
    if show_plot: plt.show()
    plt.close(fig); return path
