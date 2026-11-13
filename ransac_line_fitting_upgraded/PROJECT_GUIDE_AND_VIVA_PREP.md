# Project Guide & Viva Preparation
## Robust Line Fitting and Outlier Handling using RANSAC

### 1. Project objective
Compare Ordinary Least Squares (OLS) with a manually implemented Random Sample Consensus (RANSAC) line estimator when 2D observations contain both noisy inliers and gross outliers.

### 2. Execution
Install dependencies and activate the virtual environment as described in `README.md`.

```bash
python main.py
pytest -v
streamlit run app.py
```

### 3. Data flow
`generate_line_data()` returns a `Dataset` containing:
- `points`: an `(N, 2)` NumPy array.
- `inlier_mask`: labels retained for evaluation only.
- `true_slope`, `true_intercept`, and `x_range`: synthetic ground truth for evaluation.

Neither OLS nor RANSAC receives the ground-truth labels or parameters.

### 4. Algorithms
**OLS:** minimizes the sum of squared vertical residuals:
\[
S(m,c)=\sum_i(y_i-(mx_i+c))^2
\]
\[
m=\frac{\sum_i(x_i-\bar{x})(y_i-\bar{y})}{\sum_i(x_i-\bar{x})^2},
\quad c=\bar{y}-m\bar{x}
\]
Large residuals have quadratic influence, so outliers can shift the fitted line.

**RANSAC:**
1. Randomly select two distinct points.
2. Build a normalized implicit line `Ax + By + C = 0`.
3. Compute perpendicular distances `d = |Ax + By + C|`.
4. Classify points with `d <= threshold` as consensus inliers.
5. Keep the hypothesis with the largest consensus.
6. Refit the line using the consensus points.
7. Recalculate the final mask using the refined model.

The normalized distance formula is valid because `sqrt(A²+B²)=1`.

### 5. Adaptive iteration formula
For minimal sample size `s=2`, estimated inlier ratio `w`, and desired confidence `p`:
\[
N=\frac{\ln(1-p)}{\ln(1-w^s)}
\]
The implementation uses the best consensus ratio found so far to reduce the remaining iteration limit. This is an estimate, not an absolute guarantee; it assumes independent random sampling and a meaningful consensus estimate.

### 6. Evaluation metrics
- **Slope/intercept absolute error:** difference from synthetic ground truth.
- **MAE:** average absolute prediction deviation over a uniform x-grid.
- **RMSE:** square root of mean squared prediction deviation.
- **Precision:** fraction of RANSAC-labelled inliers that are true inliers.
- **Recall:** fraction of all true inliers recovered.
- **F1:** harmonic mean of precision and recall.
- **Classification accuracy:** fraction of all points correctly labelled.
- **Execution time:** measured around each fitting call; timing varies by machine and run.

### 7. Dashboard
The Streamlit interface allows users to change inlier/outlier counts, line parameters, noise, RANSAC threshold, iteration budget, confidence and random seed. It regenerates plots and evaluation metrics for each run.

### 8. Limitations
- The model uses `y = mx + c`; vertical lines are not represented.
- RANSAC is sensitive to the distance threshold and the quality of random samples.
- A poor consensus ratio can require many iterations.
- Synthetic results do not guarantee identical performance on real-world data.
- Ground-truth classification metrics are available only because this is a labelled synthetic experiment.

### 9. Viva questions
**Q1. What is RANSAC?** Random Sample Consensus, a robust fitting method that searches for a model supported by the largest subset of observations.

**Q2. Why does OLS struggle with outliers?** Squared residuals give distant observations disproportionate influence.

**Q3. Why sample two points?** Two distinct points define a 2D line.

**Q4. Why perpendicular distance?** It measures shortest geometric distance to the line rather than only vertical residual.

**Q5. What is the threshold?** The maximum perpendicular distance for a point to count as an inlier.

**Q6. Why refit after consensus?** To improve the parameter estimate using all points supporting the best hypothesis.

**Q7. What is precision?** `TP / (TP + FP)`; it measures how many detected inliers are actually inliers.

**Q8. What is recall?** `TP / (TP + FN)`; it measures how many true inliers were recovered.

**Q9. What does the confidence setting do?** It controls the theoretical target probability of drawing at least one all-inlier sample under the estimated inlier ratio.

**Q10. Is RANSAC always better?** No. OLS can be preferable for clean Gaussian-noise data; RANSAC is useful when gross outliers are present.

### 10. Conclusion
The project demonstrates the trade-off between least-squares fitting and consensus-based robust estimation. RANSAC reduces outlier influence by identifying a supporting subset and refining the model using that subset.
