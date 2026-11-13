# RANSAC Robust Line Fitting

Outlier handling in camera calibration: implementing **RANSAC** (Random Sample Consensus) for robust 2D line fitting and comparing it with **Ordinary Least Squares (OLS)**.

**Course:** DS307 – Computer Vision | Assignment 2
**Institution:** Faculty of Science & Technology, IFHE University, Hyderabad
**Faculty:** P. Chakradhar

## Team

| Name | Registration No. |
|------|------------------|
| Abhinay Nethi | 24STUCHH010706 |
| Nitish Chekuru | 24STUCHH010606 |
| Injiam Rishwanth | 24STUCHH010714 |

## Overview

Points detected in camera images often contain noise and wrong measurements (outliers). OLS uses every point, so far-away outliers pull the fitted line away from the true pattern. RANSAC instead searches for the line that the largest group of points agrees with, which makes it much less sensitive to outliers.

This project generates a synthetic dataset with a known true line, fits it with both OLS and RANSAC, and evaluates how well each method recovers the line.

## Dataset (default settings)

| Setting | Value |
|---------|-------|
| True line | y = 2.00x + 5.00 |
| Inlier points | 100 |
| Outlier points | 40 (28.6%) |
| Total points | 140 |
| Gaussian noise (std) | 1.0 |
| X range | 0 to 20 |
| Random seed | 42 |

## How RANSAC works

1. Randomly pick two different points.
2. Build a candidate line through them.
3. Compute the perpendicular distance of every point to the line:
   `d = |Ax + By + C| / sqrt(A² + B²)`
4. Points within the distance threshold (2.5) count as inliers.
5. Repeat for 100 iterations and keep the candidate with the most inliers.
6. Refit the final line with OLS on that inlier set.

## Project structure

| File | Purpose |
|------|---------|
| `data_generator.py` | Creates the synthetic dataset and keeps the true inlier labels |
| `line_fitting.py` | OLS, line from two points, distance calculation, RANSAC |
| `metrics.py` | Slope/intercept error, MAE, RMSE, precision, recall, F1, accuracy |
| `visualization.py` | Scatter plot and comparison graph |
| `main.py` | Runs the full pipeline and prints the results |

## Installation and usage

Requires Python 3 with NumPy, Matplotlib and pytest.

```bash
pip install numpy matplotlib pytest
python main.py
```

To run the tests:

```bash
pytest
```

The program prints a comparison table in the console and saves the result graphs.

## Results (default run)

| Metric | OLS | RANSAC |
|--------|-----|--------|
| Estimated slope | 0.9934 | **1.9680** |
| Estimated intercept | 13.1544 | **5.1408** |
| Slope absolute error | 1.0066 | **0.0320** |
| Intercept absolute error | 8.1544 | **0.1408** |
| Prediction MAE | 5.2245 | **0.2108** |
| Prediction RMSE | 6.1293 | **0.2581** |
| Inlier precision | N/A | 95.2% |
| Inlier recall | 100%* | 100% |
| Inlier F1-score | N/A | 97.6% |
| Classification accuracy | 71.4%* | 96.4% |

\*OLS fits all points and does not detect outliers, so precision and F1 are not available. Its 71.4% is simply the share of true inliers in the dataset.

**Takeaway:** RANSAC recovered y = 1.9680x + 5.1408, very close to the true line y = 2x + 5, while OLS gave y = 0.9934x + 13.1544 because the outliers pulled it off. RANSAC marked 105 points as inliers (100 genuine, 5 outliers).

## Limitations and future work

- Uses synthetic 2D points, not points from real camera images.
- Results depend on the distance threshold and number of iterations.
- Demonstrates line fitting only, not full camera calibration.
- Future work: test on real road-edge or camera points, choose the threshold automatically, compare with other robust methods, and extend to circles, planes or homography estimation.

## References

1. M. A. Fischler and R. C. Bolles, "Random Sample Consensus: A Paradigm for Model Fitting with Applications to Image Analysis and Automated Cartography," *Communications of the ACM*, vol. 24, no. 6, pp. 381–395, 1981.
2. NumPy documentation: https://numpy.org/doc/
3. Matplotlib documentation: https://matplotlib.org/stable/
4. pytest documentation: https://docs.pytest.org/
