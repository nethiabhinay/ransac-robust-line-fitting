# Robust Line Fitting: RANSAC vs Ordinary Least Squares

A college-level Python computer-vision experiment that compares a classical OLS baseline with a manually implemented RANSAC estimator on synthetic 2D data containing Gaussian-noisy inliers and randomly distributed outliers.

## Improvements in this version
- Interactive Streamlit dashboard with controls for sample counts, slope, intercept, noise, threshold, confidence, iterations and seed.
- Adaptive RANSAC iteration limit based on the current best consensus ratio and requested confidence.
- Correct normalized perpendicular-distance scoring, final inlier refit and consistent mask recalculation.
- Input validation for malformed arrays, non-finite values and invalid parameters.
- Expanded evaluation: slope/intercept error, prediction MAE/RMSE, precision, recall, F1, accuracy and execution time.
- Refreshed high-resolution charts and automated tests.

## Requirements
Python 3.10+ recommended.

```bash
python -m venv .venv
# Windows:
.venv\Scripts\activate
# macOS/Linux:
source .venv/bin/activate
python -m pip install -r requirements.txt
```

## Run the command-line experiment
```bash
python main.py
python main.py --inliers 150 --outliers 80 --noise 1.2 --threshold 3 --iterations 1000 --confidence 0.995 --seed 21
```
Charts are written to `outputs/`. Use `--no-plots` to skip chart generation. Run `python main.py --help` to see every option.

## Run the interactive dashboard
```bash
streamlit run app.py
```
Streamlit opens the dashboard in your browser. Change the controls and select **Run experiment** to regenerate the dataset, models, metrics and charts.

## Run tests
```bash
pytest -v
```

## Project structure
```text
ransac_line_fitting_upgraded/
├── app.py
├── data_generator.py
├── line_fitting.py
├── main.py
├── metrics.py
├── visualization.py
├── requirements.txt
├── pytest.ini
├── README.md
├── PROJECT_GUIDE_AND_VIVA_PREP.md
└── tests/
    └── test_project.py
```

## How it works
1. The generator samples inliers along `y = mx + c` and adds zero-mean Gaussian noise.
2. It samples outliers over a wider vertical region, labels them for evaluation, then shuffles all points.
3. OLS fits all points by minimizing squared vertical residuals.
4. RANSAC repeatedly samples two points, creates a normalized line hypothesis, and counts points within an orthogonal distance threshold.
5. The best consensus set is refined with OLS; the returned mask is recalculated against the refined line.
6. Metrics compare each estimate with the known synthetic ground truth. Ground-truth parameters and labels are never passed into either fitting algorithm.

## Important interpretation
OLS is not inherently a bad method: it is a strong baseline when the noise model is appropriate and outliers are absent or controlled. RANSAC is useful when a dataset contains a substantial number of gross outliers. RANSAC remains sensitive to its distance threshold, sample quality and iteration budget. This demo models a non-vertical line in slope-intercept form; a vertical-line model would require a general implicit-line representation.

## Viva points
- **Why two points?** Two distinct points determine a 2D line.
- **Why normalize A, B, C?** Then `|Ax + By + C|` is the true perpendicular distance.
- **What is consensus?** The set of observations whose distances to a candidate model are within the threshold.
- **Why refit?** The two-point hypothesis is noisy; least squares over the consensus set gives a more stable estimate.
- **What does confidence mean?** Under independent random sampling and the current inlier-ratio estimate, it is the target probability of drawing at least one all-inlier minimal sample.
- **Why report precision and recall?** They measure false inlier assignments and missed true inliers, respectively.
