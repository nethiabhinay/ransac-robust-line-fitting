import numpy as np
import pytest
from data_generator import generate_line_data
from line_fitting import fit_ols, fit_ransac, required_ransac_iterations
from metrics import evaluate_model

def test_generator_counts_and_reproducibility():
    a=generate_line_data(50,20,seed=7); b=generate_line_data(50,20,seed=7)
    assert a.n_points==70 and np.array_equal(a.points,b.points)
    assert np.array_equal(a.inlier_mask,b.inlier_mask)

def test_ols_clean_line():
    x=np.linspace(-5,5,50); p=np.c_[x,3*x+2]
    m=fit_ols(p); assert m.slope==pytest.approx(3); assert m.intercept==pytest.approx(2)

def test_ransac_recovers_line_with_outliers():
    d=generate_line_data(120,60,slope=1.7,intercept=4,noise_std=.5,seed=19)
    m=fit_ransac(d.points,.9,1000,seed=19)
    assert m.slope==pytest.approx(1.7,abs=.12)
    assert m.intercept==pytest.approx(4,abs=.8)
    assert m.inlier_count>=100

def test_ransac_reproducible():
    d=generate_line_data(80,30,seed=11)
    a=fit_ransac(d.points,2.5,300,seed=9); b=fit_ransac(d.points,2.5,300,seed=9)
    assert a.slope==b.slope and np.array_equal(a.inlier_mask,b.inlier_mask)

def test_iteration_bound():
    assert required_ransac_iterations(.7,.99)==pytest.approx(7,abs=1)

@pytest.mark.parametrize("kwargs",[
    {"n_inliers":1},{"n_outliers":-1},{"noise_std":-1},{"x_range":(2,2)}
])
def test_generator_rejects_invalid(kwargs):
    with pytest.raises(ValueError): generate_line_data(**kwargs)

def test_ols_rejects_vertical_data():
    with pytest.raises(ValueError): fit_ols(np.array([[1,2],[1,3],[1,4]]))

def test_metrics_are_bounded():
    d=generate_line_data(40,10,seed=3); m=fit_ransac(d.points,2.5,300,seed=3)
    result=evaluate_model(m,d)
    assert 0<=result.inlier_precision<=1
    assert 0<=result.inlier_recall<=1
