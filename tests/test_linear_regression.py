import numpy as np
import pytest
from src.ml.linear_regression import LinearRegression

def make_linear_data(n=10, w=2, b=1, seed=0):
    """Generate simple y = 2x data."""
    rng = np.random.default_rng(seed)
    X = rng.uniform(0, 10, (n, 1))
    y = w * X[:, 0] + b
    return X, y


def test_fit_computes_correct_parameters():
    X, y = make_linear_data(w=2, b=1)
    model = LinearRegression().fit(X, y)
    w_expected = [1, 2] # w[0]=b=1, w[1]=w=2 for the params is expected
    np.testing.assert_allclose(model.params, w_expected, rtol=1e-6)


def test_predict_returns_expected_values():
    X, y = make_linear_data(n=100)
    model = LinearRegression().fit(X, y)
    preds = model.predict(X)
    np.testing.assert_allclose(preds, y, rtol=1e-6)


def test_evaluate_works_with_custom_metric():
    X, y = make_linear_data()
    model = LinearRegression().fit(X, y)

    def mae(y_true, y_pred):
        return np.mean(np.abs(y_true - y_pred))

    score = model.evaluate(X, y, metric_fn=mae)
    assert score >= 0
    assert isinstance(score, float)


def test_evaluate_raises_without_metric():
    X, y = make_linear_data()
    model = LinearRegression().fit(X, y)

    with pytest.raises(ValueError):
        model.evaluate(X, y)
