import numpy as np
from typing_extensions import Self, TypeVar, Sequence, Any, Callable
import matplotlib.pyplot as plt

self = TypeVar("model", bound="LinearRegression")

class LinearRegression:
    """
    Ordinary Least Squares (OLS) linear regression model.

    This model fits a linear relationship between input features `X` and target values `y`
    by minimizing the squared error: ||y - Xw||².

    Methods
    -------
    fit(X, y)
        Estimate model parameters using the normal equation.
    predict(X)
        Compute predictions using learned parameters.
    evaluate(X, y, metric_fn)
        Evaluate predictions with a custom metric function.
    """

    def __init__(self):
        self.params = None

    def fit(self, X: Sequence[Any], y: Sequence[Any]) -> Self:
        """
        Fit the model to training data using the normal equation.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Input features.
        y : array-like of shape (n_samples,)
            Target values.

        Returns
        -------
        self : LinearRegression
            Fitted model with parameters stored in `self.params`.
        """
        X = np.asarray(X)
        y = np.asarray(y)

        XTX = X.T @ X
        XTy = X.T @ y

        self.params = np.linalg.solve(XTX, XTy)
        print("Trained to get w=", self.params)
        return self

    def predict(self, X: Sequence[Any]) -> Sequence[Any]:
        """
        Predict target values using the learned parameters.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Input data.

        Returns
        -------
        y_pred : np.ndarray of shape (n_samples,)
            Predicted target values.
        """
        return X @ self.params

    def evaluate(self, X: Sequence[Any], y: Sequence[Any], metric_fn: Callable[[Sequence[Any], Sequence[Any]], float] | None = None) -> float:
        """
        Evaluate model predictions using a custom metric function.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Input features.
        y : array-like of shape (n_samples,)
            True target values.
        metric_fn : callable
            Function that takes (y_true, y_pred) and returns a scalar score.

        Returns
        -------
        score : float
            Metric value returned by `metric_fn`.

        Raises
        ------
        ValueError
            If `metric_fn` is None.
        """
        preds = self.predict(X)
        if metric_fn is None:
            raise ValueError("metric_fn must be a callable function.")
        return metric_fn(y, preds)
    