import numpy as np
from typing_extensions import Self, TypeVar, Sequence, Any, Callable
import matplotlib.pyplot as plt

self = TypeVar("model", bound="LinearRegression")

class LinearRegression:
    """
    Ordinary Least Squares (OLS) linear regression model.

    OLS linear regression fits a linear model to the dataset.
    
    Assumptions
    -----------
    - LINEAR: The data is linear. A line goes through the data points
    better than any other type of curve.
    - NOT MULTICOLLINEAR: Each featture in the dataset is independent 
    from the other features, that is, no two features are highly
    correlated.
    - NO AUTOCORRELATION: An error can't be used to predict another 
    error. No errors are correlated.

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
        In the end, the fitted parameters will look like:

        self.params = [b, w1, w2, ..., wn]

        where self.params[0] is the bias/intercept,
        and w1, w2, ..., wn are the weights for each feature.

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
        
        X_bias = np.hstack([np.ones((X.shape[0], 1)), X])
        
        self.params = np.linalg.solve(X_bias.T @ X_bias, X_bias.T @ y)
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
        return X @ self.params[1:] + self.params[0]

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
    