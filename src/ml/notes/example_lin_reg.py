import numpy as np

X = np.array([[1, 2],
              [2, 0],
              [3, 1]])
y = np.array([1, 2, 3])

XTX = X.T @ X      # shape (2,2)
XTy = X.T @ y      # shape (2,)

w = np.linalg.solve(XTX, XTy)
print(w)

# === Predict ===
y_pred = X @ w     # (3,) — predicted values
print("Predictions:", y_pred)

# === Compare with actual ===
print("Actual:", y)
print("Residuals:", y - y_pred)
