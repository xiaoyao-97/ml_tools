"""可以为负值
import numpy as np
from scipy.optimize import minimize

def mse_loss(coeffs, X, y):
    predictions = X @ coeffs
    return ((predictions - y) ** 2).mean()

initial_coeffs = np.zeros(X_train[preds].shape[1])

result = minimize(mse_loss, initial_coeffs, args=(X_train[preds], y_train))

optimal_coeffs = result.x
print("Optimal coefficients:", optimal_coeffs)

# predictions = X_test @ optimal_coeffs
"""

"""只能为正值
import numpy as np
from scipy.optimize import minimize

def mse_loss(coeffs, X, y):
    predictions = X @ coeffs
    return ((predictions - y) ** 2).mean()

initial_coeffs = np.zeros(X_train[preds2].shape[1])

bounds = [(0, None) for _ in range(X_train[preds2].shape[1])]

result = minimize(mse_loss, initial_coeffs, args=(X_train[preds2], y_train), bounds = bounds)

optimal_coeffs = result.x
print("Optimal coefficients:", optimal_coeffs)

# predictions = X_test @ optimal_coeffs
"""

