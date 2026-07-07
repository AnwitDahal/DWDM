import numpy as np
import math

class LinearRegression:
    """Simple Linear Regression implementation from scratch."""
    
    def __init__(self):
        self.coef_ = None
        self.intercept_ = None
    
    def fit(self, X, y):
        """
        Fit the linear regression model.
        X: list of lists [[x1], [x2], ...]
        y: list of target values
        """
        # Convert to numpy arrays for easier computation
        X = np.array(X).flatten()
        y = np.array(y)
        n = len(X)
        
        # Calculate means
        x_mean = np.mean(X)
        y_mean = np.mean(y)
        
        # Calculate slope (coefficient)
        numerator = np.sum((X - x_mean) * (y - y_mean))
        denominator = np.sum((X - x_mean) ** 2)
        
        if denominator == 0:
            print("Error: All X values are identical.")
            return
        
        self.coef_ = numerator / denominator
        
        # Calculate intercept
        self.intercept_ = y_mean - self.coef_ * x_mean
    
    def predict(self, X):
        """Predict y values for given X."""
        X = np.array(X)
        return self.coef_ * X.flatten() + self.intercept_

# Main program
print("\n" + "="*50)
print(" LINEAR REGRESSION (From Scratch)")
print("="*50)

# Take input from user
try:
    n = int(input("\nEnter number of data points: "))
    if n <= 0:
        print("Error: Number of data points must be positive.")
        exit()
except ValueError:
    print("Error: Please enter a valid integer.")
    exit()

X = []
y = []

# Input values
print("\nEnter data points:")
for i in range(n):
    try:
        x_value = float(input(f"  X{i+1}: "))
        y_value = float(input(f"  Y{i+1}: "))
        X.append([x_value])
        y.append(y_value)
    except ValueError:
        print("Error: Please enter numeric values.")
        exit()

# Create and train model
model = LinearRegression()
model.fit(X, y)

# Check if model was trained successfully
if model.coef_ is None:
    exit()

# Output equation
print("\n" + "="*50)
print(" REGRESSION RESULTS")
print("="*50)
print(f"\nEquation: y = {model.coef_:.4f}x + {model.intercept_:.4f}")

# Show some statistics
X_flat = np.array(X).flatten()
y_pred = model.predict(X)

print(f"\nData Statistics:")
print(f"  Number of points: {n}")
print(f"  X range: [{min(X_flat):.2f}, {max(X_flat):.2f}]")
print(f"  Y range: [{min(y):.2f}, {max(y):.2f}]")

# Calculate and show R-squared
y_mean = np.mean(y)
ss_tot = np.sum((y - y_mean) ** 2)
ss_res = np.sum((y - y_pred) ** 2)
r2 = 1 - (ss_res / ss_tot) if ss_tot != 0 else 0
print(f"  R-squared: {r2:.4f} ({r2*100:.2f}%)")

# Prediction
print("\n" + "-"*50)
try:
    test = float(input("Enter value to predict: "))
    prediction = model.predict([[test]])[0]
    print(f"\nPrediction for x = {test:.2f}: {prediction:.4f}")
except ValueError:
    print("Error: Please enter a numeric value.")
    exit()

# Show predictions for all data points
print("\n" + "-"*50)
print("Detailed Predictions:")
print(f"  {'X':>8} {'Actual Y':>10} {'Predicted Y':>12} {'Residual':>12}")
print("  " + "-"*45)
for i in range(n):
    residual = y[i] - y_pred[i]
    print(f"  {X_flat[i]:>8.2f} {y[i]:>10.2f} {y_pred[i]:>12.4f} {residual:>12.4f}")

print("\n" + "="*50)