import numpy as np
import matplotlib.pyplot as plt
from DataPoints import a, b, x, y, x_mean, x_std, y_mean, y_std

#Fit in standardized space (z-scores)
x_Standardized, y_Standardized = x, y
x_Average, y_Average = x_Standardized.mean(), y_Standardized.mean()
slope_std = np.sum((x_Standardized - x_Average) * (y_Standardized - y_Average)) / np.sum((x_Standardized - x_Average) ** 2)
intercept_std = y_Average - slope_std * x_Average

#Converts coefficients back to original units
slope = (y_std / x_std) * slope_std
intercept = y_mean + y_std * intercept_std - slope * x_mean

#Quality (R^2) and Prediction
y_Prediction = intercept + slope * a
#Measures the total variance in y and the variance not explained by the model
total_Sum_of_Squares = np.sum((b - b.mean()) ** 2)
#Measure of variance not explained by the model, and how much y differs from predicted y
residual_Sum_of_Squares = np.sum((b - y_Prediction) ** 2)
r_Squared= 1 - residual_Sum_of_Squares / total_Sum_of_Squares

print(f"Intercept (β0): {intercept:.6f}")
print(f"Slope     (β1): {slope:.6f}")
print(f"R²: {r_Squared:.6f}")

#Plot
plt.figure()
plt.scatter(a, b, label="Data")
x_line = np.linspace(a.min()-1, a.max()+1, 200)
plt.plot(x_line, intercept + slope * x_line, label="LS fit")
plt.xlabel("Temperature (x)")
plt.ylabel("Net hourly electrical output (y)")
plt.title("Least Squares Linear Regression (from standardized data)")
plt.legend()
plt.grid(alpha=0.3)
plt.tight_layout()
plt.savefig("Least_Squares_Fit.png", dpi=160)
plt.show()
