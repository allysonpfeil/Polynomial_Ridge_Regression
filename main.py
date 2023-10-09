# Polynomial Ridge Regression of the Boston Housing Price dataset 
# Dataset c/o Keras
# By: Allyson Pfeil

# Import packages
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error, r2_score
from keras.datasets import boston_housing
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.pipeline import make_pipeline
import numpy as np

# Load the Boston Housing Price dataset
(x_train, y_train), (x_test, y_test) = boston_housing.load_data()

# Define the features of the dataset (can be found from printing)
columns = ['CRIM', 'ZN', 'INDUS', 'CHAS', 'NOX', 'RM', 'AGE', 'DIS', 'RAD', 'TAX', 'PTRATIO', 'B', 'LSTAT']

# Convert the dataset to a Pandas DataFrame for easier manipulation
train_df = pd.DataFrame(data=x_train, columns=columns)
test_df = pd.DataFrame(data=x_test, columns=columns)

# Data Preprocessing
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(train_df)
X_test_scaled = scaler.transform(test_df)

# Define the hyperparameters found in the grid search (see grid_search.py)
degree = 2  
alpha = 10  

# Create the pipeline with Polynomial Features and Ridge Regression
model = make_pipeline(
    PolynomialFeatures(degree=degree),
    StandardScaler(),
    Ridge(alpha=alpha)
)

model.fit(X_train_scaled, y_train)

# Make predictions on the test set
y_pred = model.predict(X_test_scaled)

# Evaluate the model
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print("Mean Squared Error: {:.2f}".format(mse))
print("R-squared: {:.2f}".format(r2))

# The following code is optional and is just for fun
# Output the function of the polynomial regression model
coefficients = model.named_steps['ridge'].coef_
intercept = model.named_steps['ridge'].intercept_

# Generate and output the function
function = "y = {:.2f}".format(intercept)
for degree, coef in enumerate(coefficients):
    function += " + {:.2f} * x^{}".format(coef, degree)
print("Polynomial Regression Function:")
print(function)

# Visualize the results with a simple linear regression line
plt.figure(figsize=(10, 6))
plt.scatter(y_test, y_pred)
plt.xlabel("Actual Housing Price")
plt.ylabel("Predicted Housing Price")
plt.title("Actual vs. Predicted Housing Price")

# Add the linear regression line
plt.plot(np.unique(y_test), np.poly1d(np.polyfit(y_test, y_pred, 1))(np.unique(y_test)), color='red', linewidth=2)

# Output the graph
plt.show()

# RESULTS 
# Mean Squared Error: 11.01
# R-squared: 0.87
# Polynomial Regression Function:
# y = 22.40 + 0.00 * x^0 + -0.17 * x^1 + 0.16 * x^2 + 0.02 * x^3 + 0.72 * x^4 + -1.26 * x^5 + 3.15 * x^6 + -0.97 * x^7 + -1.81 * x^8 + 1.00 * x^9 + -0.86 * x^10 + -0.45 * x^11 + 0.98 * x^12 + -3.35 * x^13 + 0.51 * x^14 + 0.07 * x^15 + -0.03 * x^16 + 1.49 * x^17 + -0.26 * x^18 + 0.55 * x^19 + -0.03 * x^20 + 0.26 * x^21 + -0.50 * x^22 + 0.07 * x^23 + -0.06 * x^24 + -0.20 * x^25 + 0.64 * x^26 + 0.28 * x^27 + -0.36 * x^28 + -0.10 * x^29 + 0.35 * x^30 + 0.19 * x^31 + -0.23 * x^32 + -0.23 * x^33 + -0.00 * x^34 + 0.64 * x^35 + -0.06 * x^36 + -0.20 * x^37 + -0.36 * x^38 + 0.48 * x^39 + -0.11 * x^40 + 0.98 * x^41 + 0.45 * x^42 + 0.61 * x^43 + 0.62 * x^44 + -0.17 * x^45 + 0.02 * x^46 + -0.57 * x^47 + -0.04 * x^48 + -0.86 * x^49 + 0.72 * x^50 + -1.31 * x^51 + -0.71 * x^52 + 0.07 * x^53 + -0.38 * x^54 + -0.28 * x^55 + 0.41 * x^56 + -0.16 * x^57 + -0.09 * x^58 + -0.39 * x^59 + -0.19 * x^60 + -0.08 * x^61 + -0.77 * x^62 + 0.98 * x^63 + -0.97 * x^64 + 0.11 * x^65 + -0.62 * x^66 + -0.15 * x^67 + 0.69 * x^68 + 0.66 * x^69 + -0.55 * x^70 + 0.40 * x^71 + -0.49 * x^72 + -0.85 * x^73 + -1.04 * x^74 + -0.08 * x^75 + -1.02 * x^76 + 0.04 * x^77 + -0.21 * x^78 + 1.22 * x^79 + -0.22 * x^80 + -0.10 * x^81 + -0.76 * x^82 + -1.06 * x^83 + 0.92 * x^84 + -0.65 * x^85 + -0.45 * x^86 + 0.15 * x^87 + -0.64 * x^88 + 0.63 * x^89 + -0.90 * x^90 + 0.81 * x^91 + 0.15 * x^92 + -0.07 * x^93 + -1.42 * x^94 + 0.09 * x^95 + 1.32 * x^96 + -0.58 * x^97 + -0.92 * x^98 + -0.02 * x^99 + -0.02 * x^100 + 0.13 * x^101 + -0.72 * x^102 + -0.65 * x^103 + 1.50 * x^104
# See Housing_Regression_Graph.png for the MatPlotLib output of the script

# the end 
