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

# END
