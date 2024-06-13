import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error, r2_score
from keras.datasets import boston_housing
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.pipeline import make_pipeline
import numpy as np

(x_train, y_train), (x_test, y_test) = boston_housing.load_data()

columns = ['CRIM', 'ZN', 'INDUS', 'CHAS', 'NOX', 'RM', 'AGE', 'DIS', 'RAD', 'TAX', 'PTRATIO', 'B', 'LSTAT']

train_df = pd.DataFrame(data=x_train, columns=columns)
test_df = pd.DataFrame(data=x_test, columns=columns)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(train_df)
X_test_scaled = scaler.transform(test_df)

degree = 2  
alpha = 10  

model = make_pipeline(
    PolynomialFeatures(degree=degree),
    StandardScaler(),
    Ridge(alpha=alpha)
)

model.fit(X_train_scaled, y_train)

y_pred = model.predict(X_test_scaled)
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print("Mean Squared Error: {:.2f}".format(mse))
print("R-squared: {:.2f}".format(r2))

# The following code is optional and is just for fun
coefficients = model.named_steps['ridge'].coef_
intercept = model.named_steps['ridge'].intercept_

function = "y = {:.2f}".format(intercept)
for degree, coef in enumerate(coefficients):
    function += " + {:.2f} * x^{}".format(coef, degree)
print("Polynomial Regression Function:")
print(function)

plt.figure(figsize=(10, 6))
plt.scatter(y_test, y_pred)
plt.xlabel("Actual Housing Price")
plt.ylabel("Predicted Housing Price")
plt.title("Actual vs. Predicted Housing Price")

plt.plot(np.unique(y_test), np.poly1d(np.polyfit(y_test, y_pred, 1))(np.unique(y_test)), color='red', linewidth=2)

plt.show()
