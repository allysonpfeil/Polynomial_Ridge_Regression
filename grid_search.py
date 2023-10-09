# Add this code to main.py to determine the best hyperparameters
# Define the parameter grid for the grid search
param_grid = {
    'polynomialfeatures__degree': [0, 1, 2, 3, 4],  # degrees of polynomial
    'ridge__alpha': [0.1, 0.5, 1.0, 5.0, 10.0, 20.0]  # alpha values to explore
}

# Create the pipeline with Polynomial Features and Ridge Regression
model = make_pipeline(
    PolynomialFeatures(),
    StandardScaler(),
    Ridge()
)

# Create the GridSearchCV object
grid_search = GridSearchCV(model
                           param_grid
                           cv=5,
                           scoring='neg_mean_squared_error',
                           verbose=1)

# Fit the grid search to the data
grid_search.fit(X_train_scaled, y_train)

# Get the best hyperparameters from the grid search
best_degree = grid_search.best_params_['polynomialfeatures__degree']
best_alpha = grid_search.best_params_['ridge__alpha']

# Fit the final model with the best hyperparameters
model = make_pipeline(
                      PolynomialFeatures(degree=best_degree),
                      StandardScaler(),
                      Ridge(alpha=best_alpha)
)

model.fit(X_train_scaled, y_train)

# Make predictions on the test set
y_pred = model.predict(X_test_scaled)

# Evaluate the model
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print("Best Degree:", best_degree)
print("Best Alpha:", best_alpha)
print("Mean Squared Error (with best hyperparameters): {:.2f}".format(mse))
print("R-squared (with best hyperparameters): {:.2f}".format(r2))

# After determining the best hyperparameters for your dataset using this code, you can then just plug those into main.py, or continue coding independently
