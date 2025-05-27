# Step 1: Import libraries
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split

# Step 2: Create sample dataset
# Features: [Size (sqft), Bedrooms, Age]
X = np.array([
    [1500, 3, 10],
    [1800, 4, 15],
    [2400, 3, 5],
    [3000, 5, 20],
    [1000, 2, 8],
    [1600, 3, 12],
    [2000, 4, 10],
    [2200, 4, 6],
    [1200, 2, 5],
    [1400, 3, 7]
])

# Target: Price (in $1000s)
y = np.array([250, 300, 375, 480, 200, 275, 320, 360, 220, 240])

# Step 3: Split into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Step 4: Train the Multiple Linear Regression model
model = LinearRegression()
model.fit(X_train, y_train)

# Step 5: Predict on test set
y_pred = model.predict(X_test)

# Step 6: Evaluate the model
print("Coefficients:", model.coef_)
print("Intercept:", model.intercept_)
print("Mean Squared Error:", mean_squared_error(y_test, y_pred))
print("R² Score:", r2_score(y_test, y_pred))

# Step 7: Predict price for a new house
# New house: 2000 sqft, 3 bedrooms, 8 years old
new_house = np.array([[2000, 3, 8]])
predicted_price = model.predict(new_house)
print("Predicted Price for new house: $", round(predicted_price[0], 2), "K")
