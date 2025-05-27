# logistic_regression.py

import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix

# Step 1: Create dataset (Exam Score 1, Exam Score 2)
X = np.array([
    [34, 78], [30, 43], [35, 72], [60, 86], [79, 75],
    [45, 56], [61, 96], [75, 46], [76, 87], [84, 43],
    [54, 60], [46, 70], [80, 90], [85, 76], [33, 50]
])

# Step 2: Labels (1 = Admitted, 0 = Not Admitted)
y = np.array([0, 0, 0, 1, 1, 0, 1, 0, 1, 1, 0, 0, 1, 1, 0])

# Step 3: Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=0
)

# Step 4: Train the Logistic Regression model
model = LogisticRegression()
model.fit(X_train, y_train)

# Step 5: Predictions and evaluation
y_pred = model.predict(X_test)

print("Confusion Matrix:")
print(confusion_matrix(y_test, y_pred))
print("\nClassification Report:")
print(classification_report(y_test, y_pred))

# Step 6: Plotting decision boundary
x_min, x_max = X[:, 0].min() - 5, X[:, 0].max() + 5
y_min, y_max = X[:, 1].min() - 5, X[:, 1].max() + 5
xx, yy = np.meshgrid(np.linspace(x_min, x_max, 100),
                     np.linspace(y_min, y_max, 100))
Z = model.predict(np.c_[xx.ravel(), yy.ravel()])
Z = Z.reshape(xx.shape)

plt.figure(figsize=(8, 6))
plt.contourf(xx, yy, Z, alpha=0.3, cmap=plt.cm.coolwarm)
plt.scatter(X[:, 0], X[:, 1], c=y, cmap=plt.cm.coolwarm, edgecolors='k')
plt.xlabel("Exam Score 1")
plt.ylabel("Exam Score 2")
plt.title("Logistic Regression Decision Boundary")
plt.grid(True)
plt.show()
