# 🧠 Multiple Linear Regression — Explained with Example

## 📘 What is Multiple Linear Regression?

**Multiple Linear Regression (MLR)** is a supervised learning algorithm that models the relationship between a **dependent variable** and **multiple independent variables**. It fits a linear equation to the data to predict outcomes based on several input features.

---

## 📐 Mathematical Representation

\[
y = \beta_0 + \beta_1x_1 + \beta_2x_2 + \cdots + \beta_nx_n + \varepsilon
\]

Where:
- \( y \) = target variable (house price)
- \( x_1, x_2, ..., x_n \) = input features (size, bedrooms, age)
- \( \beta_0 \) = intercept
- \( \beta_i \) = coefficient for feature \( x_i \)
- \( \varepsilon \) = error term

---

## 📊 Dataset Example

| Size (sqft) | Bedrooms | Age (years) | Price ($1000s) |
|-------------|----------|-------------|----------------|
| 1500        | 3        | 10          | 250            |
| 1800        | 4        | 15          | 300            |
| 2400        | 3        | 5           | 375            |
| 3000        | 5        | 20          | 480            |
| 1000        | 2        | 8           | 200            |
| 2000        | 4        | 10          | 320            |

---

## 🛠️ How the Model Works

- **Input (X)** = `[Size, Bedrooms, Age]`
- **Output (y)** = `Price in $1000s`

The model uses a dataset to **learn the coefficients** (slopes) for each feature, such that the predicted output matches the actual output as closely as possible.

### Example Learned Equation:

\[
\text{Price} = 0.1 \cdot \text{Size} + 18 \cdot \text{Bedrooms} - 1.2 \cdot \text{Age} + 50
\]

- 🏠 +1 sq.ft. adds $100
- 🛏️ +1 bedroom adds $18,000
- 🧓 +1 year old reduces $1,200

---

## 🔮 Predicting a New House Price

For a house with:
- Size = 2000 sqft
- Bedrooms = 3
- Age = 8 years

\[
\text{Price} = 0.1 \cdot 2000 + 18 \cdot 3 - 1.2 \cdot 8 + 50 = 294.4K
\]

---

## 📈 3D Visualization (Size vs Age vs Price)

- **X-axis**: Size (sqft)  
- **Y-axis**: Age (years)  
- **Z-axis**: Price ($1000s)

The 3D plot includes:
- 🔵 **Blue dots** = actual house data
- 🟩 **Regression plane** = predictions by the model

This helps visually understand how price depends on **both size and age**.

![3D Regression Surface](https://via.placeholder.com/600x400.png?text=3D+Regression+Plot)

---

## 🧪 Model Evaluation

| Metric           | Description                                      |
|------------------|--------------------------------------------------|
| **MSE**          | Measures average squared error                   |
| **R² Score**     | Indicates how well model explains variability    |

High R² and low MSE = better fit.

---

## ✅ Summary

| Element               | Description                                      |
|------------------------|--------------------------------------------------|
| Model Type             | Multiple Linear Regression                       |
| Features Used          | Size, Bedrooms, Age                              |
| Goal                   | Predict house price                              |
| Tool Used              | scikit-learn                                     |
| Visualization          | 3D plot of Size vs Age vs Price                  |
| Strengths              | Simple, interpretable, works for linear data     |
| Limitations            | Poor with non-linear or collinear features       |

---

## 🧰 Python Libraries Used

```python
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

