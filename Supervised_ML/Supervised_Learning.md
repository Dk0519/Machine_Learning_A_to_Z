# 🧠 Supervised Machine Learning

**Supervised Machine Learning** is a type of machine learning where models are trained using **labeled datasets**. The algorithm learns the relationship between input features (X) and output labels (Y) to make predictions on unseen data.

---

## 📦 How It Works

In supervised learning, the model receives data with both inputs and outputs. The aim is to learn a function `f: X → Y` that can generalize well to new, unseen inputs.

### Example:

| Features (X)         | Output (Y)   |
|----------------------|--------------|
| Hours studied = 5    | Score = 80   |
| Hours studied = 2    | Score = 45   |
| Hours studied = 9    | Score = 95   |

---

## 🍀 Types of Supervised Learning

### ✅ 1. Regression

- Used when output is **continuous**  
- Examples: Predicting temperature, house prices  
- Algorithms: Linear Regression, Ridge Regression

### 🟨 2. Classification

- Used when output is **categorical**  
- Examples: Spam detection, disease diagnosis  
- Algorithms: Logistic Regression, Decision Tree, SVM, KNN

![Classification vs Regression](https://via.placeholder.com/600x300.png?text=Classification+vs+Regression)

---

## 📘 Common Supervised Algorithms

| Algorithm            | Use Case                        |
|----------------------|----------------------------------|
| Linear Regression    | Predicting numerical values      |
| Logistic Regression  | Binary classification            |
| Decision Trees       | Easy to interpret decisions      |
| Random Forest        | Ensemble method                  |
| Support Vector Machine (SVM) | Complex classification |
| K-Nearest Neighbors (KNN) | Instance-based learning    |
| Neural Networks      | Deep learning tasks              |

---

## 🧪 Evaluation Metrics

### For Classification:

- Accuracy  
- Precision = TP / (TP + FP)  
- Recall = TP / (TP + FN)  
- F1 Score = Harmonic mean of Precision and Recall  
- Confusion Matrix

### For Regression:

- Mean Absolute Error (MAE)  
- Mean Squared Error (MSE)  
- Root Mean Squared Error (RMSE)  
- R² Score

---

## 🔧 Workflow of Supervised Learning

1. Define the problem  
2. Collect and clean the data  
3. Preprocess the features  
4. Split into training and test sets  
5. Train the model  
6. Evaluate performance  
7. Tune hyperparameters  
8. Deploy the model

---

## 📌 Real-World Applications

- 📧 Email Spam Detection  
- 💳 Credit Card Fraud Detection  
- 🏥 Disease Diagnosis  
- 🛒 Customer Churn Prediction  
- 🔍 Search Engine Ranking  
- 🎓 Student Performance Prediction

---

## 🆚 Supervised vs Unsupervised Learning

| Aspect              | Supervised Learning         | Unsupervised Learning        |
|---------------------|-----------------------------|------------------------------|
| Data                | Labeled                     | Unlabeled                    |
| Goal                | Predict outcomes            | Discover patterns            |
| Example Algorithms  | Linear Regression, SVM      | K-Means, PCA                 |

![Supervised vs Unsupervised](https://via.placeholder.com/600x300.png?text=Supervised+vs+Unsupervised)
