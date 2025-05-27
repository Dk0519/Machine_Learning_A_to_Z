<div align="center">

# 🧠 Supervised Machine Learning

**Supervised Machine Learning** is a type of machine learning where models learn from **labeled data**. The algorithm maps inputs to known outputs and improves its accuracy over time.

<img src="https://via.placeholder.com/600x300.png?text=Supervised+Learning+Workflow" width="600" alt="Supervised Learning Diagram"/>

---

## 📦 How It Works

The model receives a dataset with input features (`X`) and output labels (`Y`). It learns a mapping function `f: X → Y`.

### Example:

<table>
  <thead>
    <tr>
      <th>Features (X)</th>
      <th>Output (Y)</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>Hours studied = 5</td>
      <td>Score = 80</td>
    </tr>
    <tr>
      <td>Hours studied = 2</td>
      <td>Score = 45</td>
    </tr>
    <tr>
      <td>Hours studied = 9</td>
      <td>Score = 95</td>
    </tr>
  </tbody>
</table>

---

## 🍀 Types of Supervised Learning

### ✅ 1. Regression  
Used when output is **continuous**.

- Example: Predicting temperature, house prices  
- Algorithms: Linear Regression, Ridge Regression

### 🟨 2. Classification  
Used when output is **categorical**.

- Example: Spam detection, disease diagnosis  
- Algorithms: Logistic Regression, Decision Tree, SVM, KNN

<img src="https://via.placeholder.com/600x300.png?text=Classification+vs+Regression" width="600" alt="Classification vs Regression"/>

---

## 📘 Common Supervised Algorithms

<table>
  <thead>
    <tr>
      <th>Algorithm</th>
      <th>Use Case</th>
    </tr>
  </thead>
  <tbody>
    <tr><td>Linear Regression</td><td>Predicting numerical values</td></tr>
    <tr><td>Logistic Regression</td><td>Binary classification</td></tr>
    <tr><td>Decision Trees</td><td>Easy to interpret decisions</td></tr>
    <tr><td>Random Forest</td><td>Ensemble method with multiple trees</td></tr>
    <tr><td>SVM</td><td>Classifying complex spaces</td></tr>
    <tr><td>KNN</td><td>Instance-based learning</td></tr>
    <tr><td>Neural Networks</td><td>Advanced deep learning tasks</td></tr>
  </tbody>
</table>

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

## 🔧 Supervised Learning Workflow

1. Define the problem  
2. Collect and prepare data  
3. Preprocess the features  
4. Split data (Train/Test)  
5. Train the model  
6. Evaluate performance  
7. Tune hyperparameters  
8. Deploy the model

---

## 📌 Real-World Applications

- 📧 Spam Email Detection  
- 💳 Fraud Detection  
- 🏥 Medical Diagnosis  
- 🛒 Customer Churn Prediction  
- 🔍 Search Engine Ranking  
- 🎓 Student Performance Prediction

---

## 🆚 Supervised vs Unsupervised Learning

<table>
  <thead>
    <tr>
      <th>Aspect</th>
      <th>Supervised</th>
      <th>Unsupervised</th>
    </tr>
  </thead>
  <tbody>
    <tr><td>Data</td><td>Labeled</td><td>Unlabeled</td></tr>
    <tr><td>Goal</td><td>Prediction</td><td>Pattern discovery</td></tr>
    <tr><td>Examples</td><td>Linear Regression, SVM</td><td>K-Means, PCA</td></tr>
  </tbody>
</table>

<img src="https://via.placeholder.com/600x300.png?text=Supervised+vs+Unsupervised" width="600" alt="Comparison image"/>

</div>
