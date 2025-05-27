<div align="center">

# 🧠 Supervised Machine Learning

<p><strong>Supervised Machine Learning</strong> is a type of machine learning where the model is trained using <strong>labeled data</strong>. The algorithm learns from input-output pairs to make predictions on unseen data.</p>

<img src="https://via.placeholder.com/600x300.png?text=Supervised+Learning+Diagram" alt="Supervised Learning Example" width="600"/>

---

## 📦 How It Works

<p>The model receives a dataset with features (X) and known outcomes (Y). The objective is to learn a mapping function <code>f: X → Y</code>.</p>

<h3>Example:</h3>

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

## 🧩 Types of Supervised Learning

### 1. Regression  
<ul>
  <li>Used when output is <strong>continuous</strong>.</li>
  <li>Examples: Predicting temperature, house prices</li>
  <li>Algorithms: Linear Regression, Ridge Regression</li>
</ul>

### 2. Classification  
<ul>
  <li>Used when output is <strong>categorical</strong>.</li>
  <li>Examples: Spam detection, disease diagnosis</li>
  <li>Algorithms: Logistic Regression, Decision Tree, SVM, KNN</li>
</ul>

<img src="https://via.placeholder.com/600x300.png?text=Classification+vs+Regression" alt="Classification vs Regression" width="600"/>

---

## 🔁 Common Supervised Algorithms

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
    <tr><td>Decision Trees</td><td>Easy interpretation</td></tr>
    <tr><td>Random Forest</td><td>Ensemble method</td></tr>
    <tr><td>SVM</td><td>High-dimensional data</td></tr>
    <tr><td>KNN</td><td>Instance-based learning</td></tr>
    <tr><td>Neural Networks</td><td>Deep learning tasks</td></tr>
  </tbody>
</table>

---

## 🧪 Evaluation Metrics

### For Classification:
<ul>
  <li><strong>Accuracy</strong> – Correct predictions / Total</li>
  <li><strong>Precision</strong> – TP / (TP + FP)</li>
  <li><strong>Recall</strong> – TP / (TP + FN)</li>
  <li><strong>F1 Score</strong> – Harmonic mean of Precision and Recall</li>
  <li><strong>Confusion Matrix</strong> – TP, FP, FN, TN table</li>
</ul>

### For Regression:
<ul>
  <li>Mean Absolute Error (MAE)</li>
  <li>Mean Squared Error (MSE)</li>
  <li>Root Mean Squared Error (RMSE)</li>
  <li>R² Score</li>
</ul>

---

## 🔧 Workflow

<ol>
  <li>Define the problem</li>
  <li>Collect and clean data</li>
  <li>Preprocess features</li>
  <li>Split data</li>
  <li>Train the model</li>
  <li>Evaluate results</li>
  <li>Tune hyperparameters</li>
  <li>Deploy solution</li>
</ol>

---

## 📌 Applications

<ul>
  <li>📧 Spam Email Detection</li>
  <li>💳 Credit Card Fraud Detection</li>
  <li>🏥 Disease Diagnosis</li>
  <li>🛒 Customer Churn Prediction</li>
  <li>🔍 Search Engine Ranking</li>
  <li>🎓 Student Performance Prediction</li>
</ul>

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
    <tr><td>Goal</td><td>Predict outcome</td><td>Find structure</td></tr>
    <tr><td>Examples</td><td>Linear Regression, SVM</td><td>K-Means, PCA</td></tr>
  </tbody>
</table>

<img src="https://via.placeholder.com/600x300.png?text=Supervised+vs+Unsupervised" alt="Supervised vs Unsupervised" width="600"/>

</div>
