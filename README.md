# 📚 Machine Learning Theory Notes

---

## 📖 Table of Contents

1. [📌 What is Machine Learning](#-what-is-machine-learning)
2. [🧠 Types of Machine Learning](#-types-of-machine-learning)
3. [⚙️ Key Concepts](#️-key-concepts)
4. [🧮 Mathematics for Machine Learning](#-mathematics-for-machine-learning)
5. [📊 Model Validation](#-model-validation)
6. [🧱 Feature Engineering](#-feature-engineering)
7. [🔻 Dimensionality Reduction](#-dimensionality-reduction)
8. [🧠 Ensemble Learning](#-ensemble-learning)
9. [📚 Further Reading](#-further-reading)

---

## 📌 What is Machine Learning

> “Machine Learning is the field of study that gives computers the ability to learn without being explicitly programmed.”  
> — Arthur Samuel

ML allows systems to learn patterns from data, improving their performance over time.

![ML process](https://atriainnovation.com/uploads/2023/11/portada-9-900x743-c-center.jpg)

---

## 🧠 Types of Machine Learning

### 🔷 Supervised Learning
Trained on labeled data (input → output).

- **Algorithms**: Linear Regression, SVM, KNN, Decision Trees  
- **Use Cases**: Spam detection, fraud classification, house price prediction

### 🔶 Unsupervised Learning
Learns patterns from unlabeled data.

- **Algorithms**: K-Means, PCA, Hierarchical Clustering  
- **Use Cases**: Customer segmentation, anomaly detection

### 🔁 Reinforcement Learning
Learns actions based on rewards and penalties.

- **Key terms**: Agent, Environment, Reward, Policy  
- **Use Cases**: Robotics, games, self-driving cars

![Types of ML](https://datasciencedojo.com/wp-content/uploads/ml-ds-algos.jpg.webp)

---

## ⚙️ Key Concepts

### 🎯 Overfitting vs Underfitting

| Concept        | Description                                    |
|----------------|------------------------------------------------|
| Overfitting     | Model is too complex; memorizes training data |
| Underfitting    | Model is too simple; misses patterns          |

![Overfit vs Underfit](https://www.mathworks.com/discovery/overfitting/_jcr_content/mainParsys/image.adapt.full.medium.svg/1746469504474.svg)

---

### 🎯 Bias-Variance Tradeoff

- **Bias**: Error from incorrect assumptions  
- **Variance**: Error from model's sensitivity to small changes in the training set

We aim for **low bias & low variance** — the sweet spot of model performance.

![Bias Variance](https://miro.medium.com/v2/format:webp/1*atFRtCnfWNUJMlPhie8mfA.png)

---

### 🧪 Evaluation Metrics

#### Classification:
- Accuracy
- Precision / Recall / F1-Score
- Confusion Matrix
- ROC-AUC

#### Regression:
- MAE / MSE / RMSE
- R² Score

---

## 🧮 Mathematics for Machine Learning

Understanding the math behind ML helps you debug, tune, and trust your models.

### 📐 Linear Algebra
- Vectors, matrices, dot product
- Matrix multiplication, eigenvalues, eigendecomposition  
🛠 Used in: Data representation, PCA, neural networks

### 🔁 Calculus
- Derivatives, gradients
- Chain rule, partial derivatives  
🛠 Used in: Gradient descent, backpropagation

### 🎲 Probability & Statistics
- Distributions, conditional probability
- Bayes’ theorem, expectation, variance  
🛠 Used in: Naive Bayes, probabilistic models

![Math for ML](https://miro.medium.com/v2/resize:fit:720/format:webp/1*eI6ZzKZb-MFpHHU9BeyyqQ.png)

---

## 📊 Model Validation

### 🔁 Cross-Validation
Split data multiple times to validate generalization.

- **k-fold cross-validation**
- **Leave-one-out cross-validation**

### 🔧 Hyperparameter Tuning
Optimize model configuration.

- Grid Search
- Random Search
- Bayesian Optimization

---

## 🧱 Feature Engineering

### ✅ Feature Selection
Pick relevant features.
- Filter, wrapper, embedded methods

### 🧪 Feature Extraction
Transform raw data into meaningful inputs.
- PCA, LDA, t-SNE, autoencoders

---

## 🔻 Dimensionality Reduction

Reduce input features while retaining important info.

**Popular methods**:
- PCA
- t-SNE
- UMAP
- Autoencoders (in deep learning)

![Dimensionality Reduction](https://www.sc-best-practices.org/_images/dimensionality_reduction.jpeg)

---

## 🧠 Ensemble Learning

Combine multiple models to improve performance.

### Bagging:
- Build multiple models in parallel  
- Example: **Random Forest**

### Boosting:
- Models correct each other in sequence  
- Example: **XGBoost, AdaBoost, LightGBM**

### Stacking:
- Combine outputs of many models via meta-model

---

## 📚 Further Reading

- 📘 [CS229 - Andrew Ng's Stanford ML Notes](https://cs229.stanford.edu/main_notes.pdf)  
- 📗 [Understanding Machine Learning - Shai Shalev-Shwartz](https://www.cs.huji.ac.il/~shais/UnderstandingMachineLearning/)  
- 📘 [Dive into Deep Learning (Interactive)](https://d2l.ai/)  
- 📕 [ML Cheatsheets by Afshine & Shervine Amidi](https://stanford.edu/~shervine/teaching/cs-229/)  
- 📘 [Google Machine Learning Crash Course](https://developers.google.com/machine-learning/crash-course)

---

> 🧠 _Keep this as a master reference. You can add individual algorithms or project summaries as separate markdown files._
