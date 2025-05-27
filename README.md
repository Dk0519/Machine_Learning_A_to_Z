# 📚 Machine Learning Theory Notes

## 📖 Table of Contents

1. [Introduction to Machine Learning](#introduction-to-machine-learning)
2. [Types of Machine Learning](#types-of-machine-learning)
   - [Supervised Learning](#supervised-learning)
   - [Unsupervised Learning](#unsupervised-learning)
   - [Reinforcement Learning](#reinforcement-learning)
3. [Key Concepts in Machine Learning](#key-concepts-in-machine-learning)
   - [Overfitting and Underfitting](#overfitting-and-underfitting)
   - [Bias-Variance Tradeoff](#bias-variance-tradeoff)
   - [Evaluation Metrics](#evaluation-metrics)
4. [Model Selection and Validation](#model-selection-and-validation)
   - [Cross-Validation](#cross-validation)
   - [Hyperparameter Tuning](#hyperparameter-tuning)
5. [Feature Engineering](#feature-engineering)
   - [Feature Selection](#feature-selection)
   - [Feature Extraction](#feature-extraction)
6. [Dimensionality Reduction](#dimensionality-reduction)
7. [Ensemble Methods](#ensemble-methods)
8. [Common Machine Learning Algorithms](#common-machine-learning-algorithms)
9. [Resources for Further Reading](#resources-for-further-reading)

---

## Introduction to Machine Learning

**Machine Learning (ML)** is a subset of artificial intelligence that focuses on building systems that learn from data to improve their performance over time without being explicitly programmed.

---

## Types of Machine Learning

### Supervised Learning

In supervised learning, the model is trained on a labeled dataset, which means that each training example is paired with an output label.

**Common algorithms:**
- Linear Regression
- Logistic Regression
- Decision Trees
- Support Vector Machines (SVM)
- K-Nearest Neighbors (KNN)

### Unsupervised Learning

Unsupervised learning deals with unlabeled data. The model tries to learn the underlying structure of the data.

**Common algorithms:**
- K-Means Clustering
- Hierarchical Clustering
- Principal Component Analysis (PCA)
- Independent Component Analysis (ICA)

### Reinforcement Learning

Reinforcement learning is about taking suitable actions to maximize rewards in a particular situation. It is employed by various software and machines to find the best possible behavior or path it should take in a specific situation.

**Key concepts:**
- Agent, Environment, Action, Reward
- Policy, Value Function, Q-Function

---

## Key Concepts in Machine Learning

### Overfitting and Underfitting

- **Overfitting**: The model learns the training data too well, including its noise and outliers, which negatively impacts its performance on new data.
- **Underfitting**: The model is too simple to capture the underlying structure of the data.

### Bias-Variance Tradeoff

- **Bias**: Error due to overly simplistic assumptions in the learning algorithm.
- **Variance**: Error due to too much complexity in the learning algorithm.

The tradeoff is the balance between the error introduced by the bias and the variance.

### Evaluation Metrics

- **Classification Metrics**: Accuracy, Precision, Recall, F1-Score, ROC-AUC
- **Regression Metrics**: Mean Squared Error (MSE), Mean Absolute Error (MAE), R-squared

---

## Model Selection and Validation

### Cross-Validation

A technique for assessing how the results of a statistical analysis will generalize to an independent dataset. Common methods include k-fold cross-validation and leave-one-out cross-validation.

### Hyperparameter Tuning

The process of choosing a set of optimal hyperparameters for a learning algorithm. Techniques include:
- Grid Search
- Random Search
- Bayesian Optimization

---

## Feature Engineering

### Feature Selection

The process of selecting a subset of relevant features for model construction.

**Methods:**
- Filter Methods
- Wrapper Methods
- Embedded Methods

### Feature Extraction

Transforming the input data into a set of features. Techniques include:
- Principal Component Analysis (PCA)
- Linear Discriminant Analysis (LDA)

---

## Dimensionality Reduction

Reducing the number of random variables under consideration by obtaining a set of principal variables.

**Techniques:**
- Principal Component Analysis (PCA)
- t-Distributed Stochastic Neighbor Embedding (t-SNE)
- Autoencoders

---

## Ensemble Methods

Combining predictions from multiple machine learning algorithms to make more accurate predictions than any individual model.

**Types:**
- Bagging (e.g., Random Forest)
- Boosting (e.g., AdaBoost, Gradient Boosting)
- Stacking

---

## Common Machine Learning Algorithms

- **Linear Regression**: Predicts a continuous dependent variable based on the value of independent variables.
- **Logistic Regression**: Used for binary classification problems.
- **Decision Trees**: A flowchart-like structure for decision making.
- **Support Vector Machines (SVM)**: Finds the hyperplane that best divides a dataset into classes.
- **K-Nearest Neighbors (KNN)**: Classifies data based on the closest training examples in the feature space.
- **Naive Bayes**: Based on applying Bayes' theorem with strong independence assumptions.
- **K-Means Clustering**: Partitions data into K distinct clusters based on distance to the centroid of a cluster.
- **Principal Component Analysis (PCA)**: Reduces the dimensionality of data while preserving as much variability as possible.

---

## Resources for Further Reading

- [Understanding Machine Learning: From Theory to Algorithms](https://www.cs.huji.ac.il/~shais/UnderstandingMachineLearning/understanding-machine-learning-theory-algorithms.pdf)
- [CS229 Lecture Notes by Andrew Ng](https://cs229.stanford.edu/main_notes.pdf)
- [Dive into Deep Learning](https://d2l.ai/)
- [Machine Learning Notes - GitHub Repository](https://github.com/federicobrancasi/MachineLearningNotes)
- [Complete Machine Learning Repository by Nyandwi](https://github.com/Nyandwi/machine_learning_complete)

---

*End of Machine Learning Theory Notes*
