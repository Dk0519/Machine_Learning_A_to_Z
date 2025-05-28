# 📊 Statistics for Machine Learning

> *“Statistics is the grammar of science.”* — Karl Pearson

---

## ✨ What **is** Statistics?

Statistics is the discipline that **collects, organizes, summarizes, analyzes, and draws conclusions** from data.  In the context of machine‑learning (ML), statistics provides the theoretical backbone for **building models**, **quantifying uncertainty**, and **validating results**.

* **Population**: the complete set of items/events we care about.
* **Sample**: a (usually small) subset of the population used to make inferences.
* **Parameter** (Greek letters, e.g. μ, σ): a descriptive measure of the population (unknown!).
* **Statistic** (Latin letters, e.g. $\bar{x}$, *s*): a descriptive measure computed from a sample.

---

## 🗂️ Types of Data

| Category                      | Sub‑types      | Description       | Examples                      | Typical ML Encoding |
| ----------------------------- | -------------- | ----------------- | ----------------------------- | ------------------- |
| **Qualitative (Categorical)** | **Nominal**    | Unordered labels  | browser {Chrome, Safari}      | One‑hot             |
|                               | **Ordinal**    | Ordered labels    | Likert scale {Poor→Excellent} | Ordinal/Target      |
| **Quantitative (Numeric)**    | **Discrete**   | Integers / counts | clicks per session            | As‑is or log        |
|                               | **Continuous** | Any real number   | temperature °C                | Normalization       |

> 💡 **Why it matters:** Choosing the wrong encoding can break distance‑based models (e.g., k‑NN treats category codes as numeric distance).

<details>
<summary><strong>More on Encoding 🛠️</strong></summary>

* **Label Encoding** – preserves order → good for ordinal.
* **One‑Hot / Dummy** – breaks a column into *k* binary flags → default for nominal.
* **Frequency Encoding** – map category to its empirical probability.
* **Target Encoding** – replace category with mean target → powerful but risk of leakage.

</details>

## 🌳 Two Main Branches

| Branch          | Goal                                              | Typical Questions                               |
| --------------- | ------------------------------------------------- | ----------------------------------------------- |
| **Descriptive** | Condense & visualize data you *have*              | *“What is the average click‑through‑rate?”*     |
| **Inferential** | Generalize beyond the data & quantify uncertainty | *“Will the new UI raise CTR across all users?”* |

Below, each branch is expandable. Click to dive in! ⤵️

---

<details>
<summary><strong>📈 Descriptive Statistics</strong></summary>

### 1. Measures of Central Tendency (MCT) 🧭

| Symbol      | Name                | Formula                                  | Derivation Sketch                                                            |        |    |
| ----------- | ------------------- | ---------------------------------------- | ---------------------------------------------------------------------------- | ------ | -- |
| $\bar{x}$   | **Mean**            | $\bar{x}=\dfrac{1}{n}\sum_{i=1}^{n}x_i$  | Minimize squared error $\sum (x_i-c)^2$ ⇒ set derivative to 0 ⇒ $c=\bar{x}$. |        |    |
| $\tilde{x}$ | **Median**          | Middle value (or average of two middles) | Minimizes absolute error (\sum                                               | x\_i-c | ). |
| *Mode*      | Most frequent value | N/A                                      | Useful for categorical features.                                             |        |    |

### 2. Measures of Dispersion (MD) 🎯

| Symbol | Name                     | Formula                                    | Interpretation                                                     |
| ------ | ------------------------ | ------------------------------------------ | ------------------------------------------------------------------ |
| $s^2$  | **Sample Variance**      | $s^2 = \dfrac{1}{n-1}\sum (x_i-\bar{x})^2$ | Average squared deviation; divisor *(n‑1)* is Bessel’s correction. |
| $s$    | **Std. Deviation**       | $s=\sqrt{s^2}$                             | Back to original units.                                            |
| IQR    | **Inter‑Quartile Range** | $Q_3-Q_1$                                  | Robust to outliers; great for box plots.                           |

> 💡 **ML tie‑in:** Feature scaling (z‑score) uses mean & std‑dev; robust scaling uses median & IQR.

### 3. Shape of the Distribution 🌀

* **Skewness** $\gamma_1 = \dfrac{\mu_3}{\sigma^3}$  – asymmetry.
* **Kurtosis** $\gamma_2 = \dfrac{\mu_4}{\sigma^4}-3$ – tail heaviness.

### 4. Visual Tools 🖼️

| Plot      | Best for                |
| --------- | ----------------------- |
| Histogram | Univariate distribution |
| Box‑plot  | Spread & outliers       |
| Pair‑plot | Multivariate overview   |
| Heat‑map  | Correlation matrix      |

### 🚀 Real‑World ML Examples

* **EDA before modeling**: spot skewness → apply log‑transform.
* **Data quality checks**: high std‑dev in sensor readings may flag malfunction.

</details>

---

<details>
<summary><strong>🎲 Inferential Statistics</strong></summary>

### 1. Estimation 🔍

**Goal:** Use a sample to estimate population parameters.

| Type              | Output                                     | Formula / Method                      | ML Context                                     |                                                 |
| ----------------- | ------------------------------------------ | ------------------------------------- | ---------------------------------------------- | ----------------------------------------------- |
| **Point**         | Single value $\hat{\theta}$                | MLE: maximize (L(\theta)=\prod f(x\_i | \theta)).                                      | Fit model weights by MLE (e.g., Logistic Reg.). |
| **Interval (CI)** | Range $[\hat{\theta}\pm z_{\alpha/2}\,SE]$ | $SE=\dfrac{s}{\sqrt{n}}$ for mean.    | Reporting ±1.96·SE around validation accuracy. |                                                 |

### 2. Hypothesis Testing ⚔️

| Concept        | Symbol        | Typical Steps                                |
| -------------- | ------------- | -------------------------------------------- |
| Null vs Alt    | $H_0, H_1$    | State claims                                 |
| Test‑statistic | *t, z, χ², F* | Compute from data                            |
| p‑value        | *p*           | Prob. of observing ≥ statistic if $H_0$ true |
| Decision       | α             | Reject if *p* < α                            |

> 🧠 **Key Idea:** Small *p* → data is incompatible with $H_0$; doesn’t *prove* ﻿$H_1$.

**Common Tests**

| Test     | Use‑case            | Assumptions      |
| -------- | ------------------- | ---------------- |
| *t‑test* | Mean diff (n<30)    | Normality        |
| *z‑test* | Mean diff (known σ) | Normality        |
| *χ²*     | Categorical assoc.  | Expected freq ≥5 |
| *ANOVA*  | ≥3 group means      | Homoscedasticity |

### 3. Resampling & the CLT 🌀

* **Bootstrap**: Empirically approximate sampling distribution → robust CIs.
* **Cross‑validation**: Estimate generalization error.

### 4. Bias‑Variance Trade‑off 🎯

Derive expected test MSE:
$E[(y-\hat f(x))^2] = \underbrace{\text{Bias}^2}_{(E\hat f - f)^2} + \underbrace{\text{Variance}}_{E[(\hat f-E\hat f)^2]} + \sigma^2$

* **High‑bias models**: underfit (e.g., linear on non‑linear data).
* **High‑variance models**: overfit (deep tree without pruning).

### 🚀 Real‑World ML Examples

* **A/B Testing**: Hypothesis test on conversion rate.
* **Early stopping**: Monitor CV error → balance variance.
* **Ensembles**: Bagging (Random Forest) reduces variance via bootstrap.

</details>

---

## 📚 Further Reading & Resources

1. *Pattern Recognition & Machine Learning* — C. Bishop, Ch. 1‑2.
2. *An Introduction to Statistical Learning* — J. James et al., Ch. 2‑5.
3. *Practical Statistics for Data Scientists* — B. Bruce.

---

> 🏁 **Next Steps:** Dive into probability distributions ➡️ Normal, Bernoulli, Poisson, …

---

© 2025 Prashant Yadav. Feel free to copy / share under CC‑BY‑SA 4.0.
