# 💳 Credit Card Fraud Detection

## 🔍 Overview
This project focuses on detecting fraudulent credit card transactions using machine learning techniques. With fraud accounting for a mere 0.172% of transactions, the challenge lies in identifying rare positive cases in a highly imbalanced dataset. The goal is to maximize detection of fraudulent transactions while minimizing false positives.

---

## 📊 Dataset Description

- **Source:** [Kaggle - Credit Card Fraud Detection](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud)
- **Records:** 284,807 transactions
- **Fraud Cases:** 492 (0.172%)
- **Features:** 30 (28 PCA-transformed features + `Time` + `Amount`)
- **Target:** `Class` (1 = fraud, 0 = non-fraud)

> The dataset has been anonymized using PCA due to confidentiality. `Time` indicates the time elapsed from the first transaction, and `Amount` is the transaction value.

---

## 📦 Libraries Used

- `NumPy`, `Pandas`
- `Matplotlib`, `Seaborn` – for EDA and visualization
- `Scikit-learn` – for modeling, evaluation, and cross-validation
- `XGBoost` – for advanced tree-based modeling
- `imbalanced-learn` – for resampling strategies (handling class imbalance)

---

## 📈 Exploratory Data Analysis (EDA)

- Analyzed class distribution and found extreme imbalance.
- Visualized amount distributions for fraudulent vs. legitimate transactions.
- Identified patterns using correlation heatmaps and time-based behavior.

---

## 🔧 Preprocessing

- Handled class imbalance using:
  - `SMOTE` (Synthetic Minority Oversampling Technique)
  - `ADASYN`
  - `RandomOverSampler`
- Feature scaling applied where needed.
- Created stratified train-test splits for fair model evaluation.

---

## 🛠️ Feature Engineering

- Transformed the `Time` feature to represent **elapsed hours** from the first transaction.

---

## 🤖 Machine Learning Models Used

| Model                    | Notes                           |
|--------------------------|----------------------------------|
| Logistic Regression (CV) | Baseline linear model with CV   |
| K-Nearest Neighbors      | Distance-based model             |
| Decision Tree            | Simple non-linear classifier     |
| Random Forest            | Ensemble of decision trees       |
| Support Vector Machine   | Effective for high-dimensional data |
| XGBoost                  | Optimized gradient boosting      |

---

## 🧪 Evaluation Strategy

Each model was evaluated in **three setups**:

1. Train/test split evaluation
2. Cross-validation with **Repeated K-Fold**
3. Cross-validation with **Stratified K-Fold**

Metrics stored for all settings:
- ROC AUC
- Accuracy
- Precision, Recall, F1-Score
- Confusion Matrix

---

## ⚖️ Oversampling Techniques

To address class imbalance, the following methods were applied:
- `SMOTE`
- `ADASYN`
- `RandomOverSampler`

Each model was re-trained and evaluated on the oversampled training set. All results were stored and compared.

---

## 🔍 Model Selection & Tuning

- Based on ROC AUC score across all evaluations, **XGBoost** was the top performer.
- Hyperparameters were tuned using:
  - ✅ `RandomizedSearchCV` (efficient exploration)
  - ✅ Scoring metric: `roc_auc`

---

## 📉 Final Results

The best-performing model, **XGBoost with RandomizedSearchCV**, was evaluated on the original (non-oversampled) test set.

**Final Results:**
- ✅ **Accuracy:** 0.9995
- ✅ **ROC AUC Score:** 0.9958
- ✅ **Classification Report (Class 1 - Fraud):**
  - **Precision:** 0.93  
  - **Recall:** 0.79  
  - **F1-Score:** 0.85  
- ✅ **Macro Avg F1-Score:** 0.93  
- ✅ **Weighted Avg F1-Score:** 1.00

> These results show a high ability to detect fraud while maintaining minimal false positives on an imbalanced dataset.


## 🧪 How to Run the Code

1. Clone the repository or upload the notebook to **Google Colab**
2. Ensure all libraries are installed:
   ```bash
   pip install -r requirements.txt
