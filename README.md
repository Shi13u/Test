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

## 🧪 Evaluation Metrics

Given the class imbalance, standard accuracy is misleading. Metrics used:

- **ROC AUC Score**
- **Precision-Recall Curve**
- **Confusion Matrix**
- **Classification Report (Precision, Recall, F1-score)**

> Area Under the Precision-Recall Curve (AUPRC) is especially emphasized due to the dataset’s extreme imbalance.

---

## 📉 Results

The best model achieved:

- **ROC AUC Score:** **0.9837**
- High recall on fraudulent transactions
- Good balance of precision and recall
- Strong model robustness on imbalanced data

> Final model performance reflects effective preprocessing, oversampling, and model selection (likely XGBoost or Random Forest with tuned parameters).

---

## 🔁 Cross-Validation

- `StratifiedKFold` and `RepeatedKFold` to ensure balanced class representation.
- `GridSearchCV` for hyperparameter tuning.

---

## 🧪 How to Run the Code

1. Clone the repository or upload the notebook to **Google Colab**
2. Ensure all libraries are installed:
   ```bash
   pip install -r requirements.txt
