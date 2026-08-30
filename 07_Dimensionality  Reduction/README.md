# Experiment 7 — Dimensionality Reduction and Model Evaluation

## Overview

This experiment investigates the effect of Principal Component Analysis (PCA) on the classification performance of ten machine learning models. Each model is trained and evaluated under two conditions: using the full standardized feature set (No-PCA) and using a PCA-reduced feature set (With-PCA). Hyperparameter tuning and 5-fold cross-validation are performed for all models under both settings.

## Objective

To study how dimensionality reduction through PCA affects the accuracy, F1-score, precision, recall, and ROC-AUC of different classifier types, and to determine which models benefit from PCA and which do not.

## Dataset

**Wisconsin Diagnostic Breast Cancer (WDBC) Dataset**

- **Source:** scikit-learn (`sklearn.datasets.load_breast_cancer`)
- **Samples:** 569
- **Features:** 30 continuous numerical attributes (computed from cell nuclei measurements)
- **Target Classes:** 2 — Malignant (0) and Benign (1)
- **Missing Values:** None

## Models

1. Support Vector Machine (SVM)
2. Naïve Bayes
3. k-Nearest Neighbors (KNN)
4. Logistic Regression
5. Decision Tree
6. Random Forest
7. AdaBoost
8. Gradient Boosting
9. XGBoost
10. Stacking (SVM, NB, KNN, DT, RF as base learners; Logistic Regression as meta-learner)

## PCA

- **Target:** Retain 95% of the total variance
- **Approach:** PCA is fitted only on the training set and applied to both training and testing sets
- **Comparison:** Every model is evaluated on both the original standardized features (No-PCA) and the PCA-reduced features (With-PCA)

## Methodology

1. **Preprocessing:** Drop non-informative columns, encode the target, and apply stratified 80-20 train-test split.
2. **Standardization:** Fit `StandardScaler` on the training data only.
3. **PCA:** Apply PCA with 95% variance retention to the standardized training features.
4. **Hyperparameter Tuning:** GridSearchCV with 5-fold stratified cross-validation for each model.
5. **5-Fold Cross-Validation:** Record fold-wise and average accuracy for both No-PCA and With-PCA.
6. **Evaluation:** Accuracy, Precision, Recall, F1-score, ROC-AUC, confusion matrices, ROC curves, and Precision-Recall curves.

## Project Structure

```
07_Dimensionality  Reduction/
│
├── Experiment_7_Dimensionality_Reduction.ipynb   # Complete experiment notebook
├── Experiment_7_ML.tex                           # LaTeX documentation source
├── Academics_SSN (2).pdf                         # Assignment question PDF
├── README.md                                     # This file
├── requirements.txt                              # Python dependencies
│
└── images/                                       # All generated figures
    ├── class_distribution.png
    ├── correlation_heatmap.png
    ├── pca_scree_plot.png
    ├── confusion_matrices_nopca.png
    ├── confusion_matrices_pca.png
    ├── roc_curves_comparison.png
    ├── precision_recall_curves.png
    ├── model_comparison_barplot.png
    ├── cv_stability_comparison.png
    └── pca_impact_heatmap.png
```

## How to Run

1. Install the required packages:

```bash
pip install -r requirements.txt
```

2. Open the notebook:

```bash
jupyter notebook Experiment_7_Dimensionality_Reduction.ipynb
```

3. Run all cells in order. The notebook generates all figures, tables, and results automatically.
