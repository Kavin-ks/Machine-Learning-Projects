# Machine Learning Algorithms Laboratory

## Overview
This repository contains the complete implementation, datasets, and documentation for six academic experiments completed as part of the Machine Learning Algorithms Laboratory course. Each experiment focuses on implementing core machine learning models, conducting systematic exploratory data analysis (EDA), performing feature engineering, optimizing hyperparameters, and evaluating performance.

## Experiments

| Experiment | Topic | Main Concepts |
|---|---|---|
| [01_Universal_EDA](./01_Universal_EDA/) | Universal Exploratory Data Analysis & Preprocessing | Automated summary statistics, null/duplicate inspections, correlation heatmaps, feature distribution plots, and baseline model evaluations across five benchmark datasets. |
| [02_Email_Spam_Classification](./02_Email_Spam_Classification/) | Email Spam/Ham Classification via Naïve Bayes & KNN | Supervised text classification using Gaussian, Multinomial, and Bernoulli Naïve Bayes models alongside K-Nearest Neighbors (KNN) with KDTree and BallTree spatial structures. |
| [03_Regression_Analysis](./03_Regression_Analysis/) | Regression Analysis using Linear & Regularized Models | Credit scoring prediction comparing Ordinary Least Squares (OLS) Linear Regression with Ridge ($L_2$), Lasso ($L_1$), and ElasticNet regression models. |
| [04_BInary classification](./04_BInary%20classification/) | Binary Classification via Logistic Regression & SVM | Email spam classification utilizing regularized Logistic Regression and Support Vector Machines (SVM) with Linear and Radial Basis Function (RBF) kernels. |
| [05_DecisionTree-RandomForest](./05_DecisionTree-RandomForest/) | Tumor Classification via Decision Tree & Random Forest | Comparative breast cancer diagnosis comparing a single decision tree classifier with a bootstrap-aggregated Random Forest ensemble. |
| [06_Ensembling-Models](./06_Ensembling-Models/) | Ensemble Learning: Bagging, Boosting, & Stacking | Advanced classification combining Bootstrap Aggregation (Bagging), sequential Boosting (AdaBoost and Gradient Boosting), and heterogeneous Stacked Ensembles. |

## Repository Structure

```
Machine Learning/
├── README.md
├── Manual (1).tex
├── 01_Universal_EDA/
│   ├── README.md
│   ├── requirements.txt
│   ├── universal_eda.ipynb
│   ├── Iris.ipynb
│   ├── Loan_amount_prediction.ipynb
│   ├── Diabetes.ipynb
│   ├── Handwritten_character_recognition.ipynb
│   ├── Classification_of_Email.ipynb
│   └── Datasets/
├── 02_Email_Spam_Classification/
│   ├── README.md
│   ├── requirements.txt
│   └── Naive Bayes/
│       ├── spambase_csv.csv
│       ├── Assignment_2_Report.tex
│       └── SpamBase-KNN/
│           └── Naive.ipynb
├── 03_Regression_Analysis/
│   ├── README.md
│   ├── requirements.txt
│   ├── Loan_amount_prediction-ridge.ipynb
│   ├── train.csv
│   └── test.csv
├── 04_BInary classification/
│   ├── README.md
│   ├── requirements.txt
│   ├── Experiment -4.ipynb
│   └── spambase.csv
├── 05_DecisionTree-RandomForest/
│   ├── README.md
│   ├── requirements.txt
│   ├── decisiontree_randomforest.ipynb
│   └── wdbc.data
└── 06_Ensembling-Models/
    ├── README.md
    ├── requirements.txt
    ├── Ensemble_models.ipynb
    └── Experiment_6_Report.tex
```

Each experiment directory contains:
- A local `README.md` with detailed experiment objectives, execution guides, and results.
- A local `requirements.txt` containing only the specific python packages imported for that experiment.
- Interactive Jupyter Notebooks (`.ipynb`) containing python implementations.
- Source datasets (`.csv`, `.data`) where applicable.
- Saved visualizations inside an `images/` directory.
- Academic LaTeX report sources (`.tex`) detailing the theoretical framework and findings.

## Experiments Overview

### Experiment 1 — Universal Exploratory Data Analysis
- **Objective**: Explore Python scientific libraries and implement an automated function (`perform_eda()`) to inspect and report dataset characteristics before training models.
- **Main techniques/algorithms**: Automatic metadata analysis, duplicate and missing value checks, correlation heatmaps, feature distribution plots.
- **Dataset/task**: Automated inspection applied across Iris (multi-class), Loan Prediction (regression), Diabetes (binary), MNIST (digits), and SMS Spam (text) datasets.
- **Link**: [Experiment 1 Folder](./01_Universal_EDA/)

### Experiment 2 — Email Spam/Ham Classification via Naïve Bayes & KNN
- **Objective**: Build and compare supervised binary classifiers for email spam detection, optimizing neighborhood sizes ($k$) and spatial trees.
- **Main techniques/algorithms**: Gaussian, Bernoulli, and Multinomial Naïve Bayes; KNN classification using KDTree and BallTree spatial structures; hyperparameter optimization via `GridSearchCV` and `RandomizedSearchCV`.
- **Dataset/task**: UCI SpamBase Dataset (4,210 cleaned records, 57 features).
- **Link**: [Experiment 2 Folder](./02_Email_Spam_Classification/)

### Experiment 3 — Regression Analysis using Linear & Regularized Models
- **Objective**: Develop regularized regression models to predict loan sanction amounts, comparing coefficient shrinkage paths and residual errors.
- **Main techniques/algorithms**: Ordinary Least Squares (OLS) Linear Regression, Ridge ($L_2$) Regression, Lasso ($L_1$) Regression, ElasticNet (hybrid) Regression.
- **Dataset/task**: Predict Loan Amount Dataset (29,660 cleaned records, 46 features after one-hot encoding).
- **Link**: [Experiment 3 Folder](./03_Regression_Analysis/)

### Experiment 4 — Binary Classification via Logistic Regression & SVM
- **Objective**: Implement and compare linear and kernel-based binary classifiers to detect email spam.
- **Main techniques/algorithms**: Logistic Regression (with $L_2$ regularization), Support Vector Machines (SVM) using Linear and Radial Basis Function (RBF) kernels.
- **Dataset/task**: UCI Spambase Dataset (4,210 cleaned records, 57 features).
- **Link**: [Experiment 4 Folder](./04_BInary%20classification/)

### Experiment 5 — Tumor Classification via Decision Tree & Random Forest
- **Objective**: Perform a comparative classification study on breast cancer diagnosis using tree-based classifiers.
- **Main techniques/algorithms**: Single Decision Tree Classifier, Random Forest Classifier (bootstrap aggregated ensemble of de-correlated decision trees with random feature subsets).
- **Dataset/task**: Wisconsin Diagnostic Breast Cancer (WDBC) Dataset (569 records, 30 continuous features).
- **Link**: [Experiment 5 Folder](./05_DecisionTree-RandomForest/)

### Experiment 6 — Ensemble Learning: Bagging, Boosting, & Stacking
- **Objective**: Build and compare homogeneous and heterogeneous ensembling models to classify tumor biopsies.
- **Main techniques/algorithms**: Bagging (using Decision Tree base estimators), Boosting (AdaBoost and Gradient Boosting), Stacking (using SVM, Gaussian Naïve Bayes, and Decision Tree base estimators with a Logistic Regression meta-learner).
- **Dataset/task**: Wisconsin Diagnostic Breast Cancer (WDBC) Dataset (569 records, 30 continuous features).
- **Link**: [Experiment 6 Folder](./06_Ensembling-Models/)

## Technologies Used
Across the experiments, the following python scientific computing stack is used:
- **Python 3**
- **NumPy**: Numerical computing and array manipulation
- **Pandas**: Structured data processing and cleaning
- **SciPy**: Scientific computing and statistical tests
- **Matplotlib**: Core plotting and visualization
- **Seaborn**: Statistical data visualization
- **Scikit-Learn**: Machine learning algorithms, preprocessing, and hyperparameter tuning tools
- **Jupyter Notebook**: Interactive execution environment

## How to Use This Repository
To run any of the experiments:
1. Open the required experiment folder (e.g. `cd 05_DecisionTree-RandomForest`).
2. Read the local experiment `README.md` for specific instructions.
3. Install that experiment's dependencies:
   ```bash
   pip install -r requirements.txt
   ```
4. Open the notebook:
   ```bash
   jupyter notebook <notebook_name>.ipynb
   ```
5. Run the notebook cells in order.

*Note: Each experiment directory has its own `requirements.txt` to ensure minimal and correct library footprints tailored to that specific implementation.*

## Documentation
The file [Manual (1).tex](./Manual%20(1).tex) defines the academic report template required for the laboratory submissions. It outlines guidelines for presenting objectives, problem statements, dataset descriptions, EDA inferences, math formulations, results tables, and conclusions. The compiled results and report drafts are stored as `.tex` files within their respective experiment folders.

## Repository Navigation
- [Experiment 1: Universal EDA](./01_Universal_EDA/)
- [Experiment 2: Email Spam Classification Naive Bayes/KNN](./02_Email_Spam_Classification/)
- [Experiment 3: Loan Amount Regression Analysis](./03_Regression_Analysis/)
- [Experiment 4: Logistic Regression and SVM Binary Classification](./04_BInary%20classification/)
- [Experiment 5: Decision Tree and Random Forest Comparison](./05_DecisionTree-RandomForest/)
- [Experiment 6: Bagging, Boosting, and Stacking Ensembles](./06_Ensembling-Models/)
