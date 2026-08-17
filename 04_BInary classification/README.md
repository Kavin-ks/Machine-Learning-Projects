# Experiment 4: Binary Classification using Logistic Regression and Support Vector Machine (SVM)

## Overview
This experiment implements and compares linear and kernel-based binary classifiers on the **Spambase Dataset** to determine whether emails are Spam or Ham. Specifically, it contrasts **Logistic Regression** and **Support Vector Machines (SVM)**.

## Objective
To build robust binary classifiers, analyze the impact of regularization strengths ($C$), explore SVM kernel functions (Linear, Radial Basis Function/RBF, and Polynomial), tune hyperparameters using Grid and Randomized search under 5-Fold Stratified Cross-Validation, and evaluate model performance.

## Dataset
- **Spambase Dataset**: Available from the UCI Machine Learning Repository.
- **Size**: 4,601 samples, cleaned to 4,210 instances after removing duplicate records.
- **Features**: 57 continuous attributes representing normalized frequency metrics of terms and specific characters.
- **Target**: Binary classification (0: Legitimate/Ham, 1: Spam).

## Methods / Algorithms
- **Logistic Regression**: Linear probability classifier with L2 regularization.
- **Support Vector Machine (SVM)**: Support Vector Classifier (SVC) using Linear and RBF kernels.
- **Hyperparameter Optimization**: Grid Search (`GridSearchCV`) and Randomized Search (`RandomizedSearchCV`) to optimize penalty cost $C$ and RBF kernel width $\gamma$.
- **Evaluation**: Classification Accuracy, Precision, Recall, F1-score, and ROC-AUC.

## Project Structure
- `Experiment -4.ipynb`: Complete Python Jupyter Notebook containing the implementations and outputs.
- `Assignment_4_Report.tex`: Academic LaTeX documentation.
- `spambase.csv`: Normalized dataset containing 57 predictor columns and target variables.
- `images/`: Folder containing generated plots (class distribution, top feature correlations, confusion matrices, and ROC curves).

## How to Run
1. Open the experiment folder:
   ```bash
   cd "04_BInary classification"
   ```
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
3. Run the notebook:
   ```bash
   jupyter notebook "Experiment -4.ipynb"
   ```

## Results
A comparison details the performance metrics of Logistic Regression and SVM on an independent test set. Confusion matrices and ROC curves visualize the models' abilities to control false positives and negatives.

## Technologies Used
- Python 3
- NumPy
- Pandas
- Matplotlib
- Seaborn
- Scikit-Learn
- Jupyter Notebook

## Notes
Because SVMs find optimal hyperplanes based on geometric margins, feature standardization using `StandardScaler` is crucial for training and must be applied prior to model fitting.
