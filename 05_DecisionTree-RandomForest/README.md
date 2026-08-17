# Experiment 5: Decision Tree and Random Forest: A Comparative Classification Study

## Overview
This experiment implements and compares a single **Decision Tree** classifier and a **Random Forest** ensemble classifier on the **Wisconsin Diagnostic Breast Cancer (WDBC) Dataset** to diagnose tumors as benign or malignant.

## Objective
To study classification performance using tree-based methods, analyze the effects of critical model hyperparameters (splitting criterion, maximum tree depth, minimum split samples, number of estimators, feature subset sizes, and bootstrapping), and evaluate how ensemble voting improves generalization and reduces variance.

## Dataset
- **Wisconsin Diagnostic Breast Cancer (WDBC) Dataset**: Extracted from UCI Machine Learning Repository.
- **Size**: 569 patient biopsy records.
- **Features**: 30 continuous features computed from digitized cell nuclei images (e.g., radius, texture, perimeter, area, concavity).
- **Target**: Binary classification (0: Malignant, 1: Benign).

## Methods / Algorithms
- **Decision Tree Classifier**: Recursive partitioning using Gini impurity or Information Gain/Entropy.
- **Random Forest Classifier**: Ensemble bootstrap aggregation (bagging) of de-correlated decision trees with random feature subsets.
- **Hyperparameter Optimization**: Grid Search (`GridSearchCV`) under 5-Fold Stratified Cross-Validation.
- **Evaluation**: Classification Accuracy, Precision, Recall, F1-score, and ROC-AUC.

## Project Structure
- `decisiontree_randomforest.ipynb`: Complete Python Jupyter Notebook containing the implementations and outputs.
- `Experiment_5_Report.tex`: Academic LaTeX documentation.
- `wdbc.data`: UCI dataset file containing the biopsy geometry records.
- `images/`: Folder containing generated plots (class distribution, pairwise correlation heatmap, confusion matrices, and ROC curves).

## How to Run
1. Open the experiment folder:
   ```bash
   cd 05_DecisionTree-RandomForest
   ```
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
3. Run the notebook:
   ```bash
   jupyter notebook decisiontree_randomforest.ipynb
   ```

## Results
A comparison evaluates performance metrics of the single Decision Tree vs. the Random Forest ensemble on the independent test set. Inferences analyze how restricting depth and features controls overfitting.

## Technologies Used
- Python 3
- NumPy
- Pandas
- Matplotlib
- Seaborn
- Scikit-Learn
- Jupyter Notebook

## Notes
Because tree-based classifiers make axis-aligned split partitions, they are scale-invariant. Consequently, feature scaling (like standardization or min-max normalization) is not required for data preparation.
