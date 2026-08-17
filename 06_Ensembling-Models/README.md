# Experiment 6: Bagging, Boosting, and Stacked Ensemble Models

## Overview
This experiment implements and compares different ensemble learning strategies—namely **Bagging**, **Boosting** (AdaBoost and Gradient Boosting), and **Stacking**—on the **Wisconsin Diagnostic Breast Cancer (WDBC) Dataset** to evaluate how aggregating multiple models impacts prediction accuracy, bias, and variance.

## Objective
To understand ensemble classification, implement bootstrap aggregation (Bagging) and sequential boosting (AdaBoost and Gradient Boosting), construct a heterogeneous Stacked Ensemble using multiple base learners (SVM, Gaussian Naïve Bayes, and a Decision Tree) combined via a Logistic Regression meta-learner, tune model hyperparameters using 5-Fold Stratified Cross-Validation, and analyze the generalization gap.

## Dataset
- **Wisconsin Diagnostic Breast Cancer (WDBC) Dataset**: Extracted from UCI Machine Learning Repository (accessed locally from Experiment 5's directory).
- **Size**: 569 patient biopsy records.
- **Features**: 30 continuous features computed from digitized cell nuclei images (e.g., radius, texture, area).
- **Target**: Binary classification (0: Benign, 1: Malignant).

## Methods / Algorithms
- **Bagging Classifier**: Bootstrap aggregation utilizing Decision Tree base estimators.
- **AdaBoost Classifier**: Sequential boosting prioritizing misclassified instances, using Decision Tree weak learners.
- **Gradient Boosting Classifier**: Optimization of differentiable loss functions using regression trees.
- **Stacked Ensemble**: Heterogeneous stacking using SVC, GaussianNB, and DecisionTreeClassifier as base estimators, combined via a Logistic Regression meta-learner.
- **Hyperparameter Optimization**: Grid Search (`GridSearchCV`) under 5-Fold Stratified Cross-Validation.

## Project Structure
- `Ensemble_models.ipynb`: Complete Python Jupyter Notebook containing the implementations and outputs.
- `Experiment_6.pdf`: PDF containing the experiment objectives and structure description.
- `images/`: Folder containing extracted figures (class distribution, pairwise correlation heatmap, confusion matrices, and ROC curves).

## How to Run
1. Open the experiment folder:
   ```bash
   cd 06_Ensembling-Models
   ```
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
3. Run the notebook:
   ```bash
   jupyter notebook Ensemble_models.ipynb
   ```

## Results
A final performance comparison table reports Accuracy, Precision, Recall, F1-score, and ROC-AUC on the test set for all four ensemble models. Generalization gaps (Training vs. Testing Accuracy) and training times are analyzed to assess overfitting.

## Technologies Used
- Python 3
- NumPy
- Pandas
- Matplotlib
- Seaborn
- Scikit-Learn
- Jupyter Notebook

## Notes
Because some base learners in the Stacking Classifier (such as SVM) rely on geometric distances, a preprocessing pipeline standardizes the training features using `StandardScaler` inside the SVM estimator pipeline to prevent feature scale dominance.
