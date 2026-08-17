# Experiment 2: Email Spam/Ham Classification using Naïve Bayes and KNN

## Overview
This experiment implements and compares different supervised machine learning models (specifically Naïve Bayes and K-Nearest Neighbors) to classify email messages as either legitimate (Ham) or unsolicited (Spam).

## Objective
To build and evaluate spam classifiers, analyze Gaussian, Multinomial, and Bernoulli Naïve Bayes variants, examine KNN spatial structures (KDTree and BallTree) under varying neighborhood sizes ($k$), optimize hyperparameters via grid and randomized searches, and investigate empirical training/inference complexities.

## Dataset
- **SpamBase Dataset**: Obtained from the UCI Machine Learning Repository / Kaggle.
- **Size**: 4,601 raw records, reduced to 4,210 unique records after removing duplicates.
- **Features**: 57 continuous attributes (48 word frequencies, 6 character frequencies, and 3 capital run-length statistics).
- **Target**: Binary classification (0: Non-Spam/Ham, 1: Spam).

## Methods / Algorithms
- **Naïve Bayes Classifiers**: Gaussian NB, Multinomial NB, and Bernoulli NB.
- **K-Nearest Neighbors (KNN)**: KDTree and BallTree spatial indexes.
- **Hyperparameter Optimization**: `GridSearchCV` and `RandomizedSearchCV` for KNN.
- **Evaluation**: 5-Fold Stratified Cross-Validation on the training set (80% split) and evaluation on the test set (20% split).

## Project Structure
- `Naive Bayes/SpamBase-KNN/Naive.ipynb`: Complete Python Jupyter Notebook containing the implementations and outputs.
- `Naive Bayes/Assignment_2_Report.tex`: Academic LaTeX documentation.
- `Naive Bayes/spambase_csv.csv`: Normalized dataset containing 57 word/character frequency columns and the binary class label.
- `Naive Bayes/images/`: Directory containing generated EDA and evaluation plots.

## How to Run
1. Open the experiment folder:
   ```bash
   cd 02_Email_Spam_Classification
   ```
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
3. Run the notebook inside the subfolder:
   ```bash
   jupyter notebook "Naive Bayes/SpamBase-KNN/Naive.ipynb"
   ```

## Results
A comparative analysis evaluates classification accuracy, precision, recall, F1-score, and ROC-AUC on the test set. It examines the execution time scaling of KDTree vs. BallTree as $k$ changes.

## Technologies Used
- Python 3
- NumPy
- Pandas
- SciPy
- Matplotlib
- Seaborn
- Scikit-Learn
- Jupyter Notebook

## Notes
The dataset contains continuous capital letter run-length statistics; Bernoulli Naïve Bayes requires feature binarization, while Multinomial Naïve Bayes requires non-negative features.
