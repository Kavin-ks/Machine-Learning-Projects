# Experiment 1: Universal Exploratory Data Analysis, Data Preprocessing, and Feature Engineering

## Overview
This experiment develops a reusable, automated Exploratory Data Analysis (EDA) and data preprocessing pipeline in Python that standardizes analysis across multiple benchmark datasets with varying characteristics (dimensionality, scale, and task types).

## Objective
To explore and leverage Python scientific computing and data analysis libraries to automate the inspection of dataset shape, data types, missing value ratios, duplicate records, class distributions, pairwise correlations, and baseline model evaluations.

## Dataset
Evaluated across five benchmark datasets:
1. **Iris Dataset**: Multi-class tabular classification.
2. **Loan Amount Prediction Dataset**: Tabular regression.
3. **Diabetes Dataset**: Binary tabular classification.
4. **MNIST Handwritten Digit Dataset**: High-dimensional image classification.
5. **SMS Spam Dataset**: Unstructured text binary classification.

## Methods / Algorithms
- **Automated Statistics**: Summary statistics and structural analysis via `Pandas` and `NumPy`.
- **Statistical Tests**: Hypothesis testing and distribution skewness checks via `SciPy`.
- **Data Visualization**: Correlation heatmaps, class distributions, and feature distributions using `Matplotlib` and `Seaborn`.
- **Baseline Models**: Logistic Regression, Random Forest, SVM, Naive Bayes, Decision Trees, and KNN classifiers via `Scikit-Learn` to establish performance baselines.

## Project Structure
- `universal_eda.ipynb`: Automated universal EDA function implementation (`perform_eda()`).
- `Iris.ipynb`: Preprocessing and baseline modeling for the Iris dataset.
- `Loan_amount_prediction.ipynb`: Preprocessing and regression modeling for loan amounts.
- `Diabetes.ipynb`: Analysis and classification of the diabetes dataset.
- `Handwritten_character_recognition.ipynb`: High-dimensional MNIST digit recognition baseline.
- `Classification_of_Email.ipynb`: Text preprocessing and Naive Bayes spam classification baseline.
- `Assignment_1_Report.tex`: Academic LaTeX report documenting results and figures.
- `Datasets/`: Subfolder containing raw dataset files.

## How to Run
1. Open the experiment folder:
   ```bash
   cd 01_Universal_EDA
   ```
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
3. Open Jupyter Notebook:
   ```bash
   jupyter notebook
   ```
4. Run the notebooks in order.

## Results
Provides structured baseline evaluations (accuracy, classification reports, and correlation heatmaps) across all five domains, showcasing the efficiency of automated EDA before running complex models.

## Technologies Used
- Python 3
- NumPy
- Pandas
- Matplotlib
- Seaborn
- SciPy
- Scikit-Learn
- Jupyter Notebook

## Notes
Ensure raw datasets are downloaded and placed in the `Datasets` folder as structured in the notebooks.
