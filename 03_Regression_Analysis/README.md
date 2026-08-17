# Experiment 3: Regression Analysis using Linear and Regularized Models

## Overview
This experiment implements and evaluates linear regression models to predict a continuous target variable: **Loan Sanction Amount (USD)**. It investigates regularized linear models (Ridge, Lasso, and ElasticNet) to prevent overfitting and handle feature multicollinearity.

## Objective
To develop linear and regularized regression models, tune their regularization strength parameters ($\alpha$) using cross-validation, evaluate their performance using regression metrics ($R^2$, Adjusted $R^2$, MAE, MSE, and RMSE), and analyze model stability and the bias-variance trade-off.

## Dataset
- **Predict Loan Amount Dataset**: Tabular credit scoring dataset.
- **Size**: 30,000 applications, cleaned to 29,660 records (dropping records missing the target label).
- **Features**: 23 raw features (12 numerical, 11 categorical), expanded to 46 standardized attributes after one-hot encoding.
- **Target**: Continuous target representing the sanctioned loan amount in USD.

## Methods / Algorithms
- **Ordinary Least Squares (OLS) Linear Regression**
- **Ridge Regression ($L_2$ regularization)**
- **Lasso Regression ($L_1$ regularization/feature selection)**
- **ElasticNet Regression ($L_1$ and $L_2$ hybrid regularization)**
- **Hyperparameter Optimization**: `GridSearchCV` and `RandomizedSearchCV` for penalty weights.

## Project Structure
- `Loan_amount_prediction-ridge.ipynb`: Complete Python Jupyter Notebook containing the implementations and outputs.
- `Assignment_3_Report.tex`: Academic LaTeX report documenting the results and inferences.
- `train.csv`: Training split of the credit applications.
- `test.csv`: Testing split of the credit applications.
- `images/`: Directory containing generated EDA and regression evaluation plots.

## How to Run
1. Open the experiment folder:
   ```bash
   cd 03_Regression_Analysis
   ```
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
3. Run the notebook:
   ```bash
   jupyter notebook Loan_amount_prediction-ridge.ipynb
   ```

## Results
A comparative table lists regression metrics ($R^2$, Adjusted $R^2$, MAE, RMSE) on the test set. Plots show coefficient shrinkage paths and residual distributions to analyze regularized parameter behaviors.

## Technologies Used
- Python 3
- NumPy
- Pandas
- Matplotlib
- Scikit-Learn
- Jupyter Notebook

## Notes
Categorical attributes are imputed using the mode, and numerical features are imputed using the median before scaling. Lasso regression serves as an automatic feature selector by shrinking less important feature coefficients to zero.
