# MortgageBackedSecuritiesPrepaymentRiskModel
This project explores models using Freddie Mac's public dataset- Single Family Loan Level Dataset (Standard, Quarterly). The repository contains a IPYNB file, predicting mortgage-backed securities risk using the dataset and machine learning models. The project focuses on understanding borrower behavior and estimating delinquency-related risk at the loan level. 

### Project Overview
This notebook builds an end‑to‑end pipeline that ingests Freddie Mac–style single‑family loan data and predicts whether borrowers become delinquent and for how long. The workflow includes data cleaning, feature engineering, model training, and evaluation with multiple classification algorithms.

### Key Goals
- Identify important borrower and loan characteristics associated with delinquency risk.
- Train and compare machine learning models such as Logistic Regression, Random Forest, Gradient Boosting, and XGBoost for risk prediction.
- Provide a reproducible, exploratory framework for credit risk and MBS analytics in Python.

### Methodology
The analysis is implemented entirely in Python using standard data science and machine learning libraries.
#### Data Preprocessing
- Loading and consolidating raw exports into a single DataFrame using pandas.
- Parsing date‑like integer fields into proper datetime and extracting components such as FirstPaymentYear, FirstPaymentMonth, MaturityYear, and MaturityMonth.
- Handling categorical variables (e.g., Occupancy, LoanPurpose, PropertyState) via encoding suitable for tree and linear models.
- Managing missing values, including fields with substantial gaps such as SellerName.
- Scaling/standardizing numeric predictors with StandardScaler when required for linear models such as Logistic Regression.
#### Modeling
The notebook frames delinquency risk as a classification problem, with EverDelinquent as the primary target variable and MonthsDelinquent as a complementary measure of severity. Multiple supervised learning algorithms are trained and compared:
- Logistic Regression (Using different penalties and solvers: L1, L2, Elastic Net)
- Regularized Regression (Ridge and Lasso)
- Ensemble Tree Models (Random Forest Classifier, Gradient Boost Classifier)
Data is split into training and test sets using train_test_split, and cross‑validation is applied to estimate out‑of‑sample performance. Evaluation metrics include:
- Classification: accuracy, precision, recall, F1‑score, ROC curves, AUC, confusion matrices, and classification reports.
- Regression : R², mean squared error (MSE), and mean absolute error (MAE).​
### Conclusions
Across regression models, logistic with and L1 penalty and liblinear solver (Lasso) had the best accuracy in train and test scores. From ensemble tree models, both models produced the same accuracy and are also higher than all the regression models tested.
