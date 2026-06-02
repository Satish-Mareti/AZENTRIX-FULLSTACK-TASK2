# Telco Customer Churn Report

## Problem

The Telco churn dataset is a realistic binary classification problem with class imbalance, mixed data types, missing values disguised as blanks, and categorical values that need encoding. A toy-model workflow is not enough, so the pipeline was built to cover the full path from cleaning to deployment.

## Data understanding and EDA

The raw dataset contains 7,043 rows and the standard Telco churn fields. The target is `Churn`, which is imbalanced, so the first EDA step measures the class split, missing values, and the numeric distributions of `SeniorCitizen`, `tenure`, `MonthlyCharges`, and `TotalCharges`.

The EDA summary is generated automatically and saved to `artifacts/eda_summary.md`. That keeps the analysis reproducible and tied to the actual training run.

## Feature engineering

Three engineered features were added:

- `tenure_charge_ratio`: captures how much the customer has paid relative to monthly spend.
- `service_count`: counts how many add-on services a customer has active.
- `is_month_to_month`: flags the highest-risk contract type.

These features were chosen because churn risk in telecom is usually driven by contract structure, usage depth, and spend patterns. The raw categorical features were kept because the model still benefits from learning the exact service and payment combinations.

## Preprocessing and imbalance handling

The pipeline uses a `ColumnTransformer` with median imputation and standardization for numeric data, and most-frequent imputation plus one-hot encoding for categorical data.

Imbalance is handled with SMOTE inside the training pipeline. This keeps the test set untouched while giving the classifiers a more balanced training signal. In addition, some models use class weights so the comparison covers more than one imbalance strategy.

## Model comparison

Three models are compared with 5-fold stratified cross-validation:

- Logistic Regression
- Random Forest
- Gradient Boosting

They were chosen to cover a strong linear baseline, a bagging-based tree model, and a boosting model. The comparison is scored with accuracy, ROC AUC, F1, and recall, but F1 is used as the main model-selection metric because the positive class is the minority class and missing churners is more expensive than a few false positives.

## Hyperparameter tuning

The best cross-validated model is tuned with `RandomizedSearchCV`. The search space is intentionally small and practical so the project remains runnable on a normal laptop while still showing a real tuning step.

## Evaluation

The tuned model is evaluated on a held-out test split using:

- Accuracy
- Precision
- Recall
- F1 score
- ROC AUC

The classification report is printed at the end of training and the final pipeline is stored as a single `joblib` artifact.

## Prediction interface

A simple CLI is used for prediction. Users can either answer prompts interactively or pass a JSON payload. The CLI loads the saved pipeline and returns the churn label plus the predicted probability.

## What could be improved

- Add probability calibration so predicted churn probabilities are better aligned with reality.
- Try stronger boosting models such as XGBoost, LightGBM, or CatBoost.
- Add plotting for the EDA step and export charts into a report bundle.
- Add unit tests for feature engineering and input validation.
- Package the CLI as a small web app if a browser-based interface is preferred.
