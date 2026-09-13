# Tourism Package Prediction — Visit with Us

Predicting tourism package purchases using XGBoost, MLflow,
GitHub Actions, and Streamlit Community Cloud.

## Project links

- Live app: https://visitwithus.streamlit.app
- Repository: https://github.com/niki298/Tourism_Package_Prediction
- Automated workflow: https://github.com/niki298/Tourism_Package_Prediction/actions

## Business context and prediction scope

The project explores how customer data can support targeting for
a Wellness Tourism Package.

This implementation follows the learner template and uses customer
profiles together with sales-interaction details, including pitch
duration, follow-ups, and pitch satisfaction. Predictions therefore
apply after these details are available.

It does not fulfill a strict pre-contact prediction scenario.
A separate pre-contact model excluding unavailable features is
planned as a future learning extension.

## Platform choice

The project uses the GitHub + GitHub Actions + Streamlit Community
Cloud alternative permitted by the Program Office.

GitHub stores the dataset and source code. GitHub Actions artifacts
preserve prepared data and experiment outputs. The trained pipeline
and its configuration are versioned in GitHub for deployment.

## Data preparation

- Dataset: 4,128 records.
- Target: ProdTaken, where 1 means purchase and 0 means no purchase.
- Purchasers: 797, approximately 19.3% of records.
- Removed CustomerID and the Unnamed: 0 CSV index column.
- Standardized "Fe Male" to "Female".
- No missing values or repeated CustomerIDs were detected.
- Retained unusual numeric values because they were not confirmed errors.
- Used a stratified 80/20 split with random_state=42.
- Training: 3,302 records; test: 826 records.
- Model inputs: 12 numeric and 6 categorical features.

## Modeling and experiment tracking

The pipeline combines StandardScaler for numeric inputs,
OneHotEncoder for categorical inputs, and XGBoost.

Purchaser examples are weighted using the training-set
non-purchaser/purchaser ratio.

GridSearchCV compares 32 parameter combinations using five-fold
stratified cross-validation and average precision for selection.
Preprocessing is fitted within each training fold.

Seven classification thresholds are compared using out-of-fold
training predictions. The threshold with the highest F1 is selected;
exact ties favor the higher threshold.

MLflow records parameter combinations, cross-validation scores,
threshold comparisons, evaluation reports, and model artifacts.

## Selected production model

Trained by GitHub Actions using Python 3.11 and XGBoost 2.1.4:

| Setting | Value |
|---|---:|
| n_estimators | 200 |
| max_depth | 5 |
| learning_rate | 0.1 |
| colsample_bytree | 1.0 |
| colsample_bylevel | 1.0 |
| reg_lambda | 1.0 |
| Classification threshold | 0.50 |

Best mean cross-validation average precision: 0.8242.

## Held-out test results

These results describe the GitHub-trained deployment model,
not the earlier Colab model trained with XGBoost 3.4.1.

| Metric | Result |
|---|---:|
| Accuracy | 91.16% |
| Purchaser precision | 75.60% |
| Purchaser recall | 79.87% |
| Purchaser F1 | 0.7768 |
| Average precision | 0.8497 |

| Actual outcome | Predicted no purchase | Predicted purchase |
|---|---:|---:|
| No purchase | 626 | 41 |
| Purchase | 32 | 127 |

The model identified 127 of 159 purchasers and incorrectly flagged
41 non-purchasers.

Training purchaser F1 was 0.9694, higher than test F1, indicating
overfitting. Test results were not used to revise the selected
parameters or threshold.

## Repository structure

    .github/workflows/pipeline.yml
    README.md
    tourism_project/
        requirements.txt
        data/
            tourism.csv
        model_building/
            data_register.py
            prep.py
            train.py
        deployment/
            app.py
            requirements.txt
            best_tourism_package_model_v1.joblib
            model_config.json

## Automated workflow

Relevant pushes to main, or a manual workflow dispatch, run:

1. register-dataset: validate required columns and upload raw data.
2. data-prep: clean data and upload the four train/test split files.
3. model-training: start MLflow, tune and evaluate the model,
   preserve results, and commit the model and configuration to main.

The generated model commit uses [skip ci] to avoid a repeated
training cycle. Training-results artifacts are retained for 30 days
and should be downloaded for longer-term preservation.

The pipeline automates retraining and publication. It does not
currently include a model-performance gate or manual promotion review.

## Deployment

Streamlit Community Cloud is configured with:

- Repository: niki298/Tourism_Package_Prediction
- Branch: main
- Main file: tourism_project/deployment/app.py
- Python: 3.11, selected in Advanced settings before deployment

The app loads the saved preprocessing/model pipeline and reads
the threshold and input-column order from model_config.json.
No GitHub or ngrok token is required by the deployed app.

The hosted app was manually checked for positive predictions,
negative predictions, and rejection of child counts exceeding
the total travel-party size.

## Limitations and future work

- Interaction features prevent use as a strict pre-contact model.
- Historical performance does not establish increased sales or revenue.
- Model scores have not been evaluated for probability calibration.
- Performance may differ for future customers or unfamiliar input values.
- Out-of-fold threshold selection follows hyperparameter selection on
  the same training data; it is a development procedure, not an
  independent performance estimate.
- Future work includes a pre-contact model, monitoring, and business
  cost-based threshold selection.

## Credentials

Development credentials are stored in Colab Secrets.
Tokens should never be included in source code or committed to GitHub.
