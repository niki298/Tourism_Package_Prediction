import os
import json

import pandas as pd
import numpy as np
import joblib
import mlflow
import xgboost as xgb

from sklearn.compose import make_column_transformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import (
    GridSearchCV,
    StratifiedKFold,
    ParameterGrid,
    cross_val_predict,
)
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    average_precision_score,
    precision_score,
    recall_score,
    f1_score,
)

# Connect to the MLflow server started by the workflow
mlflow.set_tracking_uri("http://127.0.0.1:5000")
mlflow.set_experiment("Tourism_Package_Prediction")

# Load the cleaned splits produced by prep.py
Xtrain = pd.read_csv("Xtrain.csv")
Xtest = pd.read_csv("Xtest.csv")
ytrain = pd.read_csv("ytrain.csv")["ProdTaken"]
ytest = pd.read_csv("ytest.csv")["ProdTaken"]

print("Prepared datasets loaded.")
print("Training features:", Xtrain.shape)
print("Test features:", Xtest.shape)

# Define the numeric and categorical input columns
numeric_features = [
    "Age",
    "CityTier",
    "DurationOfPitch",
    "NumberOfPersonVisiting",
    "NumberOfFollowups",
    "PreferredPropertyStar",
    "NumberOfTrips",
    "Passport",
    "PitchSatisfactionScore",
    "OwnCar",
    "NumberOfChildrenVisiting",
    "MonthlyIncome",
]

categorical_features = [
    "TypeofContact",
    "Occupation",
    "Gender",
    "ProductPitched",
    "MaritalStatus",
    "Designation",
]

# Check that every input feature is listed exactly once
listed_features = numeric_features + categorical_features

assert len(listed_features) == len(set(listed_features)), \
    "A feature has been listed more than once."

assert set(listed_features) == set(Xtrain.columns), \
    "The feature lists do not match the training columns."

assert Xtrain.columns.tolist() == Xtest.columns.tolist(), \
    "Training and test feature columns do not match."

# Calculate the purchaser weight from the training target
class_counts = ytrain.value_counts()
class_weight = class_counts[0] / class_counts[1]

# Define preprocessing
preprocessor = make_column_transformer(
    (StandardScaler(), numeric_features),
    (OneHotEncoder(handle_unknown="ignore"), categorical_features),
)

# Define XGBoost using the same settings as development
xgb_model = xgb.XGBClassifier(
    objective="binary:logistic",
    scale_pos_weight=class_weight,
    random_state=42,
    n_jobs=1,
)

# Connect preprocessing and the model
model_pipeline = make_pipeline(preprocessor, xgb_model)

print("Model pipeline configured.")

# Define the 32 parameter combinations to compare
param_grid = {
    "xgbclassifier__n_estimators": [100, 200],
    "xgbclassifier__max_depth": [3, 5],
    "xgbclassifier__colsample_bytree": [0.8, 1.0],
    "xgbclassifier__colsample_bylevel": [1.0],
    "xgbclassifier__learning_rate": [0.05, 0.1],
    "xgbclassifier__reg_lambda": [1.0, 5.0],
}

# Preserve class proportions within each validation fold
cv_strategy = StratifiedKFold(
    n_splits=5,
    shuffle=True,
    random_state=42,
)

# Select the configuration with the highest mean average precision
grid_search = GridSearchCV(
    estimator=model_pipeline,
    param_grid=param_grid,
    scoring="average_precision",
    cv=cv_strategy,
    n_jobs=2,
    refit=True,
    return_train_score=True,
    error_score="raise",
)

print("Parameter combinations:", len(list(ParameterGrid(param_grid))))
print("Grid search configured.")

# Create a parent run for this automated training execution
with mlflow.start_run(run_name="XGBoost_Production_Training"):
    training_run_id = mlflow.active_run().info.run_id

    # Record the search configuration
    mlflow.log_params({
        "selection_metric": "average_precision",
        "cv_folds": 5,
        "cv_shuffle": True,
        "random_state": 42,
        "scale_pos_weight": float(class_weight),
        "classification_objective": "binary:logistic",
        "parameter_combinations": len(list(ParameterGrid(param_grid))),
    })

    # Train and compare all parameter combinations
    print("Starting grid search...", flush=True)
    grid_search.fit(Xtrain, ytrain)

    # Record each combination as a child run
    results = grid_search.cv_results_

    for i, parameters in enumerate(results["params"]):
        with mlflow.start_run(
            run_name=f"Combination_{i + 1:02d}",
            nested=True,
        ):
            mlflow.log_params(parameters)
            mlflow.log_metrics({
                "mean_validation_average_precision":
                    float(results["mean_test_score"][i]),
                "std_validation_average_precision":
                    float(results["std_test_score"][i]),
                "mean_train_average_precision":
                    float(results["mean_train_score"][i]),
            })

    # Keep the selected pipeline, already refitted on all training data
    best_model = grid_search.best_estimator_

    mlflow.log_params(grid_search.best_params_)
    mlflow.log_metric(
        "best_cv_average_precision",
        float(grid_search.best_score_),
    )

    # Save the full search results for later inspection
    pd.DataFrame(results).to_csv("grid_search_results.csv", index=False)
    mlflow.log_artifact("grid_search_results.csv")

print("Grid search completed.")
print(f"Best CV average precision: {grid_search.best_score_:.4f}")
print("Best parameters:", grid_search.best_params_)

# Generate out-of-fold scores using the selected model settings
print("Generating out-of-fold scores...", flush=True)

oof_probabilities = cross_val_predict(
    best_model,
    Xtrain,
    ytrain,
    cv=cv_strategy,
    method="predict_proba",
    n_jobs=2,
)[:, 1]

# Compare the same candidate thresholds used in development
threshold_results = []

for threshold in [0.2, 0.3, 0.4, 0.45, 0.5, 0.6, 0.7]:
    predictions = (oof_probabilities >= threshold).astype(int)

    threshold_results.append({
        "Threshold": threshold,
        "Precision": precision_score(
            ytrain, predictions, zero_division=0
        ),
        "Recall": recall_score(
            ytrain, predictions, zero_division=0
        ),
        "F1": f1_score(
            ytrain, predictions, zero_division=0
        ),
        "Customers flagged": int(predictions.sum()),
    })

threshold_comparison = pd.DataFrame(threshold_results)

# Select the highest F1; break exact ties using the higher threshold
selected_row = threshold_comparison.sort_values(
    by=["F1", "Threshold"],
    ascending=[False, False],
).iloc[0]

classification_threshold = float(selected_row["Threshold"])

# Save and log the decision
threshold_comparison.to_csv("threshold_comparison.csv", index=False)

with mlflow.start_run(run_id=training_run_id):
    mlflow.log_param(
        "classification_threshold",
        classification_threshold,
    )
    mlflow.log_param(
        "threshold_selection_method",
        "Highest OOF F1 among seven candidates; higher threshold breaks ties",
    )
    mlflow.log_artifact("threshold_comparison.csv")

print(threshold_comparison.round(4).to_string(index=False))
print("Selected threshold:", classification_threshold)

# Evaluate the fitted pipeline using the selected threshold
train_probabilities = best_model.predict_proba(Xtrain)[:, 1]
test_probabilities = best_model.predict_proba(Xtest)[:, 1]

train_predictions = (
    train_probabilities >= classification_threshold
).astype(int)

test_predictions = (
    test_probabilities >= classification_threshold
).astype(int)

train_report = classification_report(
    ytrain, train_predictions, output_dict=True, zero_division=0
)
test_report = classification_report(
    ytest, test_predictions, output_dict=True, zero_division=0
)

test_ap = average_precision_score(ytest, test_probabilities)

test_confusion = pd.DataFrame(
    confusion_matrix(ytest, test_predictions, labels=[0, 1]),
    index=["Actual no purchase", "Actual purchase"],
    columns=["Predicted no purchase", "Predicted purchase"],
)

# Print results so they are visible in the workflow logs
print("\nTRAINING PERFORMANCE")
print(classification_report(
    ytrain, train_predictions, digits=4, zero_division=0
))

print("TEST PERFORMANCE")
print(classification_report(
    ytest, test_predictions, digits=4, zero_division=0
))

print("TEST CONFUSION MATRIX")
print(test_confusion.to_string())
print(f"\nTest average precision: {test_ap:.4f}")

# Save the fitted pipeline and the configuration used by the app
deployment_dir = "tourism_project/deployment"
os.makedirs(deployment_dir, exist_ok=True)

model_path = os.path.join(
    deployment_dir,
    "best_tourism_package_model_v1.joblib",
)
joblib.dump(best_model, model_path)

config = {
    "classification_threshold": classification_threshold,
    "feature_columns": Xtrain.columns.tolist(),
    "prediction_stage": "After customer interaction details are available",
}

config_path = os.path.join(deployment_dir, "model_config.json")

with open(config_path, "w") as file:
    json.dump(config, file, indent=2)

# Log evaluation results and saved files
with mlflow.start_run(run_id=training_run_id):
    for split_name, report in [
        ("train", train_report),
        ("test", test_report),
    ]:
        mlflow.log_metrics({
            f"{split_name}_accuracy": report["accuracy"],
            f"{split_name}_precision": report["1"]["precision"],
            f"{split_name}_recall": report["1"]["recall"],
            f"{split_name}_f1": report["1"]["f1-score"],
        })

    mlflow.log_metric("test_average_precision", float(test_ap))

    mlflow.log_dict(
        {"train": train_report, "test": test_report},
        "evaluation/classification_reports.json",
    )
    mlflow.log_dict(
        test_confusion.to_dict(),
        "evaluation/test_confusion_matrix.json",
    )

    mlflow.log_artifact(model_path, artifact_path="model")
    mlflow.log_artifact(config_path, artifact_path="model")

print("Saved model:", model_path)
print("Saved configuration:", config_path)
