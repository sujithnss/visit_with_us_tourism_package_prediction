# for data manipulation
import pandas as pd
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import make_column_transformer
from sklearn.pipeline import make_pipeline
# for model training, tuning, and evaluation
import xgboost as xgb
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
# for model serialization
import joblib
# for creating a folder
import os
# for hugging face space authentication to upload files
from huggingface_hub import login, HfApi, create_repo
from huggingface_hub.utils import RepositoryNotFoundError, HfHubHTTPError
import mlflow

mlflow.set_tracking_uri("http://localhost:5000")
mlflow.set_experiment("mlops-training-experiment")

api = HfApi()


Xtrain_path = "hf://datasets/sujithpv/visit-with-us-tourism-package-prediction/Xtrain.csv"
Xtest_path = "hf://datasets/sujithpv/visit-with-us-tourism-package-prediction/Xtest.csv"
ytrain_path = "hf://datasets/sujithpv/visit-with-us-tourism-package-prediction/ytrain.csv"
ytest_path = "hf://datasets/sujithpv/visit-with-us-tourism-package-prediction/ytest.csv"

Xtrain = pd.read_csv(Xtrain_path)
Xtest = pd.read_csv(Xtest_path)
ytrain = pd.read_csv(ytrain_path)
ytest = pd.read_csv(ytest_path)

numeric_features = [
    'Age',
    'NumberOfPersonVisiting',
    'PreferredPropertyStar',
    'NumberOfTrips',
    'Passport',
    'OwnCar',
    'NumberOfChildrenVisiting',
    'MonthlyIncome',
    'PitchSatisfactionScore',
    'NumberOfFollowups',
    'DurationOfPitch'
]

categorical_features = [
    'TypeofContact',
    'CityTier',
    'Occupation',
    'Gender',
    'MaritalStatus',
    'Designation',
    'ProductPitched'
]

preprocessor = make_column_transformer(
    (StandardScaler(), numeric_features),
    (OneHotEncoder(handle_unknown='ignore'), categorical_features)
)

# Use XGBClassifier for classification task
xgb_model = xgb.XGBClassifier(random_state=42, use_label_encoder=False, eval_metric='logloss', n_jobs=-1)

# Hyperparameter grid with correct prefix for pipeline step 'xgbclassifier'
param_grid = {
    'xgbclassifier__n_estimators': [50, 100],
    'xgbclassifier__max_depth': [3, 5],
    'xgbclassifier__learning_rate': [0.01, 0.05],
    'xgbclassifier__subsample': [0.7, 0.8],
    'xgbclassifier__colsample_bytree': [0.7, 0.8],
    'xgbclassifier__reg_lambda': [0.1, 1]
}

model_pipeline = make_pipeline(preprocessor, xgb_model)

with mlflow.start_run():
    # Use scoring 'recall' or 'f1' as per your priority
    grid_search = GridSearchCV(model_pipeline, param_grid, cv=3, n_jobs=-1, scoring='f1')
    grid_search.fit(Xtrain, ytrain)

    results = grid_search.cv_results_
    for i in range(len(results['params'])):
        param_set = results['params'][i]
        mean_score = results['mean_test_score'][i]
        with mlflow.start_run(nested=True):
            mlflow.log_params(param_set)
            mlflow.log_metric("mean_f1_score", mean_score)

    mlflow.log_params(grid_search.best_params_)
    best_model = grid_search.best_estimator_

    y_pred_train = best_model.predict(Xtrain)
    y_pred_test = best_model.predict(Xtest)

    # Calculate classification metrics
    train_recall = recall_score(ytrain, y_pred_train)
    test_recall = recall_score(ytest, y_pred_test)

    train_f1 = f1_score(ytrain, y_pred_train)
    test_f1 = f1_score(ytest, y_pred_test)

    train_precision = precision_score(ytrain, y_pred_train)
    test_precision = precision_score(ytest, y_pred_test)

    # Optional: calculate ROC-AUC with probabilities
    y_prob_train = best_model.predict_proba(Xtrain)[:, 1]
    y_prob_test = best_model.predict_proba(Xtest)[:, 1]

    train_roc_auc = roc_auc_score(ytrain, y_prob_train)
    test_roc_auc = roc_auc_score(ytest, y_prob_test)

    mlflow.log_metrics({
        "train_recall": train_recall,
        "test_recall": test_recall,
        "train_f1": train_f1,
        "test_f1": test_f1,
        "train_precision": train_precision,
        "test_precision": test_precision,
        "train_roc_auc": train_roc_auc,
        "test_roc_auc": test_roc_auc
    })

    # Save the model locally
    model_path = "visit_with_us_tourism_package_prediction_model_v1.joblib"
    joblib.dump(best_model, model_path)

    # Log the model artifact
    mlflow.log_artifact(model_path, artifact_path="model")
    print(f"Model saved as artifact at: {model_path}")

    # Upload to Hugging Face
    repo_id = "sujithpv/visit-with-us-tourism-package-prediction"
    repo_type = "model"

    # Step 1: Check if the space exists
    try:
        api.repo_info(repo_id=repo_id, repo_type=repo_type)
        print(f"Space '{repo_id}' already exists. Using it.")
    except RepositoryNotFoundError:
        print(f"Space '{repo_id}' not found. Creating new space...")
        create_repo(repo_id=repo_id, repo_type=repo_type, private=False)
        print(f"Space '{repo_id}' created.")

    # create_repo("churn-model", repo_type="model", private=False)
    api.upload_file(
        path_or_fileobj="visit_with_us_tourism_package_prediction_model_v1.joblib",
        path_in_repo="visit_with_us_tourism_package_prediction_model_v1.joblib",
        repo_id=repo_id,
        repo_type=repo_type,
    )
