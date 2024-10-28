import xgboost as xgb
from sklearn.metrics import r2_score
from sklearn import datasets
from sklearn.model_selection import train_test_split
import optuna
from optuna.integration.mlflow import MLflowCallback
import mlflow
import os
import argparse

import warnings
warnings.filterwarnings("ignore")


def objective(trial):
    (data, target) = datasets.load_boston(return_X_y=True)
    train_x, valid_x, train_y, valid_y = train_test_split(data, target, test_size=0.25)
    dtrain = xgb.DMatrix(train_x, label=train_y)
    dvalid = xgb.DMatrix(valid_x, label=valid_y)

    param = {
        "objective": "reg:squarederror",
        "n_estimators": trial.suggest_int("n_estimators", 50, 150, step=5),
        "max_depth": trial.suggest_int("max_depth", 4, 8, step=1),
        "learning_rate": trial.suggest_float("learning_rate", 0.3, 1.0),
        "subsample": trial.suggest_float("subsample", 0.2, 1.0),
    }

    model = xgb.train(param, dtrain)
    preds = model.predict(dvalid)
    r2 = r2_score(valid_y, preds)
    return r2



if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('--study-name', help='study name for optuna')
    parser.add_argument('--n-trials', type=int, help='the number of trials for optuna')
    args = parser.parse_args()
    
    study_name = f"{args.study_name}"
    storage_url = "mysql+pymysql://{}:{}@{}:{}/{}".format(
        os.getenv("DB_USERNAME"),
        os.getenv("DB_PASSWORD"),
        os.getenv("DB_HOST"),
        int(os.getenv("DB_PORT")),
        os.getenv("DB_DATABASE")
    )

    existing_studies = [s.study_name for s in optuna.study.get_all_study_summaries(storage=storage_url)]

    if study_name in existing_studies:
        study = optuna.load_study(study_name=study_name, storage=storage_url)
    else:
        study = optuna.create_study(
            study_name=study_name,
            storage=storage_url,
            direction="minimize",
            load_if_exists=True
        )


    tracking_uri = os.getenv("MLFLOW_TRACKING_URI")
    mlflow.set_tracking_uri(tracking_uri)
    experiment = mlflow.get_experiment_by_name(study_name)
    if experiment is None:
        experiment_id = mlflow.create_experiment(study_name)
    else:
        if experiment.lifecycle_stage == "deleted":
            mlflow.tracking.MlflowClient().delete_experiment(experiment.experiment_id)
            experiment_id = mlflow.create_experiment(study_name)  # 完全に削除後、新規作成
        else:
            experiment_id = experiment.experiment_id  # 有効なエクスペリメントIDを使用


    mlflc = MLflowCallback(
        tracking_uri=tracking_uri,
        metric_name='r2 score', # r2に修正すること！
        mlflow_kwargs={
            "experiment_id": experiment_id,
        }
    )

    study.optimize(objective, n_trials=args.n_trials, timeout=300, callbacks=[mlflc])