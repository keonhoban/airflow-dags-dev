from __future__ import annotations
import os, json
import pandas as pd
import numpy as np
import mlflow
from mlflow.tracking import MlflowClient

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

class TrainSkippableError(Exception):
    pass

def train_and_log_model(feature_uri: str, fs_version: str, schema_hash: str, env: str, C: float = 1.0, max_iter: int = 200):
    mlflow.set_tracking_uri(os.environ["MLFLOW_TRACKING_URI"])
    df = pd.read_parquet(feature_uri)

    if "label" not in df.columns:
        raise TrainSkippableError("label column missing")

    y = df["label"].astype(int).values
    X = df.drop(columns=["label"]).select_dtypes(include=[np.number]).values

    if X.shape[0] < 20:
        raise TrainSkippableError(f"too few rows: {X.shape[0]}")

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y if len(set(y)) > 1 else None)

    model = LogisticRegression(C=C, max_iter=max_iter, multi_class="auto")
    model.fit(X_train, y_train)
    pred = model.predict(X_test)
    acc = float(accuracy_score(y_test, pred))

    with mlflow.start_run() as run:
        run_id = run.info.run_id

        # params
        mlflow.log_param("C", C)
        mlflow.log_param("max_iter", max_iter)
        mlflow.log_param("feature_uri", feature_uri)
        mlflow.log_param("fs_version", fs_version)
        mlflow.log_param("schema_hash", schema_hash)
        mlflow.log_param("train_rows", int(X_train.shape[0]))
        mlflow.log_param("test_rows", int(X_test.shape[0]))
        mlflow.log_param("n_features", int(X.shape[1]))
        mlflow.log_param("n_classes", int(len(set(y))))

        # metrics
        mlflow.log_metric("accuracy", acc)

        # tags (면접용 핵심)
        mlflow.set_tag("env", env)
        mlflow.set_tag("fs_version", fs_version)
        mlflow.set_tag("schema_hash", schema_hash)
        mlflow.set_tag("feature_uri", feature_uri)

        # model artifact (sklearn)
        mlflow.sklearn.log_model(model, artifact_path="model")

    return acc, run_id

