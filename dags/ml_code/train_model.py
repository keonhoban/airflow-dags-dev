# ml_code/train_model.py
import os
import io
import boto3
import pandas as pd

import mlflow
import mlflow.sklearn
from mlflow.tracking import MlflowClient

from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

from ml_code.config import get_tracking_uri, get_experiment_name
from airflow.utils.log.logging_mixin import LoggingMixin

# ONNX export
from skl2onnx import convert_sklearn
from skl2onnx.common.data_types import FloatTensorType

logger = LoggingMixin().log


def _parse_s3_uri(uri: str):
    if not uri.startswith("s3://"):
        raise ValueError(f"Invalid s3 uri: {uri}")
    x = uri[5:]
    bkt, key = x.split("/", 1)
    return bkt, key


def _read_parquet_from_s3(feature_uri: str) -> pd.DataFrame:
    """
    Airflow 환경에서 s3fs 의존성 없이 boto3로 내려받아 parquet 읽기
    """
    bkt, key = _parse_s3_uri(feature_uri)
    s3 = boto3.client("s3")
    obj = s3.get_object(Bucket=bkt, Key=key)
    data = obj["Body"].read()
    return pd.read_parquet(io.BytesIO(data))


def _make_label(df: pd.DataFrame) -> pd.Series:
    """
    W8 통합용 proxy label (3-class)
    - f_total_events_7d 기반 버킷
      0: low, 1: mid, 2: high
    """
    x = df["f_total_events_7d"].astype(float)
    # 경계는 예시. 필요하면 변수화 가능하지만 W8은 성공이 우선
    return pd.cut(x, bins=[-1, 3, 7, 10**9], labels=[0, 1, 2]).astype(int)


def export_onnx_and_log_artifact(clf, n_features: int):
    """
    runs:/<run_id>/onnx/model.onnx 로 항상 저장
    zipmap 비활성화 (Triton outputs tensor 유지)
    """
    initial_type = [("input", FloatTensorType([None, n_features]))]
    onnx_model = convert_sklearn(
        clf,
        initial_types=initial_type,
        options={id(clf): {"zipmap": False}},
    )

    onnx_path = "/tmp/model.onnx"
    with open(onnx_path, "wb") as f:
        f.write(onnx_model.SerializeToString())

    mlflow.log_artifact(onnx_path, artifact_path="onnx")
    logger.info("[ONNX] logged: onnx/model.onnx")


def train_model(C, max_iter, feature_uri=None, fs_version=None, schema_hash=None):
    tracking_uri = get_tracking_uri()
    mlflow.set_tracking_uri(tracking_uri)
    client = MlflowClient(tracking_uri=tracking_uri)

    experiment_name = get_experiment_name()
    experiment = client.get_experiment_by_name(experiment_name)
    experiment_id = experiment.experiment_id if experiment else client.create_experiment(experiment_name)

    # -------------------------
    # Feature 로딩 (W8 핵심)
    # -------------------------
    if feature_uri:
        df = _read_parquet_from_s3(feature_uri)
        # feature columns
        X = df[["f_total_events_7d", "f_avg_session_sec_7d", "f_last_event_age_sec"]]
        y = _make_label(df)
        logger.info("[TRAIN] using feature_uri=%s rows=%d", feature_uri, len(df))
    else:
        raise ValueError("feature_uri is required for W8 full integration")

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

    with mlflow.start_run(experiment_id=experiment_id) as run:
        run_id = run.info.run_id
        logger.info("[DEBUG] ✅ run_id: %s", run_id)

        clf = LogisticRegression(C=C, max_iter=max_iter)
        clf.fit(X_train, y_train)

        y_pred = clf.predict(X_test)
        acc = accuracy_score(y_test, y_pred)

        # -------------------------
        # Proof(재현성) 기록
        # -------------------------
        mlflow.log_param("C", C)
        mlflow.log_param("max_iter", max_iter)
        mlflow.log_param("feature_uri", feature_uri)
        if fs_version:
            mlflow.log_param("fs_version", fs_version)
        if schema_hash:
            mlflow.log_param("schema_hash", schema_hash)

        mlflow.log_metric("accuracy", acc)

        # 기존 MLflow model package
        mlflow.sklearn.log_model(clf, "model")

        # ONNX artifact
        export_onnx_and_log_artifact(clf, n_features=X.shape[1])

        logger.info("[TRAIN] acc=%.4f feature_uri=%s fs_version=%s schema_hash=%s", acc, feature_uri, fs_version, schema_hash)
        return acc, run_id
