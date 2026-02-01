import io
import os
import boto3
import pandas as pd

import mlflow
import mlflow.sklearn
from mlflow.tracking import MlflowClient

from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score

from airflow.utils.log.logging_mixin import LoggingMixin

from skl2onnx import convert_sklearn
from skl2onnx.common.data_types import FloatTensorType

from ml_code.config import get_tracking_uri, get_experiment_name

logger = LoggingMixin().log


class TrainSkippableError(RuntimeError):
    pass


def _parse_s3_uri(uri: str):
    if not uri.startswith("s3://"):
        raise ValueError(f"Invalid s3 uri: {uri}")
    x = uri[5:]
    bkt, key = x.split("/", 1)
    return bkt, key


def _read_parquet_from_s3(feature_uri: str) -> pd.DataFrame:
    bkt, key = _parse_s3_uri(feature_uri)
    s3 = boto3.client("s3")
    obj = s3.get_object(Bucket=bkt, Key=key)
    data = obj["Body"].read()
    return pd.read_parquet(io.BytesIO(data))


def export_onnx_and_log_artifact(clf, n_features: int):
    initial_type = [("input", FloatTensorType([None, n_features]))]
    onnx_model = convert_sklearn(
        clf,
        initial_types=initial_type,
        options={id(clf): {"zipmap": False}},
    )

    try:
        in_name = onnx_model.graph.input[0].name
        out_names = [o.name for o in onnx_model.graph.output]
        mlflow.log_param("onnx_input_name", in_name)
        mlflow.log_param("onnx_output_names", ",".join(out_names))
        logger.info("[ONNX] io names input=%s output=%s", in_name, out_names)
    except Exception as e:
        logger.warning("[ONNX] failed to extract io names: %s", e)

    onnx_path = "/tmp/model.onnx"
    with open(onnx_path, "wb") as f:
        f.write(onnx_model.SerializeToString())

    mlflow.log_artifact(onnx_path, artifact_path="onnx")
    logger.info("[ONNX] logged: onnx/model.onnx")


def train_model(C, max_iter, feature_uri=None, fs_version=None, schema_hash=None, env=None, code_version=None):
    tracking_uri = get_tracking_uri()
    mlflow.set_tracking_uri(tracking_uri)
    client = MlflowClient(tracking_uri=tracking_uri)

    experiment_name = get_experiment_name()
    exp = client.get_experiment_by_name(experiment_name)
    experiment_id = exp.experiment_id if exp else client.create_experiment(experiment_name)

    if not feature_uri:
        raise ValueError("feature_uri is required")

    df = _read_parquet_from_s3(feature_uri)

    feature_cols = ["f_total_events_7d", "f_avg_session_sec_7d", "f_last_event_age_sec"]
    missing = [c for c in feature_cols if c not in df.columns]
    if missing:
        raise TrainSkippableError(f"학습 스킵: feature 컬럼 누락 {missing}")

    # ✅ label은 DP에서 생성되어 있어야 함
    if "label" not in df.columns:
        raise TrainSkippableError("학습 스킵: label 컬럼이 없습니다 (DP build 단계에서 label 생성 필요)")

    # 품질 체크: 데이터 너무 작으면 학습 의미 없음
    if len(df) < 20:
        raise TrainSkippableError(f"학습 스킵: rows={len(df)} (데모 최소 20 권장, 운영은 200+ 권장)")

    y = df["label"].astype(int)
    uniq = sorted(pd.Series(y).unique().tolist())
    if len(uniq) < 2:
        raise TrainSkippableError(f"학습 스킵: 클래스 부족 (unique={uniq})")

    # 각 클래스 최소 샘플 수 체크(너무 불균형하면 split/학습이 의미 없거나 깨짐)
    vc = y.value_counts()
    if (vc.min() < 3):
        raise TrainSkippableError(f"학습 스킵: 클래스 불균형(최소 class count={int(vc.min())}) {vc.to_dict()}")

    X = df[feature_cols]

    logger.info("[TRAIN] feature_uri=%s rows=%d classes=%s", feature_uri, len(df), uniq)

    try:
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)
    except Exception:
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    with mlflow.start_run(experiment_id=experiment_id) as run:
        run_id = run.info.run_id

        # ✅ tags에 lineage 남기기 (지금 tags null 문제 해결)
        if env:
            mlflow.set_tag("env", str(env))
        if fs_version:
            mlflow.set_tag("fs_version", str(fs_version))
        if schema_hash:
            mlflow.set_tag("schema_hash", str(schema_hash))
        mlflow.set_tag("feature_uri", str(feature_uri))
        if code_version:
            mlflow.set_tag("code_version", str(code_version))

        clf = LogisticRegression(C=C, max_iter=max_iter)
        clf.fit(X_train, y_train)

        y_pred = clf.predict(X_test)
        acc = accuracy_score(y_test, y_pred)
        f1m = f1_score(y_test, y_pred, average="macro")

        # params
        mlflow.log_param("C", C)
        mlflow.log_param("max_iter", max_iter)
        mlflow.log_param("feature_cols", ",".join(feature_cols))
        mlflow.log_param("train_rows", len(df))
        mlflow.log_param("train_classes", ",".join(map(str, uniq)))
        mlflow.log_param("n_features", X.shape[1])
        mlflow.log_param("n_classes", len(uniq))

        # metrics
        mlflow.log_metric("accuracy", float(acc))
        mlflow.log_metric("f1_macro", float(f1m))

        # model artifacts
        mlflow.sklearn.log_model(clf, "model")
        export_onnx_and_log_artifact(clf, n_features=X.shape[1])

        logger.info("[TRAIN] acc=%.4f f1_macro=%.4f run_id=%s", acc, f1m, run_id)
        return float(acc), run_id

