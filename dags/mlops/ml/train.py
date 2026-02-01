from __future__ import annotations

import io
import boto3
import pandas as pd

import mlflow
import mlflow.sklearn
from mlflow.tracking import MlflowClient

from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

from airflow.utils.log.logging_mixin import LoggingMixin
from mlops.config import cfg

from skl2onnx import convert_sklearn
from skl2onnx.common.data_types import FloatTensorType

log = LoggingMixin().log


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


def _make_label(df: pd.DataFrame, q: int = 3) -> pd.Series:
    if "f_total_events_7d" not in df.columns:
        raise TrainSkippableError("label 생성 실패: f_total_events_7d 없음")

    x = df["f_total_events_7d"].astype(float)
    r = x.rank(method="first")

    if len(df) < 2:
        raise TrainSkippableError(f"label 생성 실패: rows={len(df)}")

    if q >= 3 and len(df) >= 3:
        try:
            return pd.qcut(r, q=3, labels=[0, 1, 2]).astype(int)
        except Exception:
            pass

    try:
        return pd.qcut(r, q=2, labels=[0, 1]).astype(int)
    except Exception as e:
        raise TrainSkippableError(f"label 생성 실패(qcut): {e}")


def export_onnx_and_log_artifact(clf, n_features: int):
    initial_type = [("input", FloatTensorType([None, n_features]))]
    onnx_model = convert_sklearn(clf, initial_types=initial_type, options={id(clf): {"zipmap": False}})

    try:
        in_name = onnx_model.graph.input[0].name
        out_names = [o.name for o in onnx_model.graph.output]
        mlflow.log_param("onnx_input_name", in_name)
        mlflow.log_param("onnx_output_names", ",".join(out_names))
        log.info("[ONNX] io names input=%s output=%s", in_name, out_names)
    except Exception as e:
        log.warning("[ONNX] failed to extract io names: %s", e)

    onnx_path = "/tmp/model.onnx"
    with open(onnx_path, "wb") as f:
        f.write(onnx_model.SerializeToString())

    mlflow.log_artifact(onnx_path, artifact_path="onnx")
    log.info("[ONNX] logged: onnx/model.onnx")


def task_train_and_evaluate(**context):
    ti = context["ti"]

    tracking_uri = cfg("MLFLOW_TRACKING_URI", required=True)
    mlflow.set_tracking_uri(tracking_uri)
    client = MlflowClient(tracking_uri=tracking_uri)

    exp_name = cfg("experiment_name", "train_and_register_model_exp")
    exp = client.get_experiment_by_name(exp_name)
    exp_id = exp.experiment_id if exp else client.create_experiment(exp_name)

    feature_uri = ti.xcom_pull(key="fs_feature_uri", task_ids="store_features")
    if not feature_uri:
        raise ValueError("feature_uri missing from store_features")

    df = _read_parquet_from_s3(feature_uri)

    required = ["f_total_events_7d", "f_avg_session_sec_7d", "f_last_event_age_sec"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise TrainSkippableError(f"학습 스킵: feature 컬럼 누락 {missing}")

    X = df[required]
    y = _make_label(df, q=3)

    uniq = sorted(pd.Series(y).unique().tolist())
    if len(uniq) < 2:
        raise TrainSkippableError(f"학습 스킵: 클래스 부족 (unique={uniq})")

    C = float(cfg("logreg_C", "1.0"))
    max_iter = int(cfg("logreg_max_iter", "200"))

    alias = cfg("mlflow_alias", "A")
    model_name = cfg("model_name", cfg("triton_model_name", "best_model"))
    schema_hash = ti.xcom_pull(key="fs_schema_hash", task_ids="build_features")
    fs_version = ti.xcom_pull(key="fs_version", task_ids="store_features")

    with mlflow.start_run(experiment_id=exp_id) as run:
        run_id = run.info.run_id

        clf = LogisticRegression(C=C, max_iter=max_iter)
        try:
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y)
        except Exception:
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

        clf.fit(X_train, y_train)
        acc = accuracy_score(y_test, clf.predict(X_test))

        mlflow.log_param("model_name", model_name)
        mlflow.log_param("alias", alias)
        mlflow.log_param("C", C)
        mlflow.log_param("max_iter", max_iter)
        mlflow.log_param("feature_uri", feature_uri)
        mlflow.log_param("train_rows", len(df))
        mlflow.log_param("train_classes", ",".join(map(str, uniq)))
        mlflow.log_param("n_features", X.shape[1])
        mlflow.log_param("n_classes", len(uniq))
        if fs_version:
            mlflow.log_param("fs_version", fs_version)
        if schema_hash:
            mlflow.log_param("schema_hash", schema_hash)

        mlflow.log_metric("accuracy", acc)

        mlflow.sklearn.log_model(clf, "model")
        export_onnx_and_log_artifact(clf, n_features=X.shape[1])

    ti.xcom_push(key="run_id", value=run_id)
    ti.xcom_push(key="accuracy", value=float(acc))
    ti.xcom_push(key="alias", value=alias)
    ti.xcom_push(key="model_name", value=model_name)

    log.info("[TRAIN] acc=%.4f run_id=%s model=%s alias=@%s", acc, run_id, model_name, alias)

