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

from skl2onnx import convert_sklearn
from skl2onnx.common.data_types import FloatTensorType

from airflow.utils.log.logging_mixin import LoggingMixin
from e2e.config import cfg
from e2e.utils import parse_s3_uri

log = LoggingMixin().log

class TrainSkippableError(RuntimeError):
    pass

def _read_parquet_s3(uri: str) -> pd.DataFrame:
    b, k = parse_s3_uri(uri)
    s3 = boto3.client("s3")
    obj = s3.get_object(Bucket=b, Key=k)
    return pd.read_parquet(io.BytesIO(obj["Body"].read()))

def _ensure_experiment(client: MlflowClient, name: str) -> str:
    exp = client.get_experiment_by_name(name)
    return exp.experiment_id if exp else client.create_experiment(name)

def _export_onnx_and_log(clf, n_features: int):
    initial_type = [("input", FloatTensorType([None, n_features]))]
    onnx_model = convert_sklearn(clf, initial_types=initial_type, options={id(clf): {"zipmap": False}})
    # io names
    in_name = onnx_model.graph.input[0].name
    out_names = [o.name for o in onnx_model.graph.output]
    mlflow.log_param("onnx_input_name", in_name)
    mlflow.log_param("onnx_output_names", ",".join(out_names))

    path = "/tmp/model.onnx"
    with open(path, "wb") as f:
        f.write(onnx_model.SerializeToString())
    mlflow.log_artifact(path, artifact_path="onnx")

def train_and_eval(**context):
    ti = context["ti"]
    tracking = cfg("MLFLOW_TRACKING_URI", required=True)
    mlflow.set_tracking_uri(tracking)
    client = MlflowClient(tracking_uri=tracking)

    exp_name = cfg("experiment_name", "e2e_full")
    exp_id = _ensure_experiment(client, exp_name)

    feature_uri = ti.xcom_pull(task_ids="store_features", key="feature_uri")
    if not feature_uri:
        raise ValueError("[TRAIN] feature_uri missing")

    df = _read_parquet_s3(feature_uri)

    required = ["f_total_events_7d","f_avg_session_sec_7d","f_last_event_age_sec"]
    for c in required:
        if c not in df.columns:
            raise TrainSkippableError(f"[TRAIN] missing feature column: {c}")

    X = df[required]
    # proxy label (rank qcut)
    x = df["f_total_events_7d"].astype(float)
    r = x.rank(method="first")
    try:
        y = pd.qcut(r, q=3, labels=[0,1,2]).astype(int)
    except Exception:
        y = pd.qcut(r, q=2, labels=[0,1]).astype(int)

    if len(pd.Series(y).unique()) < 2:
        raise TrainSkippableError("[TRAIN] not enough classes")

    C = float(cfg("logreg_C", "1.0"))
    max_iter = int(cfg("logreg_max_iter", "200"))

    with mlflow.start_run(experiment_id=exp_id) as run:
        run_id = run.info.run_id
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
        clf = LogisticRegression(C=C, max_iter=max_iter)
        clf.fit(X_train, y_train)

        pred = clf.predict(X_test)
        acc = accuracy_score(y_test, pred)

        mlflow.log_param("feature_uri", feature_uri)
        mlflow.log_param("n_features", X.shape[1])
        mlflow.log_param("n_classes", len(pd.Series(y).unique()))
        mlflow.log_metric("accuracy", acc)

        mlflow.sklearn.log_model(clf, "model")
        _export_onnx_and_log(clf, n_features=X.shape[1])

        alias = cfg("mlflow_alias", "A")

        ti.xcom_push(key="accuracy", value=float(acc))
        ti.xcom_push(key="run_id", value=run_id)
        ti.xcom_push(key="alias", value=alias)

        log.info("[TRAIN] acc=%.4f run_id=%s alias=%s", acc, run_id, alias)

