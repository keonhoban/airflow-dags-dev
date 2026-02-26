# dags/ml_code/train_model.py
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


def _validate_onnx_file(onnx_path: str, *, min_bytes: int = 10_000) -> None:
    """
    ✅ ONNX 무결성 게이트 (downstream(Triton) 보호)
    - 파일 존재
    - 최소 크기 (너무 작은 ONNX는 운영에서 위험 신호)
    - onnx.load + onnx.checker
    """
    import onnx

    if not os.path.exists(onnx_path):
        raise RuntimeError(f"[ONNX] missing file: {onnx_path}")

    sz = os.path.getsize(onnx_path)
    # 데모라면 낮춰도 되지만, 지금처럼 513 bytes 같은 케이스를 막기 위해 기본은 10KB
    if sz < min_bytes:
        raise RuntimeError(f"[ONNX] too small size={sz} (<{min_bytes}) path={onnx_path}")

    m = onnx.load(onnx_path)
    # graph 최소 sanity
    if len(m.graph.node) < 1:
        raise RuntimeError(f"[ONNX] empty graph nodes={len(m.graph.node)} path={onnx_path}")

    try:
        onnx.checker.check_model(m)
    except Exception as e:
        raise RuntimeError(f"[ONNX] checker failed: {e}") from e

    ins = [i.name for i in m.graph.input]
    outs = [o.name for o in m.graph.output]
    logger.info("[ONNX] validated OK size=%s nodes=%s inputs=%s outputs=%s", sz, len(m.graph.node), ins, outs)


def export_onnx_and_log_artifact(clf, n_features: int):
    initial_type = [("input", FloatTensorType([None, n_features]))]
    onnx_model = convert_sklearn(
        clf,
        initial_types=initial_type,
        options={id(clf): {"zipmap": False}},
    )

    # io names -> params
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

    # ✅ 핵심: log_artifact 전에 검증 통과 못 하면 FAIL
    _validate_onnx_file(onnx_path)

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

    if "label" not in df.columns:
        raise TrainSkippableError("학습 스킵: label 컬럼이 없습니다 (DP build 단계에서 label 생성 필요)")

    if len(df) < 20:
        raise TrainSkippableError(f"학습 스킵: rows={len(df)} (데모 최소 20 권장, 운영은 200+ 권장)")

    y = df["label"].astype(int)
    uniq = sorted(pd.Series(y).unique().tolist())
    if len(uniq) < 2:
        raise TrainSkippableError(f"학습 스킵: 클래스 부족 (unique={uniq})")

    vc = y.value_counts()
    if vc.min() < 3:
        raise TrainSkippableError(f"학습 스킵: 클래스 불균형(최소 class count={int(vc.min())}) {vc.to_dict()}")

    X = df[feature_cols]
    logger.info("[TRAIN] feature_uri=%s rows=%d classes=%s", feature_uri, len(df), uniq)

    try:
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)
    except Exception:
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    with mlflow.start_run(experiment_id=experiment_id) as run:
        run_id = run.info.run_id

        # tags lineage
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

        # artifacts
        mlflow.sklearn.log_model(clf, "model")

        # ✅ 여기서 ONNX 검증 실패하면 run은 남더라도 artifact(onnx)는 안 올라가고 task는 fail
        export_onnx_and_log_artifact(clf, n_features=X.shape[1])

        logger.info("[TRAIN] acc=%.4f f1_macro=%.4f run_id=%s", acc, f1m, run_id)
        return float(acc), run_id
