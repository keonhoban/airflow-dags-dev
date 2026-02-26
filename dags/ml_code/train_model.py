# dags/ml_code/train_model.py
from __future__ import annotations

import io
import os
from dataclasses import dataclass
from typing import Iterable, List, Optional, Tuple

import boto3
import pandas as pd
from botocore.config import Config

import mlflow
import mlflow.sklearn

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import train_test_split

from airflow.utils.log.logging_mixin import LoggingMixin

from skl2onnx import convert_sklearn
from skl2onnx.common.data_types import FloatTensorType

from ml_code.config import get_tracking_uri, get_experiment_name

logger = LoggingMixin().log


class TrainSkippableError(RuntimeError):
    """
    학습이 '불가능/의미없음'인 상태를 표현.
    - DAG에서 이 예외를 잡아 "skip 처리" / "fail 처리" 정책을 결정할 수 있습니다.
    """
    pass


# -----------------------
# Data spec (SSOT)
# -----------------------
FEATURE_COLS: List[str] = ["f_total_events_7d", "f_avg_session_sec_7d", "f_last_event_age_sec"]
LABEL_COL: str = "label"


@dataclass(frozen=True)
class TrainInput:
    feature_uri: str
    fs_version: Optional[str] = None
    schema_hash: Optional[str] = None
    env: Optional[str] = None
    code_version: Optional[str] = None


# -----------------------
# SSOT: S3 client defaults (operational safety)
# -----------------------
_S3_CFG = Config(
    retries={"max_attempts": 5, "mode": "standard"},
    connect_timeout=3,
    read_timeout=30,
)


# -----------------------
# MLflow SSOT
# -----------------------
def _ensure_tracking_uri() -> None:
    uri = get_tracking_uri()
    try:
        mlflow.set_tracking_uri(uri)
    except Exception as e:
        raise RuntimeError(f"[MLflow] set_tracking_uri failed uri={uri} err={e}") from e


# -----------------------
# S3 helpers
# -----------------------
def _parse_s3_uri(uri: str) -> Tuple[str, str]:
    if not uri.startswith("s3://"):
        raise ValueError(f"Invalid s3 uri: {uri}")
    x = uri[5:]
    bkt, key = x.split("/", 1)
    return bkt, key


def _read_parquet_from_s3(feature_uri: str, *, s3_client=None) -> pd.DataFrame:
    bkt, key = _parse_s3_uri(feature_uri)
    s3 = s3_client or boto3.client("s3", config=_S3_CFG)
    obj = s3.get_object(Bucket=bkt, Key=key)
    data = obj["Body"].read()
    return pd.read_parquet(io.BytesIO(data))


# -----------------------
# Validation helpers
# -----------------------
def _require_columns(df: pd.DataFrame, cols: Iterable[str]) -> None:
    missing = [c for c in cols if c not in df.columns]
    if missing:
        raise TrainSkippableError(f"학습 스킵: feature 컬럼 누락 {missing}")


def _validate_training_data(df: pd.DataFrame) -> pd.Series:
    """
    학습 가능 여부 검증.
    - 여기서는 '왜 불가능한지'를 결정하고,
    - 처리 정책(스킵/실패)은 상위(DAG)가 결정할 수 있게 예외로 표현합니다.
    """
    _require_columns(df, FEATURE_COLS)

    if LABEL_COL not in df.columns:
        raise TrainSkippableError("학습 스킵: label 컬럼이 없습니다 (DP build 단계에서 label 생성 필요)")

    if len(df) < 20:
        raise TrainSkippableError(f"학습 스킵: rows={len(df)} (데모 최소 20 권장, 운영은 200+ 권장)")

    y = df[LABEL_COL].astype(int)
    uniq = sorted(pd.Series(y).unique().tolist())
    if len(uniq) < 2:
        raise TrainSkippableError(f"학습 스킵: 클래스 부족 (unique={uniq})")

    vc = y.value_counts()
    if vc.min() < 3:
        raise TrainSkippableError(f"학습 스킵: 클래스 불균형(최소 class count={int(vc.min())}) {vc.to_dict()}")

    return y


# -----------------------
# ONNX gate
# -----------------------
def _validate_onnx_file(onnx_path: str, *, warn_bytes: int = 10_000) -> None:
    """
    ✅ ONNX 무결성 게이트 (downstream(Triton) 보호)

    - 파일 존재/0바이트 방지
    - onnx.load() 가능해야 함
    - onnx.checker.check_model() 통과해야 함
    - graph 최소 sanity (node/input/output)
    - 크기는 "실패 조건"이 아니라 "경고"로만 사용
    """
    import onnx

    if not os.path.exists(onnx_path):
        raise RuntimeError(f"[ONNX] missing file: {onnx_path}")

    sz = os.path.getsize(onnx_path)
    if sz <= 0:
        raise RuntimeError(f"[ONNX] empty file size={sz} path={onnx_path}")

    try:
        m = onnx.load(onnx_path)
    except Exception as e:
        raise RuntimeError(f"[ONNX] load failed: {e} path={onnx_path}") from e

    n_nodes = len(m.graph.node)
    n_ins = len(m.graph.input)
    n_outs = len(m.graph.output)
    if n_nodes < 1:
        raise RuntimeError(f"[ONNX] empty graph nodes={n_nodes} path={onnx_path}")
    if n_ins < 1 or n_outs < 1:
        raise RuntimeError(f"[ONNX] invalid io inputs={n_ins} outputs={n_outs} path={onnx_path}")

    try:
        onnx.checker.check_model(m)
    except Exception as e:
        raise RuntimeError(f"[ONNX] checker failed: {e} path={onnx_path}") from e

    ins = [i.name for i in m.graph.input]
    outs = [o.name for o in m.graph.output]

    if sz < warn_bytes:
        logger.warning(
            "[ONNX] validated OK but very small size=%s (<%s). model might be tiny (e.g., LogisticRegression). path=%s",
            sz,
            warn_bytes,
            onnx_path,
        )
    else:
        logger.info("[ONNX] validated OK size=%s nodes=%s inputs=%s outputs=%s", sz, n_nodes, ins, outs)


def export_onnx_and_log_artifact(clf, *, n_features: int, run_id: str) -> None:
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

        # 의미 힌트도 남겨서 downstream(Triton config) 설명 가능하게
        if out_names:
            prob = next((n for n in out_names if "prob" in n.lower()), out_names[0])
            label = next((n for n in out_names if "label" in n.lower()), out_names[-1])
            mlflow.log_param("onnx_output_prob_name", prob)
            mlflow.log_param("onnx_output_label_name", label)
    except Exception as e:
        logger.warning("[ONNX] failed to extract io names: %s", e)

    # ✅ 병렬/재시도 대비: run_id 기반 임시 파일
    onnx_path = f"/tmp/model_{run_id}.onnx"
    with open(onnx_path, "wb") as f:
        f.write(onnx_model.SerializeToString())

    # ✅ 핵심: log_artifact 전에 검증 통과 못 하면 FAIL
    _validate_onnx_file(onnx_path)

    mlflow.log_artifact(onnx_path, artifact_path="onnx")
    logger.info("[ONNX] logged: onnx/model.onnx (tmp=%s)", onnx_path)

    # best-effort cleanup
    try:
        os.remove(onnx_path)
    except Exception:
        pass


# -----------------------
# Main
# -----------------------
def train_model(
    C: float,
    max_iter: int,
    *,
    feature_uri: str,
    fs_version: Optional[str] = None,
    schema_hash: Optional[str] = None,
    env: Optional[str] = None,
    code_version: Optional[str] = None,
) -> Tuple[float, str]:
    """
    Returns: (accuracy, run_id)
    """
    if not feature_uri:
        raise ValueError("feature_uri is required")

    _ensure_tracking_uri()

    # ✅ experiment 관리 단순화
    mlflow.set_experiment(get_experiment_name())

    df = _read_parquet_from_s3(feature_uri)
    y = _validate_training_data(df)

    X = df[FEATURE_COLS]
    uniq = sorted(pd.Series(y).unique().tolist())

    logger.info(
        "[TRAIN] feature_uri=%s fs_version=%s schema_hash=%s rows=%d classes=%s",
        feature_uri,
        fs_version,
        schema_hash,
        len(df),
        uniq,
    )

    try:
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)
    except Exception:
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    with mlflow.start_run() as run:
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
        mlflow.log_param("feature_cols", ",".join(FEATURE_COLS))
        mlflow.log_param("train_rows", len(df))
        mlflow.log_param("train_classes", ",".join(map(str, uniq)))
        mlflow.log_param("n_features", X.shape[1])
        mlflow.log_param("n_classes", len(uniq))

        # metrics
        mlflow.log_metric("accuracy", float(acc))
        mlflow.log_metric("f1_macro", float(f1m))

        # artifacts
        mlflow.sklearn.log_model(clf, "model")

        # ✅ ONNX 검증 실패하면 run은 남더라도 artifact(onnx)는 안 올라가고 task는 fail
        export_onnx_and_log_artifact(clf, n_features=X.shape[1], run_id=str(run_id))

        logger.info("[TRAIN] acc=%.4f f1_macro=%.4f run_id=%s", acc, f1m, run_id)
        return float(acc), str(run_id)
