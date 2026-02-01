from __future__ import annotations
import os
from airflow.sdk import Variable

from utils.config import (
    env, accuracy_threshold, model_name, alias, feature_s3_prefix,
    mlflow_tracking_uri,
)
from utils.slack import info, skip, success

from dp.features import build_features_with_label
from dp.io_s3 import store_parquet_version_and_latest
from dp.quality import assert_trainable, DataNotTrainable

from ml.train import train_and_log_model, TrainSkippableError
from ml.register import register_and_set_alias
from ml.sensor import wait_until_ready

from deploy.triton import (
    snapshot_current as _snapshot_current,
    materialize_from_shared_repo,
    triton_load_ready_smoke,
    commit_current as _commit_current,
    rollback_minimal as _rollback_minimal,
)
from deploy.fastapi import reload_and_notify

import pandas as pd

# -----------------------
# DP
# -----------------------
def dp_build_and_store(**context):
    """
    실무형: raw를 어디서 가져오든, 최종 산출물은 'features + label' parquet 하나로 고정
    """
    ti = context["ti"]

    # (중요) 여기서는 예시로 “S3의 raw parquet/csv”에서 읽는다고 가정.
    # 건호님은 이미 feature-store-lite를 갖고 계시니,
    # 실제 raw 적재/스냅샷 task는 기존 dp.tasks로 붙여도 됩니다.
    raw_uri = Variable.get("raw_uri", default_var="")
    if not raw_uri:
        raise RuntimeError("raw_uri Variable is required (s3://.../raw.csv or parquet)")

    raw = pd.read_csv(raw_uri) if raw_uri.endswith(".csv") else pd.read_parquet(raw_uri)

    feats, meta = build_features_with_label(raw)
    assert_trainable(feats, label_col="label")

    out = store_parquet_version_and_latest(feats, prefix=feature_s3_prefix(), name="features.parquet")

    # schema_hash는 “피처 스키마 + 라벨 정의 + raw 버전”이 바뀌면 변해야 합니다.
    # 여기선 간단히 columns 기반으로만 예시.
    schema_hash = str(pd.util.hash_pandas_object(pd.Index(feats.columns), index=False).sum())

    ti.xcom_push(key="fs_version", value=out["fs_version"])
    ti.xcom_push(key="fs_feature_uri", value=out["fs_feature_uri"])
    ti.xcom_push(key="fs_schema_hash", value=schema_hash)

    info("DP stored features", env=env(), fs_version=out["fs_version"], feature_uri=out["fs_feature_uri"], schema_hash=schema_hash, rows=str(feats.shape[0]))


# -----------------------
# Train
# -----------------------
def train_and_log(**context):
    ti = context["ti"]

    os.environ["MLFLOW_TRACKING_URI"] = mlflow_tracking_uri()

    feature_uri = ti.xcom_pull(task_ids="dp_build_and_store", key="fs_feature_uri")
    fs_version = ti.xcom_pull(task_ids="dp_build_and_store", key="fs_version")
    schema_hash = ti.xcom_pull(task_ids="dp_build_and_store", key="fs_schema_hash")

    try:
        acc, run_id = train_and_log_model(
            feature_uri=feature_uri,
            fs_version=fs_version,
            schema_hash=schema_hash,
            env=env(),
            C=float(Variable.get("logreg_C", default_var="1.0")),
            max_iter=int(Variable.get("logreg_max_iter", default_var="200")),
        )
    except (TrainSkippableError, DataNotTrainable) as e:
        skip("Train skipped", env=env(), reason=str(e))
        ti.xcom_push(key="accuracy", value=None)
        ti.xcom_push(key="run_id", value=None)
        return

    ti.xcom_push(key="accuracy", value=float(acc))
    ti.xcom_push(key="run_id", value=run_id)
    ti.xcom_push(key="model_name", value=model_name())
    ti.xcom_push(key="alias", value=alias())

    info("Train completed", env=env(), accuracy=f"{acc:.4f}", alias=alias(), run_id=run_id, fs_version=fs_version, schema_hash=schema_hash)


# -----------------------
# Branch
# -----------------------
def branch_on_accuracy(**context):
    ti = context["ti"]
    acc = ti.xcom_pull(task_ids="train_and_log", key="accuracy")
    th = accuracy_threshold()

    if acc is None:
        info("Branch: shadow (train skipped)", env=env(), threshold=str(th))
        return "shadow_start"

    if float(acc) >= th:
        info("Branch: promotion", env=env(), accuracy=f"{float(acc):.4f}", threshold=str(th))
        return "register_and_alias"

    info("Branch: shadow (below threshold)", env=env(), accuracy=f"{float(acc):.4f}", threshold=str(th))
    return "shadow_start"


def notify_shadow_skip(**context):
    skip("Accuracy below threshold", env=env(), next_action="raw/label/feature 개선 후 재시도")


# -----------------------
# Register + Sensor
# -----------------------
def register_and_alias(**context):
    ti = context["ti"]
    os.environ["MLFLOW_TRACKING_URI"] = mlflow_tracking_uri()

    run_id = ti.xcom_pull(task_ids="train_and_log", key="run_id")
    if not run_id:
        raise RuntimeError("run_id missing")

    version = register_and_set_alias(run_id=run_id, model_name=model_name(), alias=alias())
    ti.xcom_push(key="version", value=int(version))

    success("MLflow register+alias completed", env=env(), model=model_name(), alias=alias(), version=str(version))


def wait_model_ready(**context):
    ti = context["ti"]
    os.environ["MLFLOW_TRACKING_URI"] = mlflow_tracking_uri()

    version = ti.xcom_pull(task_ids="register_and_alias", key="version")
    if not version:
        raise RuntimeError("version missing")
    wait_until_ready(model_name=model_name(), version=int(version), timeout_sec=60)


# -----------------------
# Triton + FastAPI
# -----------------------
def triton_snapshot_current(**context):
    ti = context["ti"]
    _snapshot_current(ti)

def triton_materialize_and_smoke(**context):
    ti = context["ti"]
    # (중요) 지금 건호님 /models 구조상 version 디렉터리가 이미 있으니,
    # 여기서는 “등록된 version을 target_version으로 사용”하는 정책을 추천합니다.
    version = ti.xcom_pull(task_ids="register_and_alias", key="version")
    if not version:
        raise RuntimeError("version missing")

    materialize_from_shared_repo(ti, target_version=str(version))
    triton_load_ready_smoke(model_name())

def triton_commit_current(**context):
    ti = context["ti"]
    _commit_current(ti)

def triton_rollback_minimal(**context):
    ti = context["ti"]
    _rollback_minimal(ti)

def fastapi_reload(**context):
    ti = context["ti"]
    reload_and_notify(alias(), env())

