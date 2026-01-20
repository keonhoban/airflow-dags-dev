from __future__ import annotations

from airflow.sdk import Variable
from airflow.utils.log.logging_mixin import LoggingMixin
from airflow.exceptions import AirflowSkipException

from mlflow.tracking import MlflowClient

from utils.slack_alerts import send_slack_alert

# DP tasks
from mlops_lib.dp.tasks import (
    task_extract_raw_data,
    task_validate_data,
    task_build_features,
    task_store_features,
    task_summarize_run,
)

# ML
from ml_code.train_model import train_model
from ml_code.register_model import register_model
from ml_code.rollback_model import rollback_model
from ml_code.sensor_model_ready import check_model_ready

# Triton
from ml_code.triton_deploy import (
    snapshot_current,
    materialize,
    triton_load,
    triton_ready,
    triton_infer_smoke,
    commit_current,
    rollback_minimal,
)

# FastAPI reload
from ml_code.trigger_reload import trigger_reload


log = LoggingMixin().log


# -----------------------
# Helpers
# -----------------------
def get_param(key, default, cast_func, validate_func=None):
    try:
        v = cast_func(Variable.get(key, default=str(default)))
        if validate_func and not validate_func(v):
            raise ValueError("Validation failed")
        return v
    except Exception as e:
        send_slack_alert(f"[Param] {key} 로딩 실패: {e} → 기본값 {default} 사용")
        return default


def get_version_by_alias(model_name, alias):
    try:
        return MlflowClient().get_model_version_by_alias(model_name, alias).version
    except Exception:
        return None


# -----------------------
# DP wrappers (keep thin)
# -----------------------
def dp_extract(**context):
    return task_extract_raw_data(**context)

def dp_validate(**context):
    return task_validate_data(**context)

def dp_build(**context):
    return task_build_features(**context)

def dp_store(**context):
    return task_store_features(**context)

def dp_summary(**context):
    return task_summarize_run(**context)


# -----------------------
# Train/Register
# -----------------------
def train_and_evaluate(ti, **_):
    C = get_param("logreg_C", 1.0, float, lambda x: 0.001 <= x <= 10.0)
    max_iter = get_param("logreg_max_iter", 200, int, lambda x: x > 50)
    threshold = get_param("accuracy_threshold", 0.9, float, lambda x: 0.5 <= x <= 0.99)

    model_name = Variable.get("model_name")
    alias = Variable.get("mlflow_alias")
    if not (model_name and alias):
        raise ValueError("필수 Variable 누락: model_name 또는 mlflow_alias")

    feature_uri = ti.xcom_pull(task_ids="store_features", key="fs_feature_uri")
    fs_version = ti.xcom_pull(task_ids="store_features", key="fs_version")
    schema_hash = ti.xcom_pull(task_ids="build_features", key="fs_schema_hash")

    if not feature_uri:
        raise ValueError("feature_uri 없음 → store_features 결과 확인 필요")

    acc, run_id = train_model(
        C=C,
        max_iter=max_iter,
        feature_uri=feature_uri,
        fs_version=fs_version,
        schema_hash=schema_hash,
    )

    ti.xcom_push(key="run_id", value=run_id)
    ti.xcom_push(key="model_name", value=model_name)
    ti.xcom_push(key="alias", value=alias)
    ti.xcom_push(key="acc", value=acc)
    ti.xcom_push(key="threshold", value=threshold)


def check_result(ti, **_):
    acc = ti.xcom_pull(task_ids="train_and_evaluate", key="acc")
    threshold = ti.xcom_pull(task_ids="train_and_evaluate", key="threshold")

    if acc is None or threshold is None:
        send_slack_alert("❌ check_result → XCom 누락")
        raise AirflowSkipException()

    return "register_model_task" if acc >= threshold else "notify_failure"


def register_model_task(ti, **_):
    run_id = ti.xcom_pull(task_ids="train_and_evaluate", key="run_id")
    model_name = ti.xcom_pull(task_ids="train_and_evaluate", key="model_name")
    alias = ti.xcom_pull(task_ids="train_and_evaluate", key="alias")

    prev_version = get_version_by_alias(model_name, alias)

    try:
        version = register_model(run_id, model_name, alias)
        ti.xcom_push(key="version", value=version)
        send_slack_alert(f"✅ 모델 등록 완료: {model_name} v{version} → @{alias}")
    except Exception as e:
        msg = f"❌ 모델 등록 실패: {e}"
        if prev_version:
            rollback_model(model_name, prev_version, alias)
            msg += f" → 롤백 완료: v{prev_version}"
        else:
            msg += " → 롤백 생략"
        send_slack_alert(msg)
        raise


def sensor_ready_func(ti, **_):
    model_name = ti.xcom_pull(task_ids="train_and_evaluate", key="model_name")
    version = ti.xcom_pull(task_ids="register_model_task", key="version")
    return check_model_ready(model_name, version)


# -----------------------
# Triton deploy
# -----------------------
def triton_materialize_task(ti, **_):
    alias = ti.xcom_pull(task_ids="train_and_evaluate", key="alias")
    materialize(ti=ti, alias=alias)


def triton_rollback_task(ti, **_):
    rollback_minimal(ti=ti)


# -----------------------
# FastAPI reload (optional)
# -----------------------
def fastapi_reload_task(ti, **_):
    alias = ti.xcom_pull(task_ids="train_and_evaluate", key="alias")
    trigger_reload(alias)
    send_slack_alert(f"🔁 FastAPI reload 완료: @{alias}")


def notify_failure():
    send_slack_alert("⚠️ 기준 미달 → 등록/배포 생략")
