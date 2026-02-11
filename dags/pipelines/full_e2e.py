# dags/pipelines/full_e2e.py
from __future__ import annotations

from typing import Any, Optional, Tuple

from airflow.models import Variable
from airflow.utils.log.logging_mixin import LoggingMixin

from utils.slack_alerts import notify_info, notify_success, notify_skip

from mlops_lib.dp.tasks import (
    task_extract_raw_data,
    task_validate_data,
    task_build_features,
    task_store_features,
    task_summarize_run,
)

from ml_code.train_model import train_model, TrainSkippableError
from ml_code.register_model import register_model
from ml_code.sensor_model_ready import check_model_ready

from ml_code.triton_deploy import (
    snapshot_current as triton_snapshot_current_impl,
    materialize as triton_materialize_impl,
    triton_load as triton_load_impl,
    triton_ready as triton_ready_impl,
    triton_infer_smoke as triton_infer_smoke_impl,
    commit_current as triton_commit_current_impl,
    rollback_minimal as triton_rollback_minimal_impl,
)

from ml_code.trigger_reload import trigger_reload

log = LoggingMixin().log


def _v(key: str, default: Optional[str] = None) -> Optional[str]:
    try:
        return Variable.get(key)
    except Exception:
        return default


def get_env() -> str:
    return (_v("triton_env", "dev") or "dev").strip()


def accuracy_threshold() -> float:
    raw = _v("accuracy_threshold", "0.60") or "0.60"
    try:
        return float(raw)
    except Exception:
        return 0.60


def train_params() -> Tuple[float, int]:
    raw_c = _v("logreg_C", "1.0") or "1.0"
    raw_iter = _v("logreg_max_iter", "200") or "200"
    try:
        c = float(raw_c)
    except Exception:
        c = 1.0
    try:
        max_iter = int(raw_iter)
    except Exception:
        max_iter = 200
    return c, max_iter


def model_name() -> str:
    return (_v("triton_model_name", _v("model_name", "best_model")) or "best_model").strip()


def alias() -> str:
    return (_v("mlflow_alias", "A") or "A").strip()


# -----------------------
# Data pipeline wrappers
# -----------------------
def dp_extract(**context: Any):
    return task_extract_raw_data(**context)


def dp_validate(**context: Any):
    return task_validate_data(**context)


def dp_build(**context: Any):
    return task_build_features(**context)


def dp_store(**context: Any):
    return task_store_features(**context)


def dp_summary(**context: Any):
    return task_summarize_run(**context)


# -----------------------
# Train / Branch
# -----------------------
def train_and_evaluate(**context: Any):
    ti = context["ti"]

    feature_uri = ti.xcom_pull(key="fs_feature_uri", task_ids="store_features")
    fs_version = ti.xcom_pull(key="fs_version", task_ids="store_features")
    schema_hash = ti.xcom_pull(key="fs_schema_hash", task_ids="build_features")

    c, max_iter = train_params()
    mname = model_name()
    al = alias()
    env = get_env()

    ti.xcom_push(key="alias", value=al)
    ti.xcom_push(key="model_name", value=mname)

    try:
        acc, run_id = train_model(C=c, max_iter=max_iter, feature_uri=feature_uri, fs_version=fs_version, schema_hash=schema_hash)
    except TrainSkippableError as e:
        notify_skip("Train skipped", env=env, reason=str(e))
        ti.xcom_push(key="accuracy", value=None)
        ti.xcom_push(key="run_id", value=None)
        return

    ti.xcom_push(key="accuracy", value=float(acc))
    ti.xcom_push(key="run_id", value=run_id)

    notify_info("Train completed", env=env, accuracy=f"{acc:.4f}", alias=al, run_id=run_id, fs_version=fs_version, schema_hash=schema_hash)


def check_result(**context: Any) -> str:
    ti = context["ti"]
    env = get_env()
    th = accuracy_threshold()
    acc = ti.xcom_pull(task_ids="train_and_evaluate", key="accuracy")

    if acc is None:
        notify_info("Branch: shadow (train skipped)", env=env, threshold=str(th))
        return "shadow_start"

    try:
        acc_f = float(acc)
    except Exception:
        notify_info("Branch: shadow (accuracy invalid)", env=env, threshold=str(th))
        return "shadow_start"

    if acc_f >= th:
        notify_info("Branch: promotion", env=env, accuracy=f"{acc_f:.4f}", threshold=str(th))
        return "register_model_task"

    notify_info("Branch: shadow (below threshold)", env=env, accuracy=f"{acc_f:.4f}", threshold=str(th))
    return "shadow_start"


# -----------------------
# Register / Sensor
# -----------------------
def register_model_task(**context: Any):
    ti = context["ti"]
    run_id = ti.xcom_pull(task_ids="train_and_evaluate", key="run_id")
    mname = ti.xcom_pull(task_ids="train_and_evaluate", key="model_name") or model_name()
    al = ti.xcom_pull(task_ids="train_and_evaluate", key="alias") or alias()

    if not run_id:
        raise ValueError("Promotion 불가: run_id XCom 누락 (train_and_evaluate 확인 필요)")

    version = register_model(run_id=run_id, model_name=mname, mlflow_alias=al)
    ti.xcom_push(key="version", value=int(version))
    notify_success("MLflow register+alias completed", env=get_env(), model=mname, alias=al, version=str(version))


def sensor_ready_func(**context: Any) -> bool:
    ti = context["ti"]
    mname = ti.xcom_pull(task_ids="train_and_evaluate", key="model_name") or model_name()
    version = ti.xcom_pull(task_ids="register_model_task", key="version")
    if not version:
        raise ValueError("Sensor 불가: version XCom 누락 (register_model_task 확인 필요)")
    return check_model_ready(model_name=mname, version=str(version))


def notify_failure():
    notify_skip("Accuracy below threshold", env=get_env(), next_action="feature/label/model 개선 후 재시도")


# -----------------------
# Triton deploy wrappers
# -----------------------
def snapshot_current(**context: Any):
    return triton_snapshot_current_impl(ti=context["ti"])


def triton_materialize_task(**context: Any):
    ti = context["ti"]
    env = get_env()

    al = ti.xcom_pull(task_ids="train_and_evaluate", key="alias") or alias()
    run_id = ti.xcom_pull(task_ids="train_and_evaluate", key="run_id")
    version = ti.xcom_pull(task_ids="register_model_task", key="version")

    if version:
        notify_info("Triton deploy: promotion path (alias)", env=env, alias=al, version=str(version))
        return triton_materialize_impl(ti=ti, alias=al)

    if not run_id:
        raise ValueError("Shadow deploy 불가: run_id XCom 누락 (train_and_evaluate 확인 필요)")

    notify_info("Triton deploy: shadow path (run_id)", env=env, alias=al, run_id=run_id)
    return triton_materialize_impl(ti=ti, alias=al, run_id=run_id, shadow=True)


def triton_load(**context: Any):
    return triton_load_impl(ti=context["ti"])


def triton_ready(**context: Any):
    return triton_ready_impl(ti=context["ti"])


def triton_infer_smoke(**context: Any):
    return triton_infer_smoke_impl(ti=context["ti"])


def commit_current(**context: Any):
    return triton_commit_current_impl(ti=context["ti"])


def triton_rollback_task(**context: Any):
    return triton_rollback_minimal_impl(ti=context["ti"])


# -----------------------
# FastAPI reload (✅ deploy_version로 SSOT 동기화)
# -----------------------
def fastapi_reload_task(**context: Any):
    ti = context["ti"]
    env = get_env()
    al = ti.xcom_pull(task_ids="train_and_evaluate", key="alias") or alias()
    deploy_version = ti.xcom_pull(task_ids="materialize_repo", key="deploy_version")

    trigger_reload(al, deploy_version=int(deploy_version) if deploy_version is not None else None)
    notify_success("FastAPI reload completed", env=env, alias=al, deploy_version=str(deploy_version))

