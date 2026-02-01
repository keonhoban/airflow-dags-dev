# dags/pipelines/full_e2e.py
from __future__ import annotations

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
    snapshot_current,
    materialize,
    triton_load,
    triton_ready,
    triton_infer_smoke,
    commit_current,
    rollback_minimal,
)

from ml_code.trigger_reload import trigger_reload

log = LoggingMixin().log


# -----------------------
# small helpers
# -----------------------
def _v(key: str, default=None) -> str:
    try:
        return Variable.get(key)
    except Exception:
        return default


def get_env() -> str:
    return _v("triton_env", "dev") or "dev"


def _accuracy_threshold() -> float:
    x = _v("accuracy_threshold", "0.60")
    try:
        return float(x)
    except Exception:
        return 0.60


def _train_params() -> tuple[float, int]:
    C = float(_v("logreg_C", "1.0"))
    max_iter = int(_v("logreg_max_iter", "200"))
    return C, max_iter


def _model_name() -> str:
    # MLflow registered model name (and Triton model directory name)
    return _v("triton_model_name", _v("model_name", "best_model")) or "best_model"


def _alias() -> str:
    return _v("mlflow_alias", "A") or "A"


# -----------------------
# DP tasks (thin wrappers)
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
# Train / Branch
# -----------------------
def train_and_evaluate(**context):
    ti = context["ti"]

    feature_uri = ti.xcom_pull(key="fs_feature_uri", task_ids="store_features")
    fs_version = ti.xcom_pull(key="fs_version", task_ids="store_features")
    schema_hash = ti.xcom_pull(key="fs_schema_hash", task_ids="build_features")

    C, max_iter = _train_params()
    model_name = _model_name()
    alias = _alias()
    env = get_env()

    ti.xcom_push(key="alias", value=alias)
    ti.xcom_push(key="model_name", value=model_name)

    try:
        acc, run_id = train_model(
            C=C,
            max_iter=max_iter,
            feature_uri=feature_uri,
            fs_version=fs_version,
            schema_hash=schema_hash,
        )
    except TrainSkippableError as e:
        # 학습 불가 상황은 "shadow only"로 보내기 위해 run_id를 비워둠
        notify_skip("Train skipped", env=env, reason=str(e))
        ti.xcom_push(key="accuracy", value=None)
        ti.xcom_push(key="run_id", value=None)
        return

    ti.xcom_push(key="accuracy", value=float(acc))
    ti.xcom_push(key="run_id", value=run_id)

    notify_info(
        "Train completed",
        env=env,
        accuracy=f"{acc:.4f}",
        alias=alias,
        run_id=run_id,
        fs_version=fs_version,
        schema_hash=schema_hash,
    )


def check_result(**context):
    ti = context["ti"]
    acc = ti.xcom_pull(task_ids="train_and_evaluate", key="accuracy")
    env = get_env()
    th = _accuracy_threshold()

    # train skip이면 shadow로
    if acc is None:
        notify_info("Branch: shadow (train skipped)", env=env, threshold=str(th))
        return "shadow_start"

    if float(acc) >= th:
        notify_info("Branch: promotion", env=env, accuracy=f"{acc:.4f}", threshold=str(th))
        return "register_model_task"

    notify_info("Branch: shadow (below threshold)", env=env, accuracy=f"{acc:.4f}", threshold=str(th))
    return "shadow_start"


# -----------------------
# Register / Sensor
# -----------------------
def register_model_task(**context):
    ti = context["ti"]
    run_id = ti.xcom_pull(task_ids="train_and_evaluate", key="run_id")
    model_name = ti.xcom_pull(task_ids="train_and_evaluate", key="model_name") or _model_name()
    alias = ti.xcom_pull(task_ids="train_and_evaluate", key="alias") or _alias()

    if not run_id:
        raise ValueError("Promotion 불가: run_id XCom 누락")

    version = register_model(run_id=run_id, model_name=model_name, mlflow_alias=alias)
    ti.xcom_push(key="version", value=int(version))

    notify_success("MLflow register+alias completed", env=get_env(), model=model_name, alias=alias, version=str(version))


def sensor_ready_func(**context):
    ti = context["ti"]
    model_name = ti.xcom_pull(task_ids="train_and_evaluate", key="model_name") or _model_name()
    version = ti.xcom_pull(task_ids="register_model_task", key="version")
    if not version:
        raise ValueError("Sensor 불가: version XCom 누락")
    return check_model_ready(model_name=model_name, version=str(version))


def notify_failure():
    env = get_env()
    notify_skip("Accuracy below threshold", env=env, next_action="feature/label/model 개선 후 재시도")


# -----------------------
# Triton deploy helpers
# -----------------------
def snapshot_current(**context):
    ti = context["ti"]
    return snapshot_current(ti=ti)


def triton_materialize_task(**context):
    ti = context["ti"]
    alias = ti.xcom_pull(task_ids="train_and_evaluate", key="alias") or _alias()
    env = get_env()

    version = ti.xcom_pull(task_ids="register_model_task", key="version")
    run_id = ti.xcom_pull(task_ids="train_and_evaluate", key="run_id")

    if version:
        notify_info("Triton deploy: promotion path (alias)", env=env, alias=alias, version=str(version))
        materialize(ti=ti, alias=alias)
        return

    if not run_id:
        # train skipped인데 shadow deploy까지 하고 싶으면 여기서 정책 결정 가능
        raise ValueError("Shadow deploy 불가: run_id XCom 누락 (train_and_evaluate 확인 필요)")

    notify_info("Triton deploy: shadow path (run_id)", env=env, alias=alias, run_id=run_id)
    materialize(ti=ti, alias=alias, run_id=run_id, shadow=True)


def triton_rollback_task(**context):
    ti = context["ti"]
    rollback_minimal(ti=ti)


def fastapi_reload_task(**context):
    ti = context["ti"]
    alias = ti.xcom_pull(task_ids="train_and_evaluate", key="alias") or _alias()
    env = get_env()
    trigger_reload(alias)
    notify_success("FastAPI reload completed", env=env, alias=alias)

