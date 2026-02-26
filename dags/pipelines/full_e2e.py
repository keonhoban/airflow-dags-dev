# dags/pipelines/full_e2e.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Optional

from airflow.models import Variable
from airflow.utils.log.logging_mixin import LoggingMixin

from utils.slack_alerts import notify_info, notify_success, notify_skip

from mlops_lib.dp.tasks import (
    task_extract_raw_data as _dp_extract,
    task_validate_data as _dp_validate,
    task_build_features as _dp_build,
    task_store_features as _dp_store,
    task_summarize_run as _dp_summary,
)

from ml_code.train_model import train_model, TrainSkippableError
from ml_code.register_model import register_model
from ml_code.sensor_model_ready import check_model_ready

from ml_code.triton_deploy import (
    snapshot_current as triton_snapshot_current,
    materialize as triton_materialize,
    triton_load as triton_load,
    triton_ready as triton_ready,
    triton_infer_smoke as triton_infer_smoke,
    commit_current as triton_commit_current,
    rollback_minimal as triton_rollback_minimal,
)

from ml_code.trigger_reload import trigger_reload

log = LoggingMixin().log


# -----------------------
# Settings (SSOT)
# -----------------------
def _v(key: str, default: Optional[str] = None) -> Optional[str]:
    try:
        return Variable.get(key)
    except Exception:
        return default


def _to_float(raw: Optional[str], default: float) -> float:
    try:
        return float(str(raw))
    except Exception:
        return default


def _to_int(raw: Optional[str], default: int) -> int:
    try:
        return int(str(raw))
    except Exception:
        return default


@dataclass(frozen=True)
class Settings:
    env: str
    accuracy_threshold: float
    logreg_c: float
    logreg_max_iter: int
    model_name: str
    alias: str

    @classmethod
    def load(cls) -> "Settings":
        env = (_v("triton_env", "dev") or "dev").strip()
        th = _to_float(_v("accuracy_threshold", "0.60"), 0.60)

        c = _to_float(_v("logreg_C", "1.0"), 1.0)
        it = _to_int(_v("logreg_max_iter", "200"), 200)

        model_name = (_v("triton_model_name", _v("model_name", "best_model")) or "best_model").strip()
        alias = (_v("mlflow_alias", "A") or "A").strip()

        return cls(
            env=env,
            accuracy_threshold=th,
            logreg_c=c,
            logreg_max_iter=it,
            model_name=model_name,
            alias=alias,
        )


# -----------------------
# Data pipeline wrappers (TaskGroup에서 호출)
# -----------------------
def dp_extract(**context: Any):
    return _dp_extract(**context)


def dp_validate(**context: Any):
    return _dp_validate(**context)


def dp_build(**context: Any):
    return _dp_build(**context)


def dp_store(**context: Any):
    return _dp_store(**context)


def dp_summary(**context: Any):
    return _dp_summary(**context)


# -----------------------
# Train / Branch
# -----------------------
def train_and_evaluate(**context: Any) -> None:
    s = Settings.load()
    ti = context["ti"]

    feature_uri = ti.xcom_pull(key="fs_feature_uri", task_ids="store_features")
    fs_version = ti.xcom_pull(key="fs_version", task_ids="store_features")
    schema_hash = ti.xcom_pull(key="fs_schema_hash", task_ids="build_features")

    ti.xcom_push(key="alias", value=s.alias)
    ti.xcom_push(key="model_name", value=s.model_name)

    try:
        acc, run_id = train_model(
            C=s.logreg_c,
            max_iter=s.logreg_max_iter,
            feature_uri=feature_uri,
            fs_version=fs_version,
            schema_hash=schema_hash,
        )
    except TrainSkippableError as e:
        notify_skip("Train skipped", env=s.env, reason=str(e))
        ti.xcom_push(key="accuracy", value=None)
        ti.xcom_push(key="run_id", value=None)
        return

    ti.xcom_push(key="accuracy", value=float(acc))
    ti.xcom_push(key="run_id", value=run_id)

    notify_info(
        "Train completed",
        env=s.env,
        accuracy=f"{float(acc):.4f}",
        alias=s.alias,
        run_id=run_id,
        fs_version=fs_version,
        schema_hash=schema_hash,
    )


def check_result(**context: Any) -> str:
    s = Settings.load()
    ti = context["ti"]

    acc = ti.xcom_pull(task_ids="train_and_evaluate", key="accuracy")

    if acc is None:
        notify_info("Branch: shadow (train skipped)", env=s.env, threshold=str(s.accuracy_threshold))
        return "shadow_start"

    try:
        acc_f = float(acc)
    except Exception:
        notify_info("Branch: shadow (accuracy invalid)", env=s.env, threshold=str(s.accuracy_threshold))
        return "shadow_start"

    if acc_f >= s.accuracy_threshold:
        notify_info("Branch: promotion", env=s.env, accuracy=f"{acc_f:.4f}", threshold=str(s.accuracy_threshold))
        return "register_model_task"

    notify_info("Branch: shadow (below threshold)", env=s.env, accuracy=f"{acc_f:.4f}", threshold=str(s.accuracy_threshold))
    return "shadow_start"


# -----------------------
# Register / Sensor
# -----------------------
def register_model_task(**context: Any) -> None:
    s = Settings.load()
    ti = context["ti"]

    run_id = ti.xcom_pull(task_ids="train_and_evaluate", key="run_id")
    mname = ti.xcom_pull(task_ids="train_and_evaluate", key="model_name") or s.model_name
    al = ti.xcom_pull(task_ids="train_and_evaluate", key="alias") or s.alias

    if not run_id:
        raise ValueError("Promotion 불가: run_id XCom 누락 (train_and_evaluate 확인 필요)")

    version = register_model(run_id=str(run_id), model_name=str(mname), mlflow_alias=str(al))
    ti.xcom_push(key="version", value=int(version))

    notify_success("MLflow register+alias completed", env=s.env, model=str(mname), alias=str(al), version=str(version))


def sensor_ready_func(**context: Any) -> bool:
    s = Settings.load()
    ti = context["ti"]

    mname = ti.xcom_pull(task_ids="train_and_evaluate", key="model_name") or s.model_name
    version = ti.xcom_pull(task_ids="register_model_task", key="version")
    if not version:
        raise ValueError("Sensor 불가: version XCom 누락 (register_model_task 확인 필요)")
    return check_model_ready(model_name=str(mname), version=str(version))


def notify_failure() -> None:
    s = Settings.load()
    notify_skip("Accuracy below threshold", env=s.env, next_action="feature/label/model 개선 후 재시도")


# -----------------------
# Triton deploy wrappers
# -----------------------
def snapshot_current(**context: Any) -> None:
    return triton_snapshot_current(ti=context["ti"])


def triton_materialize_task(**context: Any) -> None:
    s = Settings.load()
    ti = context["ti"]

    al = ti.xcom_pull(task_ids="train_and_evaluate", key="alias") or s.alias
    run_id = ti.xcom_pull(task_ids="train_and_evaluate", key="run_id")
    version = ti.xcom_pull(task_ids="register_model_task", key="version")

    if version:
        notify_info(
            "Triton deploy: promotion path (alias->MLflow version)",
            env=s.env,
            alias=str(al),
            version=str(version),
        )
        return triton_materialize(ti=ti, alias=str(al), shadow=False)

    if not run_id:
        raise ValueError("Shadow deploy 불가: run_id XCom 누락 (train_and_evaluate 확인 필요)")

    notify_info(
        "Triton deploy: shadow path (run_id->timestamp)",
        env=s.env,
        alias=str(al),
        run_id=str(run_id),
    )
    return triton_materialize(ti=ti, alias=str(al), run_id=str(run_id), shadow=True)


def triton_load_task(**context: Any) -> None:
    return triton_load(ti=context["ti"])


def triton_ready_task(**context: Any) -> None:
    return triton_ready(ti=context["ti"])


def triton_infer_smoke_task(**context: Any) -> None:
    return triton_infer_smoke(ti=context["ti"])


def commit_current(**context: Any) -> None:
    return triton_commit_current(ti=context["ti"])


def triton_rollback_task(**context: Any) -> None:
    return triton_rollback_minimal(ti=context["ti"])


# -----------------------
# FastAPI reload
# -----------------------
def fastapi_reload_task(**context: Any) -> None:
    s = Settings.load()
    ti = context["ti"]

    al = ti.xcom_pull(task_ids="train_and_evaluate", key="alias") or s.alias

    deploy_mode = ti.xcom_pull(task_ids="materialize_repo", key="deploy_mode") or "promote"
    deploy_version = ti.xcom_pull(task_ids="materialize_repo", key="deploy_version")
    run_id = ti.xcom_pull(task_ids="materialize_repo", key="run_id")

    if str(deploy_mode) == "shadow":
        if not run_id:
            raise ValueError("FastAPI shadow reload 불가: run_id XCom 누락 (materialize_repo 확인 필요)")
        trigger_reload(str(al), run_id=str(run_id))
        notify_success(
            "FastAPI reload completed (shadow/run_id)",
            env=s.env,
            alias=str(al),
            run_id=str(run_id),
        )
        return

    if deploy_version is None:
        raise ValueError("FastAPI promotion reload 불가: deploy_version XCom 누락 (materialize_repo 확인 필요)")

    trigger_reload(str(al), deploy_version=int(deploy_version))
    notify_success(
        "FastAPI reload completed (promotion/deploy_version)",
        env=s.env,
        alias=str(al),
        deploy_version=str(deploy_version),
    )
