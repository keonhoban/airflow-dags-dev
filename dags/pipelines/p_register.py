# dags/pipelines/p_register.py
from __future__ import annotations

from typing import Any, Optional

from mlops_lib.core.ids import (
    TRAIN_TASK_ID,
    REGISTER_TASK_ID,
    BRANCH_TASK_ID,
    XCOM_RUN_ID,
    XCOM_MODEL_NAME,
    XCOM_ALIAS,
    XCOM_VERSION,
    XCOM_SHADOW_REASON,
)

from mlops_lib.core.policy import Settings

# ✅ notify는 observability 쪽으로 이동 (core -> observability 방향 유지)
from mlops_lib.observability.notify import (
    notify_register_completed,
    notify_shadow_reason as notify_shadow_reason_policy,
)

from ml_code.register_model import register_model
from ml_code.sensor_model_ready import check_model_ready


def register_model_task(**context: Any) -> None:
    s = Settings.load()
    ti = context["ti"]

    run_id = ti.xcom_pull(task_ids=TRAIN_TASK_ID, key=XCOM_RUN_ID)
    mname = ti.xcom_pull(task_ids=TRAIN_TASK_ID, key=XCOM_MODEL_NAME) or s.model_name
    al = ti.xcom_pull(task_ids=TRAIN_TASK_ID, key=XCOM_ALIAS) or s.alias

    if not run_id:
        raise ValueError("Promotion 불가: run_id XCom 누락 (train_and_evaluate 확인 필요)")

    version = register_model(run_id=str(run_id), model_name=str(mname), mlflow_alias=str(al))
    ti.xcom_push(key=XCOM_VERSION, value=int(version))

    notify_register_completed(env=s.env, model=str(mname), alias=str(al), version=int(version))


def sensor_ready_func(**context: Any) -> bool:
    s = Settings.load()
    ti = context["ti"]

    mname = ti.xcom_pull(task_ids=TRAIN_TASK_ID, key=XCOM_MODEL_NAME) or s.model_name
    version = ti.xcom_pull(task_ids=REGISTER_TASK_ID, key=XCOM_VERSION)
    if not version:
        raise ValueError("Sensor 불가: version XCom 누락 (register_model_task 확인 필요)")

    return check_model_ready(model_name=str(mname), version=str(version))


def notify_shadow_reason(**context: Any) -> None:
    """
    Shadow로 빠진 이유를 SSOT(XCom)로 보고, 일관된 Slack 메시지를 남깁니다.
    """
    s = Settings.load()
    ti = context.get("ti")

    reason: Optional[str] = None
    if ti:
        reason = ti.xcom_pull(task_ids=BRANCH_TASK_ID, key=XCOM_SHADOW_REASON)

    notify_shadow_reason_policy(env=s.env, reason=reason)
