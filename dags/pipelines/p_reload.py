# dags/pipelines/p_reload.py
from __future__ import annotations

from typing import Any

from airflow.exceptions import AirflowSkipException

from mlops_lib.core.ids import (
    TRAIN_TASK_ID,
    DEPLOY_MATERIALIZE_TASK_ID,
    XCOM_ALIAS,
    # triton xcom keys (SSOT)
    K_DEPLOY_MODE as TRITON_XCOM_DEPLOY_MODE,
    K_DEPLOY_VERSION as TRITON_XCOM_DEPLOY_VERSION,
)

from mlops_lib.core.policy import Settings
from ml_code.trigger_reload import trigger_reload


def fastapi_reload_task(**context: Any) -> None:
    """
    Promotion path only.

    - promotion: deploy_version 기반 FastAPI metadata/cache refresh 수행
    - shadow: FastAPI pod-local override를 만들지 않기 위해 skip
    """
    s = Settings.load()
    ti = context["ti"]

    al = ti.xcom_pull(task_ids=TRAIN_TASK_ID, key=XCOM_ALIAS) or s.alias

    deploy_mode = ti.xcom_pull(
        task_ids=DEPLOY_MATERIALIZE_TASK_ID,
        key=TRITON_XCOM_DEPLOY_MODE,
    ) or "promote"

    if str(deploy_mode).strip().lower() == "shadow":
        raise AirflowSkipException("Shadow path: FastAPI reload skipped by policy")

    deploy_version = ti.xcom_pull(
        task_ids=DEPLOY_MATERIALIZE_TASK_ID,
        key=TRITON_XCOM_DEPLOY_VERSION,
    )

    if deploy_version is None:
        raise ValueError(
            "FastAPI promotion reload 불가: deploy_version XCom 누락 "
            "(deploy.materialize_repo 확인 필요)"
        )

    trigger_reload(str(al), deploy_version=int(deploy_version))
