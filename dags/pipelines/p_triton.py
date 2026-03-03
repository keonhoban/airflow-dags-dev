# dags/pipelines/p_triton.py
from __future__ import annotations

from typing import Any

from mlops_lib.core.ids import (
    TRAIN_TASK_ID,
    REGISTER_TASK_ID,
    DEPLOY_MATERIALIZE_TASK_ID,
    XCOM_ALIAS,
    XCOM_RUN_ID,
    XCOM_VERSION,
)

from mlops_lib.core.policy import Settings

# ✅ Triton은 "tasks" 계층만 바라보게 고정
from ml_code.triton_tasks import (
    snapshot_current as triton_snapshot_current,
    materialize as triton_materialize,
    triton_load_task as triton_load_task_impl,
    triton_ready_task as triton_ready_task_impl,
    triton_infer_smoke_task as triton_infer_smoke_task_impl,
    commit_current as triton_commit_current,
    rollback_minimal as triton_rollback_minimal,
)


def snapshot_current(**context: Any) -> None:
    return triton_snapshot_current(ti=context["ti"])


def triton_materialize_task(**context: Any) -> None:
    """
    - Promotion: alias -> MLflow version 기반 materialize (shadow=False)
    - Shadow: run_id -> timestamp 기반 materialize (shadow=True)
    """
    s = Settings.load()
    ti = context["ti"]

    al = ti.xcom_pull(task_ids=TRAIN_TASK_ID, key=XCOM_ALIAS) or s.alias
    run_id = ti.xcom_pull(task_ids=TRAIN_TASK_ID, key=XCOM_RUN_ID)
    version = ti.xcom_pull(task_ids=REGISTER_TASK_ID, key=XCOM_VERSION)

    # promotion path
    if version:
        return triton_materialize(ti=ti, alias=str(al), shadow=False)

    # shadow path
    if not run_id:
        raise ValueError("Shadow deploy 불가: run_id XCom 누락 (train_and_evaluate 확인 필요)")

    return triton_materialize(ti=ti, alias=str(al), run_id=str(run_id), shadow=True)


def triton_load_task(**context: Any) -> None:
    return triton_load_task_impl(ti=context["ti"])


def triton_ready_task(**context: Any) -> None:
    return triton_ready_task_impl(ti=context["ti"])


def triton_infer_smoke_task(**context: Any) -> None:
    return triton_infer_smoke_task_impl(ti=context["ti"])


def commit_current(**context: Any) -> None:
    return triton_commit_current(ti=context["ti"])


def triton_rollback_task(**context: Any) -> None:
    return triton_rollback_minimal(ti=context["ti"])
