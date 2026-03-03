# dags/ml_code/triton_xcom.py
from __future__ import annotations

from typing import Any, Optional, Sequence

from airflow.utils.log.logging_mixin import LoggingMixin

from ml_code.config import cfg
from mlops_lib.core.ids import TRITON_MAT_TASK_ID, TRITON_SNAPSHOT_TASK_ID

log = LoggingMixin().log

# -----------------------
# Legacy support (optional)
# -----------------------
_LEGACY_MAT_TASK_IDS: Sequence[str] = (TRITON_MAT_TASK_ID, "materialize_repo")
_LEGACY_SNAPSHOT_TASK_IDS: Sequence[str] = (TRITON_SNAPSHOT_TASK_ID, "snapshot_current")


def allow_legacy_task_ids() -> bool:
    return str(cfg("allow_legacy_task_ids", "false")).lower() in ("1", "true", "yes", "y", "on")


def task_id_candidates(primary: str, legacy: Sequence[str]) -> Sequence[str]:
    return legacy if allow_legacy_task_ids() else (primary,)


def mat_task_ids() -> Sequence[str]:
    return task_id_candidates(TRITON_MAT_TASK_ID, _LEGACY_MAT_TASK_IDS)


def snapshot_task_ids() -> Sequence[str]:
    return task_id_candidates(TRITON_SNAPSHOT_TASK_ID, _LEGACY_SNAPSHOT_TASK_IDS)


def xcom_pull_any(ti, *, key: str, task_ids: Sequence[str]) -> Optional[Any]:
    for tid in task_ids:
        v = ti.xcom_pull(task_ids=tid, key=key)
        if v is not None:
            return v
    return None


def require_xcom(ti, *, key: str, task_ids: Sequence[str], hint: str) -> Any:
    v = xcom_pull_any(ti, key=key, task_ids=task_ids)
    if v is None or v == "":
        raise RuntimeError(
            "XCom missing: "
            f"key='{key}' from task_ids={list(task_ids)}. "
            f"Hint: {hint}"
        )
    return v
