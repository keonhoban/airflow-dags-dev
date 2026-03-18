# dags/ml_code/triton_xcom.py
from __future__ import annotations

from typing import Any, Optional, Sequence

from airflow.utils.log.logging_mixin import LoggingMixin

from ml_code.config import cfg
from mlops_lib.core.ids import TRITON_MAT_TASK_ID, TRITON_SNAPSHOT_TASK_ID

log = LoggingMixin().log

# -----------------------
# Legacy task_id support
#
# 배경: TaskGroup 도입 전 "materialize_repo", "snapshot_current" 라는 flat task_id를
#       사용했던 시기의 XCom 데이터와 호환하기 위한 fallback.
#
# 기본값: false (레거시 경로 비활성화)
#   - Airflow Variable "allow_legacy_task_ids"를 "true"로 설정하면 활성화됨.
#   - 신규 배포에서는 절대 활성화하지 않는다.
#
# TODO(제거 조건): 다음 조건을 모두 충족하면 이 블록 전체와 호출부를 삭제한다.
#   1. 모든 실행 중인 DAGRun이 현재 task_id(TRITON_MAT_TASK_ID 등)를 사용하도록 마이그레이션 완료
#   2. 레거시 XCom 레코드가 Airflow MetadataDB에서 만료(기본 retention: 30일)
#   3. Airflow Variable "allow_legacy_task_ids"가 존재하지 않거나 "false"인 상태가 30일 이상 유지
# -----------------------
_LEGACY_MAT_TASK_IDS: Sequence[str] = (TRITON_MAT_TASK_ID, "materialize_repo")
_LEGACY_SNAPSHOT_TASK_IDS: Sequence[str] = (TRITON_SNAPSHOT_TASK_ID, "snapshot_current")

_ALLOW_LEGACY_DEFAULT = "false"  # 기본값 명시 — 변경 시 이 상수만 수정


def allow_legacy_task_ids() -> bool:
    return str(cfg("allow_legacy_task_ids", _ALLOW_LEGACY_DEFAULT)).lower() in (
        "1", "true", "yes", "y", "on"
    )


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
