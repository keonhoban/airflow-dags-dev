# dags/ml_code/triton_deploy.py
from __future__ import annotations

"""
✅ Backward-compatible adapter (re-export only)

목적:
- 기존 코드/파이프라인에서:
  from ml_code.triton_deploy import snapshot_current, materialize, triton_load, ...

  형태의 import가 깨지지 않도록,
  실제 구현은 triton_tasks / triton_actions 로 분리하고
  여기서는 "이름 그대로" re-export만 제공합니다.

원칙(면접용 한 줄):
- actions: 순수 실행(HTTP/MLflow/FS)  -> ml_code/triton_actions.py
- tasks  : Airflow ti/xcom wrapper    -> ml_code/triton_tasks.py
- deploy : 외부 import 안정 API(호환) -> (this file)
"""

# ✅ Airflow Task callable (ti/xcom wrapper)
from ml_code.triton_tasks import (
    snapshot_current,          # task callable
    materialize,               # task callable
    triton_load_task as triton_load,          # task callable (compat name)
    triton_ready_task as triton_ready,        # task callable (compat name)
    triton_infer_smoke_task as triton_infer_smoke,  # task callable (compat name)
    commit_current,            # task callable
    rollback_minimal,          # task callable
    rollback_manual,           # ops callable (manual)
)

# ✅ 외부에 노출하는 API를 SSOT로 확정
__all__ = [
    "snapshot_current",
    "materialize",
    "triton_load",
    "triton_ready",
    "triton_infer_smoke",
    "commit_current",
    "rollback_minimal",
    "rollback_manual",
]
