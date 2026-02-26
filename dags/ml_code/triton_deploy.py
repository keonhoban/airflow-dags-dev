# dags/ml_code/triton_deploy.py
from __future__ import annotations

"""
✅ 호환 어댑터 (Backward-compatible re-export)

- 기존 코드/파이프라인에서:
  from ml_code.triton_deploy import snapshot_current, materialize, triton_load, ...

  형태의 import가 깨지지 않도록,
  실제 구현은 triton_tasks / triton_actions 로 분리하고
  여기서는 "이름 그대로" re-export만 제공합니다.

- 원칙:
  - 구현(행위): ml_code/triton_actions.py
  - Airflow ti/xcom wrapper: ml_code/triton_tasks.py
  - 본 파일: import 호환 + 유지보수용 SSOT 엔트리
"""

from ml_code.triton_tasks import (
    snapshot_current,
    materialize,
    triton_load_task as triton_load,
    triton_ready_task as triton_ready,
    triton_infer_smoke_task as triton_infer_smoke,
    commit_current,
    rollback_minimal,
    rollback_manual,
)
