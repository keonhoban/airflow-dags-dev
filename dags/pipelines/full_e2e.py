# dags/pipelines/full_e2e.py
from __future__ import annotations

"""
Facade module (stable import path)

✅ 목표
- dags/e2e_full.py 의 `from pipelines import full_e2e as p` 를 절대 깨지지 않게 유지
- 기능별로 분리된 p_*.py 에서 callables를 재수출(re-export)해서
  "읽기"  + "유지보수 분리"를 동시에 달성
"""

# -----------------------
# DP task callables (existing interface)
# -----------------------
from mlops_lib.dp.tasks import (
    task_extract_raw_data as dp_extract,
    task_validate_data as dp_validate,
    task_build_features as dp_build,
    task_store_features as dp_store,
    task_summarize_run as dp_summary,
)

# -----------------------
# Orchestration callables (re-export)
# -----------------------
from pipelines.p_drift import drift_gate_task
from pipelines.p_train import train_and_evaluate, branch_by_accuracy
from pipelines.p_register import register_model_task, sensor_ready_func, notify_shadow_reason
from pipelines.p_triton import (
    snapshot_current,
    triton_materialize_task,
    triton_load_task,
    triton_ready_task,
    triton_infer_smoke_task,
    commit_current,
    triton_rollback_task,
)
from pipelines.p_reload import fastapi_reload_task
from pipelines.p_observe import observe_post_deploy_metrics

__all__ = [
    # DP
    "dp_extract",
    "dp_validate",
    "dp_build",
    "dp_store",
    "dp_summary",
    # quality
    "drift_gate_task",
    # train/branch
    "train_and_evaluate",
    "branch_by_accuracy",
    # register/sensor/notify
    "register_model_task",
    "sensor_ready_func",
    "notify_shadow_reason",
    # triton wrappers
    "snapshot_current",
    "triton_materialize_task",
    "triton_load_task",
    "triton_ready_task",
    "triton_infer_smoke_task",
    "commit_current",
    "triton_rollback_task",
    # reload
    "fastapi_reload_task",
    # observe
    "observe_post_deploy_metrics",
]
