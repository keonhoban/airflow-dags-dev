# dags/pipelines/full_e2e.py
from __future__ import annotations

from typing import Any
from airflow.utils.log.logging_mixin import LoggingMixin
from airflow.exceptions import AirflowException

from mlops_lib.core.ids import (
    # task ids
    DP_STORE_TASK_ID,
    DP_BUILD_TASK_ID,
    TRAIN_TASK_ID,
    BRANCH_TASK_ID,
    REGISTER_TASK_ID,
    DEPLOY_MATERIALIZE_TASK_ID,
    SHADOW_START_TASK_ID,
    DRIFT_GATE_TASK_ID,
    # xcom keys (pipeline scope)
    XCOM_ALIAS,
    XCOM_MODEL_NAME,
    XCOM_ACCURACY,
    XCOM_RUN_ID,
    XCOM_VERSION,
    XCOM_FS_FEATURE_URI,
    XCOM_FS_VERSION,
    XCOM_FS_SCHEMA_HASH,
    XCOM_SHADOW_REASON,
    XCOM_DRIFT_BLOCK_PROMOTION,
    XCOM_DRIFT_REASON,
    # shadow reason (SSOT)
    SHADOW_REASON_TRAIN_SKIPPED,
    SHADOW_REASON_ACCURACY_INVALID,
    SHADOW_REASON_BELOW_THRESHOLD,
    SHADOW_REASON_DRIFT_DETECTED,
    # triton xcom keys (SSOT)
    K_DEPLOY_MODE as TRITON_XCOM_DEPLOY_MODE,
    K_DEPLOY_VERSION as TRITON_XCOM_DEPLOY_VERSION,
    K_RUN_ID as TRITON_XCOM_RUN_ID,
)

from mlops_lib.core.policy import (
    Settings,
    notify_train_completed,
    notify_branch_promotion,
    notify_branch_shadow,
    notify_register_completed,
    notify_shadow_reason as notify_shadow_reason_policy,
)

from mlops_lib.dp.tasks import (
    task_extract_raw_data as dp_extract,
    task_validate_data as dp_validate,
    task_build_features as dp_build,
    task_store_features as dp_store,
    task_summarize_run as dp_summary,
)

from ml_code.train_model import train_model, TrainSkippableError
from ml_code.register_model import register_model
from ml_code.sensor_model_ready import check_model_ready

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

from ml_code.trigger_reload import trigger_reload

# ✅ Observability (metric-based auto rollback) - "근본 해결"
from mlops_lib.observability.auto_rollback import AutoRollback

# ✅ Drift Gate (Pre-deploy quality gate)
from mlops_lib.quality.drift_gate import drift_gate

log = LoggingMixin().log

"""
✅ This module contains ONLY orchestration callables used by DAG entrypoints.
- No DAG() definitions here.
- SSOT for task_id/xcom keys: mlops_lib.core.ids
- Policy(Settings/notify): mlops_lib.core.policy
"""


# -----------------------
# Drift Gate
# -----------------------
def drift_gate_task(**context: Any) -> None:
    return drift_gate(**context)


# -----------------------
# Train / Branch
# -----------------------
def train_and_evaluate(**context: Any) -> None:
    s = Settings.load()
    ti = context["ti"]

    feature_uri = ti.xcom_pull(key=XCOM_FS_FEATURE_URI, task_ids=DP_STORE_TASK_ID)
    fs_version = ti.xcom_pull(key=XCOM_FS_VERSION, task_ids=DP_STORE_TASK_ID)
    schema_hash = ti.xcom_pull(key=XCOM_FS_SCHEMA_HASH, task_ids=DP_BUILD_TASK_ID)

    # ✅ DAG 기준 SSOT를 XCom에 기록 (후속 task에서 Settings 변동에도 안전)
    ti.xcom_push(key=XCOM_ALIAS, value=s.alias)
    ti.xcom_push(key=XCOM_MODEL_NAME, value=s.model_name)

    try:
        acc, run_id = train_model(
            C=s.logreg_c,
            max_iter=s.logreg_max_iter,
            feature_uri=feature_uri,
            fs_version=fs_version,
            schema_hash=schema_hash,
            env=s.env,
            code_version=s.code_version,
        )
    except TrainSkippableError:
        # ✅ train은 "결과만" 남김. shadow reason 결정은 branch가 SSOT.
        ti.xcom_push(key=XCOM_ACCURACY, value=None)
        ti.xcom_push(key=XCOM_RUN_ID, value=None)
        return

    ti.xcom_push(key=XCOM_ACCURACY, value=float(acc))
    ti.xcom_push(key=XCOM_RUN_ID, value=str(run_id))

    notify_train_completed(
        env=s.env,
        accuracy=float(acc),
        alias=s.alias,
        run_id=str(run_id),
        fs_version=str(fs_version),
        schema_hash=str(schema_hash),
        code_version=str(s.code_version) if s.code_version else "",
    )


def branch_by_accuracy(**context: Any) -> str:
    """
    Returns task_id to follow:
    - REGISTER_TASK_ID (promotion)
    - SHADOW_START_TASK_ID (shadow)
    """
    s = Settings.load()
    ti = context["ti"]

    # ======================================================
    # 0) Drift gate 우선 (Pre-deploy 품질 게이트)
    # - drift_gate_task가 block이면 무조건 shadow로 보냄
    # - BUT: notify_failure는 BRANCH_TASK_ID의 XCom을 읽으므로
    #        branch에서 shadow_reason을 한 번 더 SSOT로 남겨준다.
    # ======================================================
    drift_block = ti.xcom_pull(task_ids=DRIFT_GATE_TASK_ID, key=XCOM_DRIFT_BLOCK_PROMOTION)
    if str(drift_block).strip().lower() in ("1", "true", "yes", "y", "on"):
        # ✅ 핵심: notify_shadow_reason()가 BRANCH_TASK_ID에서 읽게 보장
        ti.xcom_push(key=XCOM_SHADOW_REASON, value=SHADOW_REASON_DRIFT_DETECTED)

        drift_reason = ti.xcom_pull(task_ids=DRIFT_GATE_TASK_ID, key=XCOM_DRIFT_REASON) or "DRIFT_BLOCK"
        log.warning("[branch_by_accuracy] drift_block=true -> shadow. drift_reason=%s", drift_reason)

        # (선택) “Branch: shadow” 알림을 여기서도 남기고 싶으면 아래 라인 활성화 가능
        # notify_branch_shadow(env=s.env, reason=SHADOW_REASON_DRIFT_DETECTED, threshold=s.accuracy_threshold)

        return SHADOW_START_TASK_ID

    acc = ti.xcom_pull(task_ids=TRAIN_TASK_ID, key=XCOM_ACCURACY)

    if acc is None:
        ti.xcom_push(key=XCOM_SHADOW_REASON, value=SHADOW_REASON_TRAIN_SKIPPED)
        notify_branch_shadow(env=s.env, reason=SHADOW_REASON_TRAIN_SKIPPED, threshold=s.accuracy_threshold)
        return SHADOW_START_TASK_ID

    try:
        acc_f = float(acc)
    except Exception:
        ti.xcom_push(key=XCOM_SHADOW_REASON, value=SHADOW_REASON_ACCURACY_INVALID)
        notify_branch_shadow(env=s.env, reason=SHADOW_REASON_ACCURACY_INVALID, threshold=s.accuracy_threshold)
        return SHADOW_START_TASK_ID

    if acc_f >= s.accuracy_threshold:
        notify_branch_promotion(env=s.env, accuracy=acc_f, threshold=s.accuracy_threshold)
        return REGISTER_TASK_ID

    ti.xcom_push(key=XCOM_SHADOW_REASON, value=SHADOW_REASON_BELOW_THRESHOLD)
    notify_branch_shadow(
        env=s.env,
        reason=SHADOW_REASON_BELOW_THRESHOLD,
        threshold=s.accuracy_threshold,
        accuracy=acc_f,
    )
    return SHADOW_START_TASK_ID


# -----------------------
# Register / Sensor
# -----------------------
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

    reason = None
    if ti:
        reason = ti.xcom_pull(task_ids=BRANCH_TASK_ID, key=XCOM_SHADOW_REASON)

    notify_shadow_reason_policy(env=s.env, reason=reason)


# -----------------------
# Triton deploy wrappers
# -----------------------
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


# -----------------------
# FastAPI reload
# -----------------------
def fastapi_reload_task(**context: Any) -> None:
    s = Settings.load()
    ti = context["ti"]

    al = ti.xcom_pull(task_ids=TRAIN_TASK_ID, key=XCOM_ALIAS) or s.alias

    deploy_mode = ti.xcom_pull(task_ids=DEPLOY_MATERIALIZE_TASK_ID, key=TRITON_XCOM_DEPLOY_MODE) or "promote"
    deploy_version = ti.xcom_pull(task_ids=DEPLOY_MATERIALIZE_TASK_ID, key=TRITON_XCOM_DEPLOY_VERSION)
    run_id = ti.xcom_pull(task_ids=DEPLOY_MATERIALIZE_TASK_ID, key=TRITON_XCOM_RUN_ID)

    if str(deploy_mode) == "shadow":
        if not run_id:
            raise ValueError("FastAPI shadow reload 불가: run_id XCom 누락 (deploy.materialize_repo 확인 필요)")
        trigger_reload(str(al), run_id=str(run_id))
        return

    if deploy_version is None:
        raise ValueError("FastAPI promotion reload 불가: deploy_version XCom 누락 (deploy.materialize_repo 확인 필요)")

    trigger_reload(str(al), deploy_version=int(deploy_version))


# -----------------------
# Post-deploy observation (Auto Rollback)
# -----------------------
def observe_post_deploy_metrics(**context: Any) -> None:
    """
    배포 후 관측 결과가 나쁘면 task 실패 -> e2e_full.py에서 rollback_minimal 트리거
    - 여기서는 '결정'만 내리고
    - 롤백 실행은 DAG(trigger_rule=ONE_FAILED)에게 맡긴다 (오케스트레이션 SSOT)
    """
    s = Settings.load()

    ar = AutoRollback()
    decision = ar.evaluate()

    log.info(
        "[observe_post_deploy_metrics] env=%s decision=%s signals=%s",
        s.env,
        decision.reason,
        getattr(decision, "signals", None),
    )

    if getattr(decision, "should_rollback", False):
        raise AirflowException(f"[AUTO-ROLLBACK] {decision.reason} | signals={decision.signals}")
