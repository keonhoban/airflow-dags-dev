# dags/pipelines/p_train.py
from __future__ import annotations

from typing import Any
from airflow.utils.log.logging_mixin import LoggingMixin

from mlops_lib.core.ids import (
    # task ids
    DP_STORE_TASK_ID,
    DP_BUILD_TASK_ID,
    TRAIN_TASK_ID,
    DRIFT_GATE_TASK_ID,
    SHADOW_START_TASK_ID,
    REGISTER_TASK_ID,
    # xcom keys
    XCOM_ALIAS,
    XCOM_MODEL_NAME,
    XCOM_ACCURACY,
    XCOM_RUN_ID,
    XCOM_FS_FEATURE_URI,
    XCOM_FS_VERSION,
    XCOM_FS_SCHEMA_HASH,
    XCOM_SHADOW_REASON,
    XCOM_DRIFT_BLOCK_PROMOTION,
    XCOM_DRIFT_REASON,
    # shadow reason
    SHADOW_REASON_TRAIN_SKIPPED,
    SHADOW_REASON_ACCURACY_INVALID,
    SHADOW_REASON_BELOW_THRESHOLD,
    SHADOW_REASON_DRIFT_DETECTED,
)

from mlops_lib.core.policy import Settings

# ✅ notify는 observability 쪽으로 이동 (core -> observability 방향 유지)
from mlops_lib.observability.notify import (
    notify_train_completed,
    notify_branch_promotion,
    notify_branch_shadow,
)

from ml_code.train_model import train_model, TrainSkippableError

log = LoggingMixin().log


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

    # 0) Drift gate 우선 (Pre-deploy quality gate)
    drift_block = ti.xcom_pull(task_ids=DRIFT_GATE_TASK_ID, key=XCOM_DRIFT_BLOCK_PROMOTION)
    if str(drift_block).strip().lower() in ("1", "true", "yes", "y", "on"):
        # ✅ branch에서 SSOT로 남김
        ti.xcom_push(key=XCOM_SHADOW_REASON, value=SHADOW_REASON_DRIFT_DETECTED)

        drift_reason = ti.xcom_pull(task_ids=DRIFT_GATE_TASK_ID, key=XCOM_DRIFT_REASON) or "DRIFT_BLOCK"
        log.warning("[branch_by_accuracy] drift_block=true -> shadow. drift_reason=%s", drift_reason)

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
