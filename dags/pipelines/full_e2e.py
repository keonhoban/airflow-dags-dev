# dags/pipelines/full_e2e.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Optional

from airflow.models import Variable
from airflow.utils.log.logging_mixin import LoggingMixin

from utils.slack_alerts import notify_info, notify_success, notify_skip

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

# ✅ Triton은 "tasks" 계층만 바라보게 고정 (deploy/actions 혼재 방지)
# ✅ 이름 충돌 방지: *impl 로 import (actions의 triton_load/ready/infer와 절대 충돌 금지)
from ml_code.triton_tasks import (
    snapshot_current as triton_snapshot_current,
    materialize as triton_materialize,
    triton_load_task as triton_load_task_impl,
    triton_ready_task as triton_ready_task_impl,
    triton_infer_smoke_task as triton_infer_smoke_task_impl,
    commit_current as triton_commit_current,
    rollback_minimal as triton_rollback_minimal,
    # ✅ materialize()가 push하는 XCom key SSOT를 직접 사용 (암묵 문자열 일치 제거)
    K_DEPLOY_MODE as TRITON_XCOM_DEPLOY_MODE,
    K_DEPLOY_VERSION as TRITON_XCOM_DEPLOY_VERSION,
    K_RUN_ID as TRITON_XCOM_RUN_ID,
)

from ml_code.trigger_reload import trigger_reload

log = LoggingMixin().log


"""
✅ This module contains ONLY orchestration callables used by DAG entrypoints.
- No DAG() definitions here.
- All "stringly-typed" IDs/keys are centralized below as SSOT.
"""


# -----------------------
# SSOT: Task IDs / XCom keys
# -----------------------
# TaskGroup 사용 시 task_id는 "group.task" 형태로 고정됨
DP_STORE_TASK_ID = "dp.store_features"
DP_BUILD_TASK_ID = "dp.build_features"
TRAIN_TASK_ID = "train_and_evaluate"
REGISTER_TASK_ID = "register_model_task"
DEPLOY_MATERIALIZE_TASK_ID = "deploy.materialize_repo"

# Branch targets (SSOT) - DAG(e2e_full.py)에서 존재해야 함
SHADOW_START_TASK_ID = "shadow_start"

# XCom keys (pipeline scope)
XCOM_ALIAS = "alias"
XCOM_MODEL_NAME = "model_name"
XCOM_ACCURACY = "accuracy"
XCOM_RUN_ID = "run_id"
XCOM_VERSION = "version"

XCOM_FS_FEATURE_URI = "fs_feature_uri"
XCOM_FS_VERSION = "fs_version"
XCOM_FS_SCHEMA_HASH = "fs_schema_hash"

# shadow reason SSOT (면접/운영 설명 포인트)
XCOM_SHADOW_REASON = "shadow_reason"
SHADOW_REASON_TRAIN_SKIPPED = "train_skipped"
SHADOW_REASON_ACCURACY_INVALID = "accuracy_invalid"
SHADOW_REASON_BELOW_THRESHOLD = "below_threshold"


# -----------------------
# Settings (SSOT)
# -----------------------
def _v(key: str, default: Optional[str] = None) -> Optional[str]:
    try:
        return Variable.get(key)
    except Exception:
        return default


def _to_float(raw: Optional[str], default: float) -> float:
    try:
        return float(str(raw))
    except Exception:
        return default


def _to_int(raw: Optional[str], default: int) -> int:
    try:
        return int(str(raw))
    except Exception:
        return default


@dataclass(frozen=True)
class Settings:
    env: str
    accuracy_threshold: float
    logreg_c: float
    logreg_max_iter: int
    model_name: str
    alias: str
    code_version: Optional[str]

    @classmethod
    def load(cls) -> "Settings":
        env = (_v("triton_env", "dev") or "dev").strip()
        th = _to_float(_v("accuracy_threshold", "0.60"), 0.60)

        c = _to_float(_v("logreg_C", "1.0"), 1.0)
        it = _to_int(_v("logreg_max_iter", "200"), 200)

        model_name = (_v("triton_model_name", _v("model_name", "best_model")) or "best_model").strip()
        alias = (_v("mlflow_alias", "A") or "A").strip()

        # 제출/운영용: git sha 같은 버전값을 Variable로 주입 가능(없으면 None)
        code_version = (_v("code_version", None) or None)
        if code_version is not None:
            code_version = str(code_version).strip() or None

        return cls(
            env=env,
            accuracy_threshold=th,
            logreg_c=c,
            logreg_max_iter=it,
            model_name=model_name,
            alias=alias,
            code_version=code_version,
        )


# -----------------------
# Train / Branch
# -----------------------
def train_and_evaluate(**context: Any) -> None:
    s = Settings.load()
    ti = context["ti"]

    feature_uri = ti.xcom_pull(key=XCOM_FS_FEATURE_URI, task_ids=DP_STORE_TASK_ID)
    fs_version = ti.xcom_pull(key=XCOM_FS_VERSION, task_ids=DP_STORE_TASK_ID)
    schema_hash = ti.xcom_pull(key=XCOM_FS_SCHEMA_HASH, task_ids=DP_BUILD_TASK_ID)

    # DAG 기준 SSOT를 XCom에 기록 (후속 task에서 Settings 변동/override에도 안정)
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
    except TrainSkippableError as e:
        notify_skip("Train skipped", env=s.env, reason=str(e))
        ti.xcom_push(key=XCOM_ACCURACY, value=None)
        ti.xcom_push(key=XCOM_RUN_ID, value=None)
        return

    ti.xcom_push(key=XCOM_ACCURACY, value=float(acc))
    ti.xcom_push(key=XCOM_RUN_ID, value=str(run_id))

    notify_info(
        "Train completed",
        env=s.env,
        accuracy=f"{float(acc):.4f}",
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

    acc = ti.xcom_pull(task_ids=TRAIN_TASK_ID, key=XCOM_ACCURACY)

    if acc is None:
        ti.xcom_push(key=XCOM_SHADOW_REASON, value=SHADOW_REASON_TRAIN_SKIPPED)
        notify_info("Branch: shadow (train skipped)", env=s.env, threshold=str(s.accuracy_threshold))
        return SHADOW_START_TASK_ID

    try:
        acc_f = float(acc)
    except Exception:
        ti.xcom_push(key=XCOM_SHADOW_REASON, value=SHADOW_REASON_ACCURACY_INVALID)
        notify_info("Branch: shadow (accuracy invalid)", env=s.env, threshold=str(s.accuracy_threshold))
        return SHADOW_START_TASK_ID

    if acc_f >= s.accuracy_threshold:
        notify_info("Branch: promotion", env=s.env, accuracy=f"{acc_f:.4f}", threshold=str(s.accuracy_threshold))
        return REGISTER_TASK_ID

    ti.xcom_push(key=XCOM_SHADOW_REASON, value=SHADOW_REASON_BELOW_THRESHOLD)
    notify_info(
        "Branch: shadow (below threshold)",
        env=s.env,
        accuracy=f"{acc_f:.4f}",
        threshold=str(s.accuracy_threshold),
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

    notify_success(
        "MLflow register+alias completed",
        env=s.env,
        model=str(mname),
        alias=str(al),
        version=str(version),
    )


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
    - train_skipped / accuracy_invalid / below_threshold
    """
    s = Settings.load()
    ti = context.get("ti")

    reason = None
    if ti:
        reason = ti.xcom_pull(task_ids="check_result", key=XCOM_SHADOW_REASON)

    title = "Shadow path selected"
    if reason == SHADOW_REASON_TRAIN_SKIPPED:
        notify_skip(title, env=s.env, reason="train skipped", next_action="데이터/피처/라벨 조건 확인")
        return
    if reason == SHADOW_REASON_ACCURACY_INVALID:
        notify_skip(title, env=s.env, reason="accuracy invalid", next_action="train task의 accuracy 산출/형 변환 확인")
        return

    # default: below threshold
    notify_skip(
        title,
        env=s.env,
        reason="accuracy below threshold",
        next_action="feature/label/model 개선 후 재시도",
    )


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
        notify_info(
            "Triton deploy: promotion path (alias->MLflow version)",
            env=s.env,
            alias=str(al),
            version=str(version),
        )
        return triton_materialize(ti=ti, alias=str(al), shadow=False)

    # shadow path
    if not run_id:
        raise ValueError("Shadow deploy 불가: run_id XCom 누락 (train_and_evaluate 확인 필요)")

    notify_info(
        "Triton deploy: shadow path (run_id->timestamp)",
        env=s.env,
        alias=str(al),
        run_id=str(run_id),
    )
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

    # ✅ Triton materialize()가 push한 키를 그대로 사용 (SSOT)
    deploy_mode = ti.xcom_pull(task_ids=DEPLOY_MATERIALIZE_TASK_ID, key=TRITON_XCOM_DEPLOY_MODE) or "promote"
    deploy_version = ti.xcom_pull(task_ids=DEPLOY_MATERIALIZE_TASK_ID, key=TRITON_XCOM_DEPLOY_VERSION)
    run_id = ti.xcom_pull(task_ids=DEPLOY_MATERIALIZE_TASK_ID, key=TRITON_XCOM_RUN_ID)

    if str(deploy_mode) == "shadow":
        if not run_id:
            raise ValueError("FastAPI shadow reload 불가: run_id XCom 누락 (deploy.materialize_repo 확인 필요)")
        trigger_reload(str(al), run_id=str(run_id))
        notify_success("FastAPI reload completed (shadow/run_id)", env=s.env, alias=str(al), run_id=str(run_id))
        return

    if deploy_version is None:
        raise ValueError("FastAPI promotion reload 불가: deploy_version XCom 누락 (deploy.materialize_repo 확인 필요)")

    trigger_reload(str(al), deploy_version=int(deploy_version))
    notify_success(
        "FastAPI reload completed (promotion/deploy_version)",
        env=s.env,
        alias=str(al),
        deploy_version=str(deploy_version),
    )
