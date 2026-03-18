# dags/mlops_lib/core/ids.py
from __future__ import annotations

"""
SSOT (Single Source of Truth)

- DAG ID
- TaskGroup IDs
- Task IDs (TaskGroup 포함 형태까지)
- XCom keys
- Shadow reason codes
- Triton deploy 관련 key

이 파일은 "문자열 하드코딩 제거"를 위한 중앙 정의 지점입니다.
DAG / pipelines / ml_code 어디에서도 문자열을 직접 쓰지 않습니다.

- 기존 상수들은 그대로 유지(호환성)
- DAG 파일 import 압축을 위해 E2E 클래스(alias 집합) 추가
- OBSERVE task_id 네이밍 정합성 개선
"""

# ============================================================
# DAG / TaskGroups (SSOT)
# ============================================================

DAG_ID_E2E_FULL = "e2e_full"

TG_DP = "dp"
TG_DEPLOY = "deploy"

# ============================================================
# Task IDs (SSOT)
# ============================================================

# --- DP TaskGroup ---
DP_EXTRACT_TASK_ID = f"{TG_DP}.extract_raw_data"
DP_VALIDATE_TASK_ID = f"{TG_DP}.validate_data"
DP_BUILD_TASK_ID = f"{TG_DP}.build_features"
DP_STORE_TASK_ID = f"{TG_DP}.store_features"

SUMMARIZE_TASK_ID = "summarize_run"

# --- Quality / Gate ---
DRIFT_GATE_TASK_ID = "drift_gate"

# --- Train / Branch ---
TRAIN_TASK_ID = "train_and_evaluate"
BRANCH_TASK_ID = "check_result"
REGISTER_TASK_ID = "register_model_task"

PROMOTION_START_TASK_ID = "promotion_start"
SHADOW_START_TASK_ID = "shadow_start"
NOTIFY_FAILURE_TASK_ID = "notify_failure"

SENSOR_MODEL_READY_TASK_ID = "check_model_ready"

# --- Deploy TaskGroup ---
DEPLOY_SNAPSHOT_TASK_ID = f"{TG_DEPLOY}.snapshot_current"
DEPLOY_MATERIALIZE_TASK_ID = f"{TG_DEPLOY}.materialize_repo"
DEPLOY_TRITON_LOAD_TASK_ID = f"{TG_DEPLOY}.triton_load"
DEPLOY_TRITON_READY_TASK_ID = f"{TG_DEPLOY}.triton_ready"
DEPLOY_TRITON_SMOKE_TASK_ID = f"{TG_DEPLOY}.triton_infer_smoke"

COMMIT_CURRENT_TASK_ID = "commit_current"
FASTAPI_RELOAD_TASK_ID = "fastapi_reload"

# ✅ 정합성 개선: callable 의미와 task_id를 같은 어휘로 맞춤
OBSERVE_METRICS_TASK_ID = "observe_post_deploy_metrics"

ROLLBACK_MINIMAL_TASK_ID = "rollback_minimal"

# ============================================================
# XCom keys (Pipeline scope SSOT)
# ============================================================

XCOM_ALIAS = "alias"
XCOM_MODEL_NAME = "model_name"
XCOM_ACCURACY = "accuracy"
XCOM_RUN_ID = "run_id"
XCOM_VERSION = "version"

XCOM_FS_FEATURE_URI = "fs_feature_uri"
XCOM_FS_VERSION = "fs_version"
XCOM_FS_SCHEMA_HASH = "fs_schema_hash"

# dp/store.py가 이미 push 중인 key를 SSOT로 승격
XCOM_FS_LATEST_PREFIX = "fs_latest_prefix"

XCOM_SHADOW_REASON = "shadow_reason"

# Drift gate 결과 SSOT
XCOM_DRIFT_BLOCK_PROMOTION = "drift_block_promotion"  # bool
XCOM_DRIFT_REASON = "drift_reason"  # str

# ============================================================
# Shadow reason codes (SSOT)
# ============================================================

SHADOW_REASON_TRAIN_SKIPPED = "train_skipped"
SHADOW_REASON_ACCURACY_INVALID = "accuracy_invalid"
SHADOW_REASON_BELOW_THRESHOLD = "below_threshold"
SHADOW_REASON_DRIFT_DETECTED = "drift_detected"

# ============================================================
# Triton XCom keys (SSOT)
# ============================================================

K_MODEL = "model"
K_MODEL_DIR = "model_dir"
K_DEPLOY_VERSION = "deploy_version"
K_RUN_ID = "run_id"
K_ALIAS = "alias"
K_DEPLOY_MODE = "deploy_mode"
K_N_FEATURES = "n_features"
K_N_CLASSES = "n_classes"
K_ONNX_INPUT_NAME = "onnx_input_name"

K_PREV_CURRENT = "prev_current"

# ============================================================
# Triton Task ID alias (가독성용)
# ============================================================

TRITON_MAT_TASK_ID = DEPLOY_MATERIALIZE_TASK_ID
TRITON_SNAPSHOT_TASK_ID = DEPLOY_SNAPSHOT_TASK_ID


# ============================================================
# DAG import 압축용 alias 묶음 (NEW)
# - DAG 파일은 "from ids import E2E as I" 하나로 끝낼 수 있음
# - 기존 상수는 유지되므로 다른 모듈 영향 없음
# ============================================================

class E2E:
    # dag / tg
    DAG_ID = DAG_ID_E2E_FULL
    TG_DP = TG_DP
    TG_DEPLOY = TG_DEPLOY

    # dp
    DP_EXTRACT = DP_EXTRACT_TASK_ID
    DP_VALIDATE = DP_VALIDATE_TASK_ID
    DP_BUILD = DP_BUILD_TASK_ID
    DP_STORE = DP_STORE_TASK_ID
    SUMMARIZE = SUMMARIZE_TASK_ID

    # gate/train/branch
    DRIFT_GATE = DRIFT_GATE_TASK_ID
    TRAIN = TRAIN_TASK_ID
    BRANCH = BRANCH_TASK_ID
    REGISTER = REGISTER_TASK_ID
    PROMOTION_START = PROMOTION_START_TASK_ID
    SHADOW_START = SHADOW_START_TASK_ID
    NOTIFY_FAILURE = NOTIFY_FAILURE_TASK_ID
    SENSOR_MODEL_READY = SENSOR_MODEL_READY_TASK_ID

    # deploy
    DEPLOY_SNAPSHOT = DEPLOY_SNAPSHOT_TASK_ID
    DEPLOY_MATERIALIZE = DEPLOY_MATERIALIZE_TASK_ID
    DEPLOY_TRITON_LOAD = DEPLOY_TRITON_LOAD_TASK_ID
    DEPLOY_TRITON_READY = DEPLOY_TRITON_READY_TASK_ID
    DEPLOY_TRITON_SMOKE = DEPLOY_TRITON_SMOKE_TASK_ID

    # post
    COMMIT = COMMIT_CURRENT_TASK_ID
    FASTAPI_RELOAD = FASTAPI_RELOAD_TASK_ID
    OBSERVE = OBSERVE_METRICS_TASK_ID
    ROLLBACK_MINIMAL = ROLLBACK_MINIMAL_TASK_ID

    # ----------------------------------------------------------
    # TaskGroup 내부 task_id (suffix only)
    #
    # TaskGroup 안에서 task_id를 등록할 때는 group prefix 없이
    # suffix만 전달해야 한다. Airflow가 자동으로 prefix를 붙여
    # 최종 task_id = "{group_id}.{suffix}" 가 된다.
    #
    # 전체 경로(DP_EXTRACT 등)와 1:1로 대응하므로,
    # 새 task_id 추가 시 전체 경로와 suffix 양쪽을 함께 정의한다.
    # ----------------------------------------------------------

    # dp TaskGroup suffixes
    DP_EXTRACT_S = "extract_raw_data"
    DP_VALIDATE_S = "validate_data"
    DP_BUILD_S = "build_features"
    DP_STORE_S = "store_features"

    # deploy TaskGroup suffixes
    DEPLOY_SNAPSHOT_S = "snapshot_current"
    DEPLOY_MATERIALIZE_S = "materialize_repo"
    DEPLOY_TRITON_LOAD_S = "triton_load"
    DEPLOY_TRITON_READY_S = "triton_ready"
    DEPLOY_TRITON_SMOKE_S = "triton_infer_smoke"
