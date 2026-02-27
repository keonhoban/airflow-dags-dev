# dags/mlops_lib/core/ids.py
from __future__ import annotations

# -----------------------
# DAG / TaskGroups (SSOT)
# -----------------------
DAG_ID_E2E_FULL = "e2e_full"
TG_DP = "dp"
TG_DEPLOY = "deploy"

# -----------------------
# Task IDs (SSOT)
# -----------------------
# (TaskGroup 포함된 task_id는 "tg.task" 형태가 됩니다)
DP_EXTRACT_TASK_ID = f"{TG_DP}.extract_raw_data"
DP_VALIDATE_TASK_ID = f"{TG_DP}.validate_data"
DP_BUILD_TASK_ID = f"{TG_DP}.build_features"
DP_STORE_TASK_ID = f"{TG_DP}.store_features"

SUMMARIZE_TASK_ID = "summarize_run"

TRAIN_TASK_ID = "train_and_evaluate"
BRANCH_TASK_ID = "check_result"
REGISTER_TASK_ID = "register_model_task"

PROMOTION_START_TASK_ID = "promotion_start"
SHADOW_START_TASK_ID = "shadow_start"
NOTIFY_FAILURE_TASK_ID = "notify_failure"

SENSOR_MODEL_READY_TASK_ID = "check_model_ready"

# Deploy TaskGroup
DEPLOY_SNAPSHOT_TASK_ID = f"{TG_DEPLOY}.snapshot_current"
DEPLOY_MATERIALIZE_TASK_ID = f"{TG_DEPLOY}.materialize_repo"
DEPLOY_TRITON_LOAD_TASK_ID = f"{TG_DEPLOY}.triton_load"
DEPLOY_TRITON_READY_TASK_ID = f"{TG_DEPLOY}.triton_ready"
DEPLOY_TRITON_SMOKE_TASK_ID = f"{TG_DEPLOY}.triton_infer_smoke"

COMMIT_CURRENT_TASK_ID = "commit_current"
FASTAPI_RELOAD_TASK_ID = "fastapi_reload"
ROLLBACK_MINIMAL_TASK_ID = "rollback_minimal"

# -----------------------
# XCom keys (SSOT) - pipeline scope
# -----------------------
XCOM_ALIAS = "alias"
XCOM_MODEL_NAME = "model_name"
XCOM_ACCURACY = "accuracy"
XCOM_RUN_ID = "run_id"
XCOM_VERSION = "version"

XCOM_FS_FEATURE_URI = "fs_feature_uri"
XCOM_FS_VERSION = "fs_version"
XCOM_FS_SCHEMA_HASH = "fs_schema_hash"

XCOM_SHADOW_REASON = "shadow_reason"

# -----------------------
# Shadow reasons (SSOT)
# -----------------------
SHADOW_REASON_TRAIN_SKIPPED = "train_skipped"
SHADOW_REASON_ACCURACY_INVALID = "accuracy_invalid"
SHADOW_REASON_BELOW_THRESHOLD = "below_threshold"

# -----------------------
# Triton XCom keys (SSOT)
# -----------------------
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

# -----------------------
# Triton task ids (SSOT)
# -----------------------
TRITON_MAT_TASK_ID = DEPLOY_MATERIALIZE_TASK_ID
TRITON_SNAPSHOT_TASK_ID = DEPLOY_SNAPSHOT_TASK_ID
