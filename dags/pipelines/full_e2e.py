# dags/pipelines/full_e2e.py
from __future__ import annotations

import os
from airflow.utils.log.logging_mixin import LoggingMixin

from utils.slack_alerts import (
    notify_info,
    notify_success,
    notify_fail,
    notify_skip,
)

# -----------------------
# DP / Train / Register / Sensor / Reload
# -----------------------
from mlops_lib.dp.tasks import (
    task_extract_raw_data as dp_extract_impl,
    task_validate_data as dp_validate_impl,
    task_build_features as dp_build_impl,
    task_store_features as dp_store_impl,
    task_summarize_run as dp_summary_impl,
)

from ml_code.train_model import train_model as train_model_impl, TrainSkippableError
from ml_code.register_model import register_model as register_model_impl
from ml_code.sensor_model_ready import check_model_ready as check_model_ready_impl
from ml_code.trigger_reload import trigger_reload as trigger_reload_impl

# -----------------------
# Triton deploy (IMPLEMENTATIONS)  ✅ 반드시 alias로 import
# -----------------------
from ml_code.triton_deploy import (
    snapshot_current as triton_snapshot_current_impl,
    materialize as triton_materialize_impl,
    triton_load as triton_load_impl,
    triton_ready as triton_ready_impl,
    triton_infer_smoke as triton_smoke_impl,
    commit_current as triton_commit_current_impl,
    rollback_minimal as triton_rollback_minimal_impl,
)

log = LoggingMixin().log

# -----------------------
# Helpers: config / vars
# -----------------------
def _get_env() -> str:
    # Airflow Variable에 triton_env 를 이미 쓰고 계심
    # 없으면 dev 로 fallback
    from airflow.models import Variable

    try:
        return Variable.get("triton_env")
    except Exception:
        return os.getenv("TRITON_ENV", "dev")


def _get_var(key: str, default=None):
    from airflow.models import Variable

    try:
        return Variable.get(key)
    except Exception:
        return default


def _as_float(x, default=0.0):
    try:
        return float(x)
    except Exception:
        return default


def _as_int(x, default=0):
    try:
        return int(x)
    except Exception:
        return default


# -----------------------
# Feast repo path resolution (BashOperator 실패 방지)
# -----------------------
def resolve_feast_repo_path() -> str:
    """
    Airflow GitSync 구조가 바뀌어도 안전하게 Feast repo 경로를 잡습니다.

    우선순위:
    1) ENV FEAST_REPO_PATH
    2) Airflow Variable feast_repo_path
    3) 잘 알려진 후보 경로들 중 존재하는 것
    """
    env_path = os.getenv("FEAST_REPO_PATH")
    if env_path and os.path.isdir(env_path):
        return env_path

    var_path = _get_var("feast_repo_path", None)
    if var_path and os.path.isdir(var_path):
        return var_path

    candidates = [
        # (기존 하드코딩 경로)
        "/opt/airflow/dags/repo/dags/feast_repo",
        # (최근 실제 구조: /opt/airflow/dags/repo/dags/e2e 아래로 옮겼을 가능성)
        "/opt/airflow/dags/repo/dags/e2e/feast_repo",
        "/opt/airflow/dags/repo/dags/e2e/feast",
        "/opt/airflow/dags/repo/dags/feast",
        # (혹시 dags root에 둘 경우)
        "/opt/airflow/dags/feast_repo",
        "/opt/airflow/dags/feast",
    ]

    for p in candidates:
        if os.path.isdir(p):
            return p

    # 마지막: 친절한 에러
    raise FileNotFoundError(
        "[FEAST] repo path not found. "
        "Set FEAST_REPO_PATH env or airflow variable 'feast_repo_path'."
    )


# -----------------------
# Data Pipeline tasks (wrappers)
# -----------------------
def dp_extract(**context):
    return dp_extract_impl(**context)


def dp_validate(**context):
    return dp_validate_impl(**context)


def dp_build(**context):
    return dp_build_impl(**context)


def dp_store(**context):
    return dp_store_impl(**context)


def dp_summary(**context):
    return dp_summary_impl(**context)


# -----------------------
# Train / Evaluate (single task wrapper)
# -----------------------
def train_and_evaluate(**context):
    """
    ✅ XCom outputs:
    - accuracy
    - run_id
    - alias
    - fs_version
    - schema_hash
    """
    ti = context["ti"]
    env = _get_env()

    # training hyperparams
    C = _as_float(_get_var("logreg_C", "1.0"), 1.0)
    max_iter = _as_int(_get_var("logreg_max_iter", "200"), 200)

    # feature_uri from DP store_features
    feature_uri = ti.xcom_pull(task_ids="store_features", key="fs_feature_uri")
    fs_version = ti.xcom_pull(task_ids="store_features", key="fs_version")
    schema_hash = ti.xcom_pull(task_ids="build_features", key="fs_schema_hash")

    # alias selection (A/B)
    alias = _get_var("mlflow_alias", "A")

    if not feature_uri:
        raise ValueError("[TRAIN] feature_uri missing. check store_features XCom (fs_feature_uri)")

    try:
        acc, run_id = train_model_impl(
            C=C,
            max_iter=max_iter,
            feature_uri=feature_uri,
            fs_version=fs_version,
            schema_hash=schema_hash,
        )
    except TrainSkippableError as e:
        # 스킵성 실패는 운영적으로 "fail"이라기보다 "skip" 처리할 수 있음
        notify_skip("Train skipped", env=env, reason=str(e))
        # Branch에서 shadow로 흘릴지, 즉시 중단할지 정책 선택 가능
        # 여기서는 "shadow deploy 불가"로 처리하기 위해 run_id 없이 반환
        ti.xcom_push(key="accuracy", value=0.0)
        ti.xcom_push(key="run_id", value=None)
        ti.xcom_push(key="alias", value=alias)
        ti.xcom_push(key="fs_version", value=fs_version)
        ti.xcom_push(key="schema_hash", value=schema_hash)
        return

    ti.xcom_push(key="accuracy", value=float(acc))
    ti.xcom_push(key="run_id", value=run_id)
    ti.xcom_push(key="alias", value=alias)
    ti.xcom_push(key="fs_version", value=fs_version)
    ti.xcom_push(key="schema_hash", value=schema_hash)

    notify_info(
        "Train completed",
        env=env,
        accuracy=f"{acc:.4f}",
        alias=alias,
        run_id=run_id,
        fs_version=fs_version,
        schema_hash=schema_hash,
    )


# -----------------------
# Branch decision
# -----------------------
def check_result(**context):
    """
    ✅ BranchPythonOperator 용
    - success(>=threshold) => register_model_task
    - fail(<threshold)     => shadow_start
    """
    ti = context["ti"]
    env = _get_env()

    acc = _as_float(ti.xcom_pull(task_ids="train_and_evaluate", key="accuracy"), 0.0)
    threshold = _as_float(_get_var("accuracy_threshold", "0.9"), 0.9)

    if acc >= threshold:
        notify_info("Branch: promotion (above threshold)", env=env, accuracy=f"{acc:.4f}", threshold=str(threshold))
        return "register_model_task"

    notify_info("Branch: shadow (below threshold)", env=env, accuracy=f"{acc:.4f}", threshold=str(threshold))
    return "shadow_start"


# -----------------------
# Register model (promotion only)
# -----------------------
def register_model_task(**context):
    """
    ✅ XCom outputs:
    - version
    """
    ti = context["ti"]
    env = _get_env()

    run_id = ti.xcom_pull(task_ids="train_and_evaluate", key="run_id")
    alias = ti.xcom_pull(task_ids="train_and_evaluate", key="alias") or _get_var("mlflow_alias", "A")

    model_name = _get_var("model_name", None) or _get_var("triton_model_name", None) or os.getenv("MODEL_NAME", "best_model")
    if not run_id:
        raise ValueError("[REGISTER] missing run_id from train_and_evaluate")

    version = register_model_impl(run_id=run_id, model_name=model_name, mlflow_alias=alias)
    ti.xcom_push(key="version", value=int(version))

    notify_success("MLflow register completed", env=env, model=model_name, alias=alias, version=str(version))


# -----------------------
# Sensor: wait model READY
# -----------------------
def sensor_ready_func(**context):
    ti = context["ti"]
    model_name = _get_var("model_name", None) or _get_var("triton_model_name", None) or os.getenv("MODEL_NAME", "best_model")

    version = ti.xcom_pull(task_ids="register_model_task", key="version")
    if not version:
        # promotion path only — but sensor는 promotion_start 뒤에만 있으니 보통 여기 오지 않음
        return False

    return check_model_ready_impl(model_name=model_name, version=str(version))


# -----------------------
# Shadow notification (optional)
# -----------------------
def notify_failure(**context):
    """
    accuracy 미달 시:
    - 프로모션(register)은 스킵
    - shadow deploy + smoke까지는 수행 가능
    """
    env = _get_env()
    notify_skip("Accuracy below threshold", env=env, next_action="feature/label/model 개선 후 재시도")


# -----------------------
# Triton deploy wrappers  ✅ recursion 방지 규칙 적용
# -----------------------
def snapshot_current(**context):
    ti = context["ti"]
    return triton_snapshot_current_impl(ti=ti)


def triton_materialize_task(**context):
    """
    full_e2e.py에서 보여주신 구조 그대로 유지:
    - promotion: register_model_task 에서 version이 있으면 alias 기반 materialize
    - shadow: version 없으면 run_id 기반 materialize(shadow=True)
    """
    ti = context["ti"]
    env = _get_env()

    alias = ti.xcom_pull(task_ids="train_and_evaluate", key="alias") or _get_var("mlflow_alias", "A")
    run_id = ti.xcom_pull(task_ids="train_and_evaluate", key="run_id")
    version = ti.xcom_pull(task_ids="register_model_task", key="version")

    # promotion
    if version:
        notify_info(
            "Triton deploy: promotion path (alias)",
            env=env,
            alias=alias,
            version=str(version),
        )
        return triton_materialize_impl(ti=ti, alias=alias)

    # shadow
    if not run_id:
        raise ValueError("Shadow deploy 불가: run_id XCom 누락 (train_and_evaluate 확인 필요)")

    notify_info(
        "Triton deploy: shadow path (run_id)",
        env=env,
        alias=alias,
        run_id=run_id,
    )
    return triton_materialize_impl(ti=ti, alias=alias, run_id=run_id, shadow=True)


def triton_load(**context):
    ti = context["ti"]
    return triton_load_impl(ti=ti)


def triton_ready(**context):
    ti = context["ti"]
    return triton_ready_impl(ti=ti)


def triton_infer_smoke(**context):
    ti = context["ti"]
    return triton_smoke_impl(ti=ti)


def commit_current(**context):
    ti = context["ti"]
    return triton_commit_current_impl(ti=ti)


def triton_rollback_task(**context):
    ti = context["ti"]
    return triton_rollback_minimal_impl(ti=ti)


# -----------------------
# FastAPI reload wrapper
# -----------------------
def fastapi_reload_task(**context):
    ti = context["ti"]
    env = _get_env()

    alias = ti.xcom_pull(task_ids="train_and_evaluate", key="alias") or _get_var("mlflow_alias", "A")
    trigger_reload_impl(alias)
    notify_success("FastAPI reload completed", env=env, alias=alias)


# -----------------------
# Feast commands (if you want BashOperator to call Python instead)
# (선택) BashOperator 유지가 아니라 PythonOperator로 바꾸고 싶을 때 사용
# -----------------------
def feast_apply_py(**context):
    """
    BashOperator가 경로 문제로 자주 깨지면,
    PythonOperator로 바꾸고 여기서 subprocess로 실행하세요.
    """
    import subprocess

    repo = resolve_feast_repo_path()
    log.info("[FEAST] apply repo=%s", repo)
    subprocess.run(["bash", "-lc", f"set -euo pipefail; cd {repo}; feast apply"], check=True)


def feast_materialize_py(**context):
    import subprocess

    repo = resolve_feast_repo_path()
    # ds / macros는 Python에서 직접 접근하기보다,
    # BashOperator에서 jinja를 쓰는 게 더 깔끔합니다.
    # 필요하면 execution_date 기반으로 계산하도록 확장 가능합니다.
    raise NotImplementedError("Prefer BashOperator for ds macros or implement execution_date window here.")

