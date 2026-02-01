# dags/pipelines/e2e_compact.py
from __future__ import annotations

from airflow.sdk import Variable
from airflow.exceptions import AirflowSkipException

from mlops_lib.dp.tasks import (
    task_extract_raw_data,
    task_validate_data,
    task_build_features,
    task_store_features,
    task_summarize_run as dp_summarize,
)

from ml_code.train_model import train_model, TrainSkippableError
from ml_code.register_model import register_model
from ml_code.sensor_model_ready import check_model_ready
from ml_code.triton_deploy import (
    snapshot_current,
    materialize,
    triton_load,
    triton_ready,
    triton_infer_smoke,
    commit_current,
    rollback_minimal,
)
from ml_code.trigger_reload import trigger_reload

from utils.slack_alerts import notify_info, notify_skip, notify_success, notify_fail


def _var(key: str, default=None):
    try:
        return Variable.get(key)
    except Exception:
        return default


# -----------------------
# DP tasks wrappers
# -----------------------
def dp_extract(**context):
    return task_extract_raw_data(**context)


def dp_validate(**context):
    return task_validate_data(**context)


def dp_build(**context):
    return task_build_features(**context)


def dp_store(**context):
    return task_store_features(**context)


# -----------------------
# Train / Branch
# -----------------------
def train_and_eval(ti, **_):
    """
    ✅ 제출/운영 기준 핵심:
    - run_id / alias / accuracy는 반드시 XCom으로 보장
    - 학습 불가 조건은 FAIL이 아니라 SKIP로 처리(=운영에서 정상 상황)
    """
    # 하이퍼파라미터(Variable)
    C = float(_var("logreg_C", 1.0))
    max_iter = int(_var("logreg_max_iter", 200))

    # DP 결과로부터 feature uri
    feature_uri = ti.xcom_pull(key="fs_feature_uri", task_ids="store_features")
    fs_version = ti.xcom_pull(key="fs_version", task_ids="store_features")
    schema_hash = ti.xcom_pull(key="fs_schema_hash", task_ids="build_features")

    # alias 선택(제출용: 고정 A)
    alias = str(_var("mlflow_alias", "A") or "A")

    if not feature_uri:
        raise RuntimeError("feature_uri missing from XCom (store_features)")

    try:
        acc, run_id = train_model(
            C=C,
            max_iter=max_iter,
            feature_uri=feature_uri,
            fs_version=fs_version,
            schema_hash=schema_hash,
        )
    except TrainSkippableError as e:
        # ✅ 운영 기준: 데이터 조건 불만족은 FAIL이 아니라 SKIP
        notify_skip("Train skipped", reason=str(e))
        ti.xcom_push(key="accuracy", value=None)
        ti.xcom_push(key="run_id", value=None)
        ti.xcom_push(key="alias", value=alias)
        raise AirflowSkipException(str(e))

    ti.xcom_push(key="accuracy", value=float(acc))
    ti.xcom_push(key="run_id", value=run_id)
    ti.xcom_push(key="alias", value=alias)

    notify_info(
        "Train completed",
        accuracy=f"{acc:.4f}",
        run_id=run_id,
        alias=alias,
        fs_version=fs_version,
        schema_hash=schema_hash,
    )
    return acc


def check_result(ti, **_):
    """
    Branch:
      - accuracy >= threshold: promo_start
      - else: shadow_start
    """
    thr = float(_var("accuracy_threshold", 0.5))
    acc = ti.xcom_pull(task_ids="train_and_eval", key="accuracy")

    # train이 SKIP이면 shadow로 가도 의미가 없으니 shadow_start로 보내고,
    # shadow materialize에서 다시 'run_id 없으면 SKIP' 처리
    if acc is None:
        return "shadow_start"

    if float(acc) >= thr:
        return "promo_start"
    return "shadow_start"


def notify_failure(**_):
    notify_skip("Accuracy below threshold", next_action="features/label/model params 개선")


# -----------------------
# Promotion path: register + sensor
# -----------------------
def register_model_task(ti, **_):
    run_id = ti.xcom_pull(task_ids="train_and_eval", key="run_id")
    alias = ti.xcom_pull(task_ids="train_and_eval", key="alias") or "A"
    model_name = _var("triton_model_name", _var("model_name", "best_model"))

    if not run_id:
        raise AirflowSkipException("run_id missing -> skip register")

    version = register_model(run_id=run_id, model_name=model_name, mlflow_alias=alias)
    ti.xcom_push(key="version", value=int(version))
    notify_success("Model registered", model=model_name, version=str(version), alias=alias)
    return version


def sensor_ready_func(ti, **_):
    model_name = _var("triton_model_name", _var("model_name", "best_model"))
    version = ti.xcom_pull(task_ids="register_model_task", key="version")
    if not version:
        return True  # shadow path에서는 sensor 의미 없음
    return check_model_ready(model_name=model_name, version=str(version))


# -----------------------
# Triton deploy tasks
# -----------------------
def snapshot_current(ti, **_):
    return snapshot_current(ti=ti)


def triton_materialize_promo(ti, **_):
    """
    promo: alias 기반 materialize
    - register를 넣고 싶으면 DAG에서 register_model_task + sensor를 붙이면 됨
    - 제출용 최소 버전에서는 alias만으로도 충분히 설명 가능
    """
    alias = ti.xcom_pull(task_ids="train_and_eval", key="alias") or "A"
    notify_info("Triton materialize (promo)", alias=alias)
    return materialize(ti=ti, alias=alias, shadow=False)


def triton_materialize_shadow(ti, **_):
    """
    ✅ 핵심: run_id가 없으면 FAIL이 아니라 SKIP (제출/운영 기준)
    """
    alias = ti.xcom_pull(task_ids="train_and_eval", key="alias") or "A"
    run_id = ti.xcom_pull(task_ids="train_and_eval", key="run_id")

    if not run_id:
        raise AirflowSkipException("run_id missing -> skip shadow materialize")

    notify_info("Triton materialize (shadow)", alias=alias, run_id=run_id)
    return materialize(ti=ti, alias=alias, run_id=run_id, shadow=True)


def triton_load(ti, **_):
    return triton_load(ti=ti)


def triton_ready(ti, **_):
    return triton_ready(ti=ti)


def triton_infer_smoke(ti, **_):
    return triton_infer_smoke(ti=ti)


def commit_current(ti, **_):
    return commit_current(ti=ti)


def rollback_minimal(ti, **_):
    return rollback_minimal(ti=ti)


# -----------------------
# FastAPI reload
# -----------------------
def fastapi_reload(ti, **_):
    alias = ti.xcom_pull(task_ids="train_and_eval", key="alias") or "A"
    trigger_reload(alias)
    notify_success("FastAPI reload completed", alias=alias)


# -----------------------
# Summary
# -----------------------
def summarize_run(**context):
    # DP 요약 + (원하면) 더 많은 정보 추가 가능
    dp_summarize(**context)
    notify_info("E2E compact finished (summary)", note="Check task logs for details")

