# dags/pipelines/full_e2e.py
from __future__ import annotations

from airflow.models import Variable  # classic Variable API
from airflow.utils.log.logging_mixin import LoggingMixin
from airflow.exceptions import AirflowSkipException

from mlflow.tracking import MlflowClient

from utils.slack_alerts import (
    send_slack_alert,
    notify_skip,
    notify_info,
    notify_success,
    notify_fail,
)

# DP tasks
from mlops_lib.dp.tasks import (
    task_extract_raw_data,
    task_validate_data,
    task_build_features,
    task_store_features,
    task_summarize_run,
)

# ML
from ml_code.train_model import train_model, TrainSkippableError
from ml_code.register_model import register_model
from ml_code.rollback_model import rollback_model
from ml_code.sensor_model_ready import check_model_ready

# Triton
from ml_code.triton_deploy import (
    snapshot_current,
    materialize,
    triton_load,
    triton_ready,
    triton_infer_smoke,
    commit_current,
    rollback_minimal,
)

# FastAPI reload
from ml_code.trigger_reload import trigger_reload


log = LoggingMixin().log


# -----------------------
# Helpers
# -----------------------
def get_param(key, default, cast_func, validate_func=None):
    """
    Variable을 환경별로 튜닝 가능하게 가져오되,
    잘못된 값이면 Slack에 남기고 기본값 사용
    """
    try:
        raw = Variable.get(key, default_var=str(default))
        v = cast_func(raw)
        if validate_func and not validate_func(v):
            raise ValueError("Validation failed")
        return v
    except Exception as e:
        send_slack_alert(f"[Param] {key} 로딩 실패: {e} → 기본값 {default} 사용")
        return default


def get_version_by_alias(model_name, alias):
    try:
        return MlflowClient().get_model_version_by_alias(model_name, alias).version
    except Exception:
        return None


def get_env():
    return Variable.get("triton_env", default_var="dev")


# -----------------------
# DP wrappers (keep thin)
# -----------------------
def dp_extract(**context):
    return task_extract_raw_data(**context)


def dp_validate(**context):
    return task_validate_data(**context)


def dp_build(**context):
    return task_build_features(**context)


def dp_store(**context):
    return task_store_features(**context)


def dp_summary(**context):
    return task_summarize_run(**context)


# -----------------------
# Train/Register
# -----------------------
def train_and_evaluate(ti, **_):
    # 정책 변수(환경별 튜닝)
    min_rows = get_param("train_min_rows", 20, int, lambda x: x >= 2)

    C = get_param("logreg_C", 1.0, float, lambda x: 0.001 <= x <= 10.0)
    max_iter = get_param("logreg_max_iter", 200, int, lambda x: x > 50)
    threshold = get_param("accuracy_threshold", 0.9, float, lambda x: 0.5 <= x <= 0.99)

    model_name = Variable.get("model_name")  # 필수
    alias = Variable.get("mlflow_alias")     # 필수
    env = get_env()

    if not (model_name and alias):
        raise ValueError("필수 Variable 누락: model_name 또는 mlflow_alias")

    feature_uri = ti.xcom_pull(task_ids="store_features", key="fs_feature_uri")
    fs_version = ti.xcom_pull(task_ids="store_features", key="fs_version")
    schema_hash = ti.xcom_pull(task_ids="build_features", key="fs_schema_hash")
    rows = (
        ti.xcom_pull(task_ids="build_features", key="fs_feature_rows")
        or ti.xcom_pull(task_ids="store_features", key="fs_feature_rows")
    )

    if not feature_uri:
        raise ValueError("feature_uri 없음 → store_features 결과 확인 필요")

    # 1) 표본 부족은 실패가 아니라 정상 스킵
    if rows is not None and int(rows) < int(min_rows):
        notify_skip(
            "Train skipped: not enough rows",
            env=env,
            model=model_name,
            alias=alias,
            rows=str(rows),
            min_rows=str(min_rows),
            feature_uri=feature_uri,
            next_action="raw data 늘리거나 train_min_rows 낮추기",
        )
        raise AirflowSkipException(f"not enough rows: {rows} < {min_rows}")

    # 2) 학습 시도 (학습 불가능 조건이면 TrainSkippableError로 처리)
    try:
        acc, run_id = train_model(
            C=C,
            max_iter=max_iter,
            feature_uri=feature_uri,
            fs_version=fs_version,
            schema_hash=schema_hash,
        )
    except TrainSkippableError as e:
        notify_skip(
            "Train skipped: data not trainable",
            env=env,
            model=model_name,
            alias=alias,
            rows=str(rows),
            min_rows=str(min_rows),
            feature_uri=feature_uri,
            reason=str(e),
            next_action="feature 분산/label 정책 확인 또는 raw data 품질 개선",
        )
        raise AirflowSkipException(str(e))
    except ValueError as e:
        msg = str(e)
        if "at least 2 classes" in msg:
            notify_skip(
                "Train skipped: single class",
                env=env,
                model=model_name,
                alias=alias,
                rows=str(rows),
                min_rows=str(min_rows),
                feature_uri=feature_uri,
                reason=msg,
                next_action="label 생성 정책(qcut) 또는 raw 데이터 분산 개선",
            )
            raise AirflowSkipException(msg)
        raise

    # XCom
    ti.xcom_push(key="run_id", value=run_id)
    ti.xcom_push(key="model_name", value=model_name)
    ti.xcom_push(key="alias", value=alias)
    ti.xcom_push(key="acc", value=acc)
    ti.xcom_push(key="threshold", value=threshold)

    # (선택) shadow smoke에서 참고용으로 남겨두기
    ti.xcom_push(key="feature_uri", value=feature_uri)

    notify_success(
        "Train completed",
        env=env,
        model=model_name,
        alias=alias,
        acc=f"{acc:.4f}",
        threshold=str(threshold),
        rows=str(rows),
        feature_uri=feature_uri,
        run_id=run_id,
    )


def check_result(ti, **_):
    acc = ti.xcom_pull(task_ids="train_and_evaluate", key="acc")
    threshold = ti.xcom_pull(task_ids="train_and_evaluate", key="threshold")

    if acc is None or threshold is None:
        send_slack_alert("⏭️ check_result: train이 SKIP 되었거나 XCom 누락 → 분기 스킵")
        raise AirflowSkipException()

    # ✅ 성능 통과면 promotion(등록) / 미달이면 shadow로 끝까지 검증
    return "register_model_task" if acc >= threshold else "shadow_start"


def register_model_task(ti, **_):
    run_id = ti.xcom_pull(task_ids="train_and_evaluate", key="run_id")
    model_name = ti.xcom_pull(task_ids="train_and_evaluate", key="model_name")
    alias = ti.xcom_pull(task_ids="train_and_evaluate", key="alias")
    env = get_env()

    prev_version = get_version_by_alias(model_name, alias)

    try:
        version = register_model(run_id, model_name, alias)
        ti.xcom_push(key="version", value=version)
        notify_success("Model registered", env=env, model=model_name, version=str(version), alias=alias)
    except Exception as e:
        msg = f"모델 등록 실패: {e}"
        if prev_version:
            rollback_model(model_name, prev_version, alias)
            msg += f" → 롤백 완료: v{prev_version}"
        else:
            msg += " → 롤백 생략"
        notify_fail("Model register failed", env=env, model=model_name, alias=alias, reason=msg)
        raise


def sensor_ready_func(ti, **_):
    model_name = ti.xcom_pull(task_ids="train_and_evaluate", key="model_name")
    version = ti.xcom_pull(task_ids="register_model_task", key="version")
    return check_model_ready(model_name, version)


# -----------------------
# Triton deploy
# -----------------------
def triton_materialize_task(ti, **_):
    """
    ✅ 실무형:
    - register 성공(=version 존재) -> Promotion 경로: alias 기준 materialize
    - register 스킵/미실행 -> Shadow 경로: run_id 기준 materialize (pipeline 끝까지 검증)
    """
    alias = ti.xcom_pull(task_ids="train_and_evaluate", key="alias")
    run_id = ti.xcom_pull(task_ids="train_and_evaluate", key="run_id")
    env = get_env()

    version = ti.xcom_pull(task_ids="register_model_task", key="version")

    if version:
        notify_info(
            "Triton deploy: promotion path (alias)",
            env=env,
            alias=alias,
            version=str(version),
        )
        materialize(ti=ti, alias=alias)
        return

    # shadow
    if not run_id:
        raise ValueError("Shadow deploy 불가: run_id XCom 누락 (train_and_evaluate 확인 필요)")

    notify_info(
        "Triton deploy: shadow path (run_id)",
        env=env,
        alias=alias,
        run_id=run_id,
    )
    materialize(ti=ti, alias=alias, run_id=run_id, shadow=True)


def triton_rollback_task(ti, **_):
    rollback_minimal(ti=ti)


# -----------------------
# FastAPI reload (optional)
# -----------------------
def fastapi_reload_task(ti, **_):
    alias = ti.xcom_pull(task_ids="train_and_evaluate", key="alias")
    env = get_env()
    trigger_reload(alias)
    notify_success("FastAPI reload completed", env=env, alias=alias)


def notify_failure():
    """
    accuracy 미달 시:
    - 프로모션(register)은 스킵
    - 하지만 (구성에 따라) shadow deploy + smoke까지는 여전히 수행 가능
    """
    env = get_env()
    notify_skip("Accuracy below threshold", env=env, next_action="특성/라벨/모델 파라미터 개선")
