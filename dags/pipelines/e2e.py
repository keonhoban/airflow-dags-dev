# dags/pipelines/e2e.py
from __future__ import annotations

from airflow.exceptions import AirflowSkipException
from airflow.utils.log.logging_mixin import LoggingMixin

# ML 단계 코드 (이미 건호님이 갖고 있는 모듈)
from ml_code.train_model import train_model, TrainSkippableError
from ml_code.register_model import register_model
from ml_code.sensor_model_ready import check_model_ready

# Triton 배포 코드 (건호님 현재 구조 기준)
from ml_code.triton_deploy import snapshot_current, materialize, triton_load, triton_ready, triton_infer_smoke, commit_current
from utils.slack_alerts import notify_info, notify_success, notify_skip

# (선택) Variable 읽기
try:
    from airflow.sdk import Variable
except Exception:
    from airflow.models import Variable


log = LoggingMixin().log


def _var(key: str, default=None):
    try:
        return Variable.get(key)
    except Exception:
        return default


# -----------------------
# Train
# -----------------------
def task_train_and_eval(ti, **_):
    """
    ✅ 반드시 XCom: run_id, alias, accuracy 를 남긴다.
    - 학습 자체가 불가능하면 SKIP 처리 (shadow/materialize도 자연스럽게 스킵)
    """
    # 제출용 최소: 변수들 없으면 기본값으로 동작
    C = float(_var("logreg_C", "1.0"))
    max_iter = int(_var("logreg_max_iter", "200"))
    alias = str(_var("mlflow_alias", "A"))

    # DP에서 만든 feature_uri
    feature_uri = ti.xcom_pull(key="fs_feature_uri", task_ids="store_features")
    fs_version = ti.xcom_pull(key="fs_version", task_ids="store_features")
    schema_hash = ti.xcom_pull(key="fs_schema_hash", task_ids="build_features")

    try:
        acc, run_id = train_model(
            C=C,
            max_iter=max_iter,
            feature_uri=feature_uri,
            fs_version=fs_version,
            schema_hash=schema_hash,
        )
    except TrainSkippableError as e:
        # ✅ 운영 기준: 학습 불가 조건은 "실패"가 아니라 "스킵" 처리
        notify_skip("Train skipped (not enough data/classes)", reason=str(e))
        raise AirflowSkipException(str(e))

    # ✅ 여기서 핵심: downstream이 key="run_id"/"alias"로 pull하므로 반드시 push
    ti.xcom_push(key="run_id", value=run_id)
    ti.xcom_push(key="alias", value=alias)
    ti.xcom_push(key="accuracy", value=float(acc))

    log.info("[E2E] train OK acc=%.4f run_id=%s alias=%s", acc, run_id, alias)
    return float(acc)


# -----------------------
# Register (promotion path)
# -----------------------
def task_register_model(ti, **_):
    """
    promotion path: MLflow Model Registry 등록 + alias 세팅
    """
    run_id = ti.xcom_pull(task_ids="train_and_eval", key="run_id")
    alias = ti.xcom_pull(task_ids="train_and_eval", key="alias") or str(_var("mlflow_alias", "A"))
    model_name = str(_var("model_name", "best_model"))

    if not run_id:
        raise AirflowSkipException("register_model skipped: run_id missing")

    version = register_model(run_id=run_id, model_name=model_name, mlflow_alias=alias)
    ti.xcom_push(key="version", value=str(version))
    notify_info("MLflow register completed", model=model_name, alias=alias, version=str(version))
    return str(version)


def task_wait_model_ready(ti, **_):
    model_name = str(_var("model_name", "best_model"))
    version = ti.xcom_pull(task_ids="register_model", key="version")

    if not version:
        raise AirflowSkipException("model_ready skipped: version missing")

    return check_model_ready(model_name=model_name, version=str(version))


# -----------------------
# Triton deploy
# -----------------------
def task_snapshot_current(ti, **_):
    snapshot_current(ti=ti)


def task_triton_materialize_shadow(ti, **_):
    """
    ✅ shadow 배포는 run_id가 없으면 실패가 아니라 스킵이 맞다.
    """
    alias = ti.xcom_pull(task_ids="train_and_eval", key="alias") or str(_var("mlflow_alias", "A"))
    run_id = ti.xcom_pull(task_ids="train_and_eval", key="run_id")

    if not run_id:
        raise AirflowSkipException("shadow materialize skipped: run_id missing")

    notify_info("Triton materialize (shadow)", alias=alias, run_id=run_id)
    materialize(ti=ti, alias=alias, run_id=run_id, shadow=True)


def task_triton_materialize_promotion(ti, **_):
    """
    promotion 배포: alias 기준으로 Registry version 선택
    """
    alias = ti.xcom_pull(task_ids="train_and_eval", key="alias") or str(_var("mlflow_alias", "A"))
    notify_info("Triton materialize (promotion)", alias=alias)
    materialize(ti=ti, alias=alias)


def task_triton_load(ti, **_):
    triton_load(ti=ti)


def task_triton_ready(ti, **_):
    triton_ready(ti=ti)


def task_triton_smoke(ti, **_):
    triton_infer_smoke(ti=ti)


def task_commit_current(ti, **_):
    commit_current(ti=ti)
    alias = ti.xcom_pull(task_ids="materialize_repo", key="alias")
    ver = ti.xcom_pull(task_ids="materialize_repo", key="deploy_version")
    notify_success("Triton deploy committed", alias=str(alias), version=str(ver))

