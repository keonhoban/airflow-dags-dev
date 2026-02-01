# dags/pipelines/e2e.py
from __future__ import annotations

from airflow.utils.log.logging_mixin import LoggingMixin

from ml_code.config import cfg, get_env
from ml_code.train import train_model_and_log
from ml_code.register import register_model_and_set_alias
from ml_code.sensor import is_model_ready
from ml_code.triton import (
    snapshot_current,
    materialize_shadow_from_run,
    triton_load,
    triton_ready,
    triton_infer_smoke,
    rollback_minimal,
    commit_current,
)
from ml_code.fastapi import trigger_reload
from utils.slack import notify_info, notify_success, notify_fail, notify_skip

log = LoggingMixin().log


def task_train_and_eval(ti, **_):
    """
    - train + mlflow log + onnx artifact
    - run_id, alias(고정) 등을 XCom으로 남김
    """
    env = get_env()
    alias = cfg("mlflow_alias", "A")  # dev/prod 변수로 제어

    acc, run_id, n_features = train_model_and_log(ti=ti)

    ti.xcom_push(key="accuracy", value=float(acc))
    ti.xcom_push(key="run_id", value=run_id)
    ti.xcom_push(key="alias", value=alias)
    ti.xcom_push(key="n_features", value=int(n_features))

    notify_info("Train completed", env=env, run_id=run_id, acc=f"{acc:.4f}", alias=alias)
    return acc


def task_snapshot_current(ti, **_):
    snapshot_current(ti=ti)


def task_triton_materialize_shadow(ti, **_):
    """
    ✅ 항상 shadow deploy
    - 레지스트리/alias 상관 없이 run_id로 모델 repo에 materialize
    - 배포 검증의 기준점을 고정
    """
    run_id = ti.xcom_pull(task_ids="train_and_eval", key="run_id")
    alias = ti.xcom_pull(task_ids="train_and_eval", key="alias")
    env = get_env()

    if not run_id:
        raise RuntimeError("run_id missing from XCom (train_and_eval)")

    notify_info("Triton deploy (shadow)", env=env, alias=alias, run_id=run_id)
    materialize_shadow_from_run(ti=ti, run_id=run_id, alias=alias)


def task_triton_load(ti, **_):
    triton_load(ti=ti)


def task_triton_ready(ti, **_):
    triton_ready(ti=ti)


def task_triton_infer_smoke(ti, **_):
    triton_infer_smoke(ti=ti)
    env = get_env()
    model = cfg("triton_model_name", required=True)
    ver = ti.xcom_pull(task_ids="triton_materialize_shadow", key="deploy_version")
    notify_success("Triton smoke OK", env=env, model=model, version=str(ver))


def task_triton_rollback(ti, **_):
    """
    ✅ 배포 구간 실패시에만 호출되도록 DAG에서 연결
    """
    env = get_env()
    try:
        rollback_minimal(ti=ti)
        notify_fail("Rollback executed", env=env)
    except Exception as e:
        notify_fail("Rollback failed", env=env, error=str(e))
        raise


def task_gate_promotion(ti, **_):
    """
    ShortCircuitOperator:
      - True  -> promotion 진행
      - False -> promotion 스킵 (하지만 shadow 배포 검증은 이미 완료)
    """
    env = get_env()
    threshold = float(cfg("accuracy_threshold", "0.70"))
    acc = float(ti.xcom_pull(task_ids="train_and_eval", key="accuracy") or 0.0)

    if acc >= threshold:
        notify_info("Promotion gate passed", env=env, acc=f"{acc:.4f}", threshold=str(threshold))
        return True

    notify_skip("Promotion skipped (below threshold)", env=env, acc=f"{acc:.4f}", threshold=str(threshold))
    return False


def task_register_if_promoted(ti, **_):
    """
    조건 통과 시:
      - model version 생성
      - alias 갱신
      - version XCom push
    """
    env = get_env()
    run_id = ti.xcom_pull(task_ids="train_and_eval", key="run_id")
    alias = ti.xcom_pull(task_ids="train_and_eval", key="alias")
    model_name = cfg("model_name", required=True)

    version = register_model_and_set_alias(run_id=run_id, model_name=model_name, alias=alias)
    ti.xcom_push(key="version", value=int(version))
    notify_success("MLflow registry updated", env=env, model=model_name, alias=alias, version=str(version))


def task_wait_model_ready_if_promoted(ti, **_):
    model_name = cfg("model_name", required=True)
    version = ti.xcom_pull(task_ids="register_model", key="version")
    if not version:
        raise RuntimeError("version missing from XCom (register_model)")
    return is_model_ready(model_name=model_name, version=str(version))


def task_commit_current_if_promoted(ti, **_):
    """
    ✅ 여기서부터는 '운영 상태 기록'
    - 실패해도 rollback 대상 아님(배포 검증은 이미 끝)
    """
    env = get_env()
    commit_current(ti=ti)
    notify_success("current.json committed", env=env)


def task_fastapi_reload_if_promoted(ti, **_):
    env = get_env()
    alias = ti.xcom_pull(task_ids="train_and_eval", key="alias") or "A"
    trigger_reload(alias=alias)
    notify_success("FastAPI reload completed", env=env, alias=alias)

