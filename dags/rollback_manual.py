# dags/rollback_manual.py
from __future__ import annotations

from datetime import datetime
import pendulum

from airflow import DAG
from airflow.models import Variable
from airflow.operators.python import PythonOperator

from ml_code.triton_deploy import rollback_manual as triton_rollback_manual
from ml_code.trigger_reload import trigger_reload
from utils.slack_alerts import alert_slack

KST = pendulum.timezone("Asia/Seoul")


def _get_var(key: str, default: str = "") -> str:
    try:
        return (Variable.get(key) or default).strip()
    except Exception:
        return default


def _get_bool(key: str, default: str = "false") -> bool:
    v = (_get_var(key, default) or "").strip().lower()
    return v in ("1", "true", "yes", "y", "on")


def _run():
    # Airflow Variables (UI에서 즉시 제어 가능)
    model = _get_var("rollback_model_name", "")
    dv_raw = _get_var("rollback_deploy_version", "")
    deploy_version = int(dv_raw) if dv_raw else None

    # 1) Triton rollback (SSOT)
    triton_rollback_manual(model=model or None, deploy_version=deploy_version)

    # 2) FastAPI 동기화 (옵션 토글)
    # - rollback_fastapi_reload=false면 "Triton만 SSOT"로 끝 (서비스 영향 최소)
    if _get_bool("rollback_fastapi_reload", "false"):
        variant = _get_var("rollback_fastapi_variant", "A") or "A"
        trigger_reload(variant, deploy_version=deploy_version)


with DAG(
    dag_id="core_rollback_manual",
    start_date=datetime(2026, 2, 1, tzinfo=KST),
    schedule=None,
    catchup=False,
    tags=["rollback", "triton", "ops"],
    on_failure_callback=alert_slack,
) as dag:
    PythonOperator(task_id="rollback_manual", python_callable=_run)

