# dags/rollback_manual.py
from __future__ import annotations

from datetime import datetime
import pendulum

from airflow import DAG
from airflow.models import Variable
from airflow.operators.python import PythonOperator

from ml_code.triton_deploy import rollback_manual
from ml_code.trigger_reload import trigger_reload
from utils.slack_alerts import alert_slack

KST = pendulum.timezone("Asia/Seoul")


def _get_var(key: str, default: str = "") -> str:
    try:
        return (Variable.get(key) or default).strip()
    except Exception:
        return default


def _as_bool(raw: str) -> bool:
    return (raw or "").strip().lower() in {"1", "true", "yes", "y", "on"}


def _run():
    """
    Variables:
      - rollback_model_name (optional)
      - rollback_deploy_version (optional int)
      - rollback_fastapi_reload (optional bool, default false)
      - rollback_fastapi_variant (optional, default A)
    """
    model = _get_var("rollback_model_name", "")
    dv_raw = _get_var("rollback_deploy_version", "")
    deploy_version = int(dv_raw) if dv_raw else None

    # ✅ Triton rollback (current.json + version_policy + unload/load)
    rollback_manual(model=model or None, deploy_version=deploy_version)

    # ✅ 기본 OFF (원하면 Variable로 켜기)
    if _as_bool(_get_var("rollback_fastapi_reload", "false")):
        variant = (_get_var("rollback_fastapi_variant", "A") or "A").strip()
        trigger_reload(variant)


with DAG(
    dag_id="core_rollback_manual",
    start_date=datetime(2026, 2, 1, tzinfo=KST),
    schedule=None,
    catchup=False,
    tags=["rollback", "triton", "ops"],
    on_failure_callback=alert_slack,
) as dag:
    PythonOperator(task_id="rollback_manual", python_callable=_run)

