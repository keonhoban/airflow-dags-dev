from __future__ import annotations

from mlflow.tracking import MlflowClient
from airflow.utils.log.logging_mixin import LoggingMixin
from mlops.config import cfg

log = LoggingMixin().log


def sensor_model_ready(**context) -> bool:
    ti = context["ti"]
    tracking_uri = cfg("MLFLOW_TRACKING_URI", required=True)
    client = MlflowClient(tracking_uri=tracking_uri)

    model_name = ti.xcom_pull(task_ids="train_and_evaluate", key="model_name") or cfg("model_name", "best_model")
    version = ti.xcom_pull(task_ids="register_model", key="version")

    if not version:
        raise ValueError("version missing from register_model")

    mv = client.get_model_version(name=model_name, version=str(version))
    log.info("[SENSOR] model=%s version=%s status=%s", model_name, version, mv.status)

    if mv.status == "READY":
        return True
    if mv.status == "FAILED_REGISTRATION":
        raise RuntimeError("MLflow model registration failed (FAILED_REGISTRATION)")
    return False

