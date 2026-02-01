# dags/ml_code/sensor.py
from __future__ import annotations

from mlflow.tracking import MlflowClient
from airflow.utils.log.logging_mixin import LoggingMixin
from ml_code.config import cfg

log = LoggingMixin().log


def is_model_ready(*, model_name: str, version: str) -> bool:
    uri = cfg("MLFLOW_TRACKING_URI", required=True)
    c = MlflowClient(tracking_uri=uri)

    mv = c.get_model_version(name=model_name, version=version)
    log.info("[SENSOR] %s v%s status=%s", model_name, version, mv.status)

    if mv.status == "READY":
        return True
    if mv.status == "FAILED_REGISTRATION":
        raise RuntimeError("Model registration failed")
    return False

