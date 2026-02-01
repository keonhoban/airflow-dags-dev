# dags/ml_code/sensor_model_ready.py
from airflow.utils.log.logging_mixin import LoggingMixin
from ml_code.config import get_mlflow_client

logger = LoggingMixin().log


def check_model_ready(model_name: str, version: str) -> bool:
    client = get_mlflow_client()
    mv = client.get_model_version(name=model_name, version=version)

    logger.info("[Sensor] %s v%s status=%s", model_name, version, mv.status)
    if mv.status == "READY":
        return True
    if mv.status == "FAILED_REGISTRATION":
        raise RuntimeError("model registration failed")
    return False

