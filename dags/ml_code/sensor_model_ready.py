# ml_code/sensor_model_ready.py

import mlflow
from ml_code.config import get_mlflow_client
from airflow.utils.log.logging_mixin import LoggingMixin

logger = LoggingMixin().log

def check_model_ready(model_name: str, version: str) -> bool:
    client = get_mlflow_client()

    mv = client.get_model_version(name=model_name, version=version)

    logger.info(f"[Sensor] 상태 체크: {model_name} v{version} → {mv.status}")
    if mv.status == "READY":
        return True
    elif mv.status == "FAILED_REGISTRATION":
        raise RuntimeError("❌ 모델 등록 실패: 상태 FAILED_REGISTRATION")
    return False
