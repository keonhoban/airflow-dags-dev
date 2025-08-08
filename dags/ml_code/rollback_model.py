# ml_code/rollback_model.py

import mlflow
from ml_code.config import get_mlflow_client
from mlflow.exceptions import MlflowException
from airflow.utils.log.logging_mixin import LoggingMixin

logger = LoggingMixin().log

def rollback_model(model_name: str, version: str, alias: str):
    try:
        if not model_name or not version or not alias:
            raise ValueError("❌ model_name / version / alias 값이 비어있음")

        client = get_mlflow_client()

        current_version = client.get_model_version_by_alias(model_name, alias).version
        if str(current_version) == str(version):
            raise RuntimeError(f"[Rollback] 이미 @{alias} → v{version} 상태")

        # ✅ 대상 버전 READY 상태 확인
        model_info = client.get_model_version(name=model_name, version=version)
        if model_info.status != "READY":
            raise RuntimeError(f"[Rollback] 대상 버전 READY 아님 → 현재: {model_info.status}")

        # ✅ 기존 alias 삭제 후 rollback 대상에 할당
        client.delete_registered_model_alias(model_name, alias)
        client.set_registered_model_alias(model_name, alias, version)

        logger.info(f"[Rollback] 성공: @{alias} → v{version}")

    except MlflowException as e:
        raise RuntimeError(f"[Rollback Error] MLflow 예외 발생: {e}") from e
    except Exception as e:
        logger.error(f"[Rollback Error] {e}")
        raise
