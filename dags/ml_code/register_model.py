# ml_code/register_model.py

import mlflow
from ml_code.config import get_mlflow_client
from airflow.utils.log.logging_mixin import LoggingMixin

logger = LoggingMixin().log

def register_model(run_id: str, model_name: str, mlflow_alias: str) -> int:
    client = get_mlflow_client()

    try:
        client.create_registered_model(model_name)
        logger.info(f"[Register] 모델명 등록 완료: {model_name}")
    except mlflow.exceptions.RestException as e:
        if "RESOURCE_ALREADY_EXISTS" in str(e):
            logger.info(f"[Register] 모델명 이미 존재: {model_name}")
        else:
            raise RuntimeError(f"[Register] 모델명 등록 실패: {e}")

    try:
        result = client.create_model_version(
            name=model_name,
            source=f"runs:/{run_id}/model",
            run_id=run_id
        )
    except Exception as e:
        raise RuntimeError(f"❌ 모델 등록 실패: {e}")

    version = result.version
    logger.info(f"[Register] 모델 등록 완료: {model_name} / version: {version}")

    try:
        client.delete_registered_model_alias(model_name, mlflow_alias)
        logger.info(f"[Alias] {model_name}에서 기존 alias '{mlflow_alias}' 삭제 완료")
    except mlflow.exceptions.RestException as e:
        logger.warning(f"[Alias] {model_name}에서 기존 alias 삭제 실패 또는 없음: {e}")

    try:
        client.set_registered_model_alias(model_name, mlflow_alias, version)
        logger.info(f"[Alias] {model_name} v{version} → @{mlflow_alias} 할당 완료")
    except mlflow.exceptions.RestException as e:
        logger.error(f"[Alias] {model_name}에서 alias 설정 실패: {e}")
        raise RuntimeError(f"[Alias] alias 설정 실패: {e}")

    return version
