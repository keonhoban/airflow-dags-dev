# dags/ml_code/register_model.py
import mlflow
from airflow.utils.log.logging_mixin import LoggingMixin
from ml_code.config import get_mlflow_client

logger = LoggingMixin().log


def register_model(run_id: str, model_name: str, mlflow_alias: str) -> str:
    client = get_mlflow_client()

    model_name = str(model_name).strip()
    mlflow_alias = str(mlflow_alias).strip()
    run_id = str(run_id).strip()

    if not model_name or not mlflow_alias or not run_id:
        raise ValueError(f"invalid args: run_id={run_id!r}, model_name={model_name!r}, alias={mlflow_alias!r}")

    try:
        client.create_registered_model(model_name)
        logger.info("[Register] model created: %s", model_name)
    except mlflow.exceptions.RestException as e:
        if "RESOURCE_ALREADY_EXISTS" in str(e):
            logger.info("[Register] model exists: %s", model_name)
        else:
            raise

    result = client.create_model_version(
        name=model_name,
        source=f"runs:/{run_id}/model",
        run_id=run_id,
    )

    # ✅ MLflow는 version을 문자열로 다루는 게 가장 안전합니다
    version = str(result.version).strip()
    logger.info("[Register] version created: %s v%s (type=%s)", model_name, version, type(version).__name__)

    # alias move (delete -> set)
    try:
        client.delete_registered_model_alias(model_name, mlflow_alias)
    except Exception:
        pass

    # ✅ 핵심: version을 str로 넘긴다
    client.set_registered_model_alias(model_name, mlflow_alias, version)
    logger.info("[Alias] %s v%s -> @%s", model_name, version, mlflow_alias)

    return version

