# dags/ml_code/rollback_model.py
from mlflow.exceptions import MlflowException
from airflow.utils.log.logging_mixin import LoggingMixin
from ml_code.config import get_mlflow_client

logger = LoggingMixin().log


def rollback_model(model_name: str, version: str, alias: str):
    client = get_mlflow_client()
    try:
        mv = client.get_model_version(name=model_name, version=version)
        if mv.status != "READY":
            raise RuntimeError(f"target not READY: {mv.status}")

        try:
            client.delete_registered_model_alias(model_name, alias)
        except Exception:
            pass
        client.set_registered_model_alias(model_name, alias, version)
        logger.info("[Rollback] ok model=%s alias=@%s version=%s", model_name, alias, version)

    except MlflowException as e:
        raise RuntimeError(f"MLflow error: {e}") from e

