# dags/ml_code/register_model.py
import mlflow
from airflow.utils.log.logging_mixin import LoggingMixin
from ml_code.config import get_mlflow_client

logger = LoggingMixin().log


def register_model(run_id: str, model_name: str, mlflow_alias: str) -> int:
    client = get_mlflow_client()

    try:
        client.create_registered_model(model_name)
    except mlflow.exceptions.RestException as e:
        if "RESOURCE_ALREADY_EXISTS" not in str(e):
            raise

    result = client.create_model_version(
        name=model_name,
        source=f"runs:/{run_id}/model",
        run_id=run_id,
    )
    version = int(result.version)

    # alias overwrite
    try:
        client.delete_registered_model_alias(model_name, mlflow_alias)
    except Exception:
        pass
    client.set_registered_model_alias(model_name, mlflow_alias, version)

    logger.info("[Register] model=%s version=%s alias=@%s", model_name, version, mlflow_alias)
    return version

