# dags/ml_code/register.py
from __future__ import annotations

import mlflow
from mlflow.tracking import MlflowClient
from airflow.utils.log.logging_mixin import LoggingMixin
from ml_code.config import cfg

log = LoggingMixin().log


def register_model_and_set_alias(*, run_id: str, model_name: str, alias: str) -> int:
    uri = cfg("MLFLOW_TRACKING_URI", required=True)
    c = MlflowClient(tracking_uri=uri)

    try:
        c.create_registered_model(model_name)
    except mlflow.exceptions.RestException as e:
        if "RESOURCE_ALREADY_EXISTS" not in str(e):
            raise

    mv = c.create_model_version(name=model_name, source=f"runs:/{run_id}/model", run_id=run_id)
    version = int(mv.version)

    # alias 갱신
    try:
        c.delete_registered_model_alias(model_name, alias)
    except Exception:
        pass

    c.set_registered_model_alias(model_name, alias, version)
    log.info("[REG] model=%s version=%s alias=@%s", model_name, version, alias)
    return version

