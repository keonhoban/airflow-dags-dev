from __future__ import annotations

import mlflow
from mlflow.tracking import MlflowClient
from airflow.utils.log.logging_mixin import LoggingMixin
from mlops.config import cfg
from mlops.slack import notify

log = LoggingMixin().log


def task_register_model(**context):
    ti = context["ti"]
    tracking_uri = cfg("MLFLOW_TRACKING_URI", required=True)
    client = MlflowClient(tracking_uri=tracking_uri)

    run_id = ti.xcom_pull(task_ids="train_and_evaluate", key="run_id")
    model_name = ti.xcom_pull(task_ids="train_and_evaluate", key="model_name") or cfg("model_name", "best_model")
    alias = ti.xcom_pull(task_ids="train_and_evaluate", key="alias") or cfg("mlflow_alias", "A")

    if not run_id:
        raise ValueError("run_id missing from train_and_evaluate")

    try:
        client.create_registered_model(model_name)
    except mlflow.exceptions.RestException as e:
        if "RESOURCE_ALREADY_EXISTS" not in str(e):
            raise

    mv = client.create_model_version(name=model_name, source=f"runs:/{run_id}/model", run_id=run_id)
    version = int(mv.version)

    # alias swap
    try:
        client.delete_registered_model_alias(model_name, alias)
    except Exception:
        pass
    client.set_registered_model_alias(model_name, alias, version)

    ti.xcom_push(key="version", value=version)
    notify("PROMOTE: model registered + alias moved", model=model_name, alias=alias, version=version, run_id=run_id)
    log.info("[REGISTER] model=%s version=%s alias=@%s run_id=%s", model_name, version, alias, run_id)

