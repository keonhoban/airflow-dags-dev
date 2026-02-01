from __future__ import annotations
import mlflow
from mlflow.tracking import MlflowClient
from airflow.utils.log.logging_mixin import LoggingMixin
from e2e.config import cfg

log = LoggingMixin().log

def _client():
    uri = cfg("MLFLOW_TRACKING_URI", required=True)
    mlflow.set_tracking_uri(uri)
    return MlflowClient(tracking_uri=uri)

def register_alias(**context):
    ti = context["ti"]
    run_id = ti.xcom_pull(task_ids="train", key="run_id")
    if not run_id:
        raise ValueError("[REG] run_id missing")

    model_name = cfg("model_name", "best_model")
    alias = ti.xcom_pull(task_ids="train", key="alias") or cfg("mlflow_alias", "A")

    c = _client()
    try:
        c.create_registered_model(model_name)
    except Exception:
        pass

    mv = c.create_model_version(name=model_name, source=f"runs:/{run_id}/model", run_id=run_id)
    version = mv.version

    # alias update (best-effort delete)
    try:
        c.delete_registered_model_alias(model_name, alias)
    except Exception:
        pass
    c.set_registered_model_alias(model_name, alias, version)

    ti.xcom_push(key="version", value=str(version))
    log.info("[REG] model=%s version=%s alias=@%s", model_name, version, alias)

def sensor_model_ready(**context) -> bool:
    ti = context["ti"]
    model_name = cfg("model_name", "best_model")
    version = ti.xcom_pull(task_ids="register_model", key="version")
    if not version:
        raise ValueError("[REG] version missing for ready sensor")

    c = _client()
    mv = c.get_model_version(name=model_name, version=version)
    log.info("[REG] ready check model=%s v=%s status=%s", model_name, version, mv.status)

    if mv.status == "READY":
        return True
    if mv.status == "FAILED_REGISTRATION":
        raise RuntimeError("[REG] model registration failed")
    return False

