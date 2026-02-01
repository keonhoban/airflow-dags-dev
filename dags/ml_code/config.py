# dags/ml_code/config.py
import os
import mlflow
from mlflow.tracking import MlflowClient
from airflow.models import Variable


def cfg(key: str, default=None, *, required: bool = False):
    v = os.getenv(key)
    if v is not None and str(v).strip() != "":
        return v
    try:
        if default is None:
            v = Variable.get(key)
        else:
            v = Variable.get(key, default_var=str(default))
        if v is not None and str(v).strip() != "":
            return v
    except Exception:
        pass
    if required:
        raise RuntimeError(f"[Config] missing required key: {key}")
    return default


def get_tracking_uri() -> str:
    return cfg("MLFLOW_TRACKING_URI", required=True)


def get_mlflow_client() -> MlflowClient:
    mlflow.set_tracking_uri(get_tracking_uri())
    return MlflowClient(tracking_uri=get_tracking_uri())


def get_fastapi_reload_url() -> str:
    return cfg("FASTAPI_RELOAD_URL", required=True)


def get_reload_token() -> str:
    return cfg("RELOAD_SECRET_TOKEN", required=True)


def get_experiment_name() -> str:
    return cfg("experiment_name", required=True)

