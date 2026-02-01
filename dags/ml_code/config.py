# dags/ml_code/config.py
import os
import mlflow
from mlflow.tracking import MlflowClient


def get_tracking_uri():
    uri = os.getenv("MLFLOW_TRACKING_URI")
    if not uri:
        raise ValueError("MLFLOW_TRACKING_URI missing")
    return uri


def set_tracking_uri_for_logging():
    mlflow.set_tracking_uri(get_tracking_uri())


def get_mlflow_client():
    set_tracking_uri_for_logging()
    return MlflowClient(tracking_uri=get_tracking_uri())


def get_fastapi_reload_url():
    url = os.getenv("FASTAPI_RELOAD_URL")
    if not url:
        raise ValueError("FASTAPI_RELOAD_URL missing")
    return url


def get_reload_token():
    token = os.getenv("RELOAD_SECRET_TOKEN")
    if not token:
        raise ValueError("RELOAD_SECRET_TOKEN missing")
    return token

