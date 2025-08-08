# ml_code/config.py

import os
import mlflow
from mlflow.tracking import MlflowClient
from airflow.sdk import Variable

# MLflow URI
def get_tracking_uri():
    uri = os.getenv("MLFLOW_TRACKING_URI")
    if not uri:
        raise ValueError("❌ MLFLOW_TRACKING_URI 환경변수 누락")
    return uri

# MLflow Client
def get_mlflow_client():
    set_tracking_uri_for_logging()
    return MlflowClient(tracking_uri=get_tracking_uri())

# FastAPI Reload URL
def get_fastapi_reload_url():
    url = os.getenv("FASTAPI_RELOAD_URL")
    if not url:
        raise ValueError("❌ FASTAPI_RELOAD_URL 환경변수 누락")
    return url

# Reload Secret Token
def get_reload_token():
    token = os.getenv("RELOAD_SECRET_TOKEN")
    if not token:
        raise ValueError("❌ RELOAD_SECRET_TOKEN 환경변수 누락")
    return token

# 로그용 tracking URI 세팅
def set_tracking_uri_for_logging():
    mlflow.set_tracking_uri(get_tracking_uri())

# Experiment Name 
def get_experiment_name():
    variable = Variable.get("experiment_name")
    if not variable:
        raise ValueError("❌ airflow variable 누락")
    return variable
