# dags/ml_code/config.py
from __future__ import annotations

import os
import mlflow
from mlflow.tracking import MlflowClient
from airflow.models import Variable


def cfg(key: str, default=None, *, required: bool = False):
    """
    Config precedence:
    1) env var
    2) Airflow Variable (same key)
    3) default
    """
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


def get_fastapi_base_url() -> str:
    """
    ✅ SSOT: Airflow Variable `fastapi_base_url`
    - env FASTAPI_BASE_URL 지원
    - trailing slash 제거
    """
    base = cfg("FASTAPI_BASE_URL", None, required=False)
    if base:
        return str(base).rstrip("/")

    base = cfg("fastapi_base_url", None, required=True)  # Airflow Variable
    return str(base).rstrip("/")


def get_fastapi_reload_url() -> str:
    """
    Reload URL = base_url (service root)
    trigger_reload.py에서 /variant/{alias}/reload 를 붙이므로 여기서는 base만 리턴
    """
    return get_fastapi_base_url()


def get_reload_token() -> str:
    # ✅ env(RELOAD_SECRET_TOKEN) or Airflow Variable(reload_token) 둘 다 지원
    # 운영에서는 RELOAD_SECRET_TOKEN을 Secret/env로 주는 게 정석
    token = cfg("RELOAD_SECRET_TOKEN", None, required=False)
    if token:
        return str(token).strip()

    # fallback: Airflow Variable로도 주입 가능하게(실습/로컬용)
    token = cfg("reload_token", None, required=True)
    return str(token).strip()


def get_experiment_name() -> str:
    # 기존 키 유지 (원하면 experiment_name도 대문자로 통일 가능)
    return cfg("experiment_name", required=True)

