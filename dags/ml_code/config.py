# dags/ml_code/config.py
from __future__ import annotations

import os
from typing import Any, Optional

import mlflow
from airflow.models import Variable
from mlflow.tracking import MlflowClient

# ---- Key SSOT (면접/운영 설명용 고정) ----
K_MLFLOW_TRACKING_URI = "MLFLOW_TRACKING_URI"

K_FASTAPI_BASE_URL_ENV = "FASTAPI_BASE_URL"
K_FASTAPI_BASE_URL_VAR = "fastapi_base_url"

K_RELOAD_TOKEN_ENV = "RELOAD_SECRET_TOKEN"
K_RELOAD_TOKEN_VAR = "reload_token"

K_EXPERIMENT_NAME = "experiment_name"


def _non_empty(v: Any) -> Optional[str]:
    if v is None:
        return None
    s = str(v).strip()
    return s if s else None


def cfg(key: str, default: Any = None, *, required: bool = False) -> Any:
    """
    Config precedence:
      1) env var (key)
      2) Airflow Variable (same key)
      3) default

    NOTE:
      - 운영에서는 Secret/env 주입이 정석
      - Variable fallback은 로컬/실습 편의용
    """
    v = _non_empty(os.getenv(key))
    if v is not None:
        return v

    try:
        # default_var는 문자열로 들어가므로, default가 None이면 "없음"으로 간주
        if default is None:
            v = Variable.get(key)  # 없으면 예외
        else:
            v = Variable.get(key, default_var=str(default))
        v = _non_empty(v)
        if v is not None:
            return v
    except Exception:
        pass

    if required:
        raise RuntimeError(f"[Config] missing required key: {key}")
    return default


def get_tracking_uri() -> str:
    return str(cfg(K_MLFLOW_TRACKING_URI, required=True))


def get_mlflow_client() -> MlflowClient:
    uri = get_tracking_uri()
    mlflow.set_tracking_uri(uri)
    return MlflowClient(tracking_uri=uri)


def get_fastapi_base_url() -> str:
    """
    ✅ FastAPI base url precedence:
      1) env FASTAPI_BASE_URL
      2) Airflow Variable fastapi_base_url (SSOT)
    """
    base = cfg(K_FASTAPI_BASE_URL_ENV, None, required=False)
    if base:
        return str(base).rstrip("/")

    base = cfg(K_FASTAPI_BASE_URL_VAR, None, required=True)
    return str(base).rstrip("/")


def get_fastapi_reload_url() -> str:
    """
    Reload URL = base_url (service root)
    trigger_reload.py에서 /variant/{alias}/reload 를 붙이므로 여기서는 base만 리턴
    """
    return get_fastapi_base_url()


def get_reload_token() -> str:
    """
    ✅ Token precedence:
      1) env RELOAD_SECRET_TOKEN (운영 정석)
      2) Airflow Variable reload_token (fallback)
    """
    token = cfg(K_RELOAD_TOKEN_ENV, None, required=False)
    if token:
        return str(token).strip()

    token = cfg(K_RELOAD_TOKEN_VAR, None, required=True)
    return str(token).strip()


def get_experiment_name() -> str:
    return str(cfg(K_EXPERIMENT_NAME, required=True))
