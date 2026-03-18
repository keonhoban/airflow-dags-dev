# dags/ml_code/config.py
from __future__ import annotations

from typing import Any

import mlflow
from mlflow.tracking import MlflowClient

from mlops_lib.core.policy import get_var as _get_var

# -----------------------
# Key SSOT (면접/운영 설명용 고정)
# -----------------------
K_MLFLOW_TRACKING_URI = "MLFLOW_TRACKING_URI"

K_FASTAPI_BASE_URL_ENV = "FASTAPI_BASE_URL"
K_FASTAPI_BASE_URL_VAR = "fastapi_base_url"

K_RELOAD_TOKEN_ENV = "RELOAD_SECRET_TOKEN"
K_RELOAD_TOKEN_VAR = "reload_token"

K_EXPERIMENT_NAME = "experiment_name"

# -----------------------
# Timeout SSOT
# -----------------------
K_FASTAPI_RELOAD_TIMEOUT_ENV = "FASTAPI_RELOAD_TIMEOUT"
K_FASTAPI_RELOAD_TIMEOUT_VAR = "fastapi_reload_timeout"
DEFAULT_FASTAPI_RELOAD_TIMEOUT = 10


def cfg(key: str, default: Any = None, *, required: bool = False) -> Any:
    """
    Config precedence: env var → Airflow Variable → default.

    구현은 mlops_lib.core.policy.get_var에 위임.
    ml_code 전체에서 이 함수를 통해 설정을 읽으므로 이름은 유지한다.
    """
    str_default = str(default) if default is not None else None
    return _get_var(key, str_default, required=required)


def cfg_int(key: str, default: int, *, required: bool = False) -> int:
    raw = cfg(key, default, required=required)
    try:
        return int(str(raw).strip())
    except Exception:
        if required:
            raise RuntimeError(f"[Config] invalid int key={key} value={raw!r}")
        return int(default)


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
      2) Airflow Variable fastapi_base_url
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


# -----------------------
# Exported SSOT constants
# -----------------------
T_FASTAPI_RELOAD = cfg_int(K_FASTAPI_RELOAD_TIMEOUT_ENV, DEFAULT_FASTAPI_RELOAD_TIMEOUT, required=False)
# env가 없으면 Variable도 허용(실습/운영 유연성)
T_FASTAPI_RELOAD = cfg_int(K_FASTAPI_RELOAD_TIMEOUT_VAR, T_FASTAPI_RELOAD, required=False)
