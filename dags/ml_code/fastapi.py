# dags/ml_code/fastapi.py
from __future__ import annotations
import requests
from airflow.utils.log.logging_mixin import LoggingMixin
from ml_code.config import cfg

log = LoggingMixin().log


def trigger_reload(*, alias: str):
    base = cfg("FASTAPI_RELOAD_URL", required=True)
    token = cfg("RELOAD_SECRET_TOKEN", required=True)

    url = f"{base}/variant/{alias}/reload"
    r = requests.post(url, headers={"x-token": token}, timeout=5)
    if r.status_code != 200:
        raise RuntimeError(f"reload failed: {r.status_code} {r.text}")
    log.info("reload OK alias=%s resp=%s", alias, r.text[:200])

