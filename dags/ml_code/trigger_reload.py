# dags/ml_code/trigger_reload.py
import requests
from airflow.utils.log.logging_mixin import LoggingMixin
from ml_code.config import get_fastapi_reload_url, get_reload_token

logger = LoggingMixin().log


def trigger_reload(variant="A"):
    base_url = get_fastapi_reload_url()
    token = get_reload_token()

    url = f"{base_url}/variant/{variant}/reload"
    res = requests.post(url, headers={"x-token": token}, timeout=5)

    if res.status_code != 200:
        raise RuntimeError(f"FastAPI reload failed: {res.status_code} {res.text}")

    body = res.json()
    if body.get("status") != "success":
        raise RuntimeError(f"FastAPI reload failed: {body}")

    logger.info("[Reload] ok variant=%s resp=%s", variant, body)
    return body

