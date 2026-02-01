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

    j = res.json()
    if j.get("status") != "success":
        raise RuntimeError(f"FastAPI reload bad response: {j}")

    logger.info("[Reload] OK variant=%s resp=%s", variant, j)
    return j

