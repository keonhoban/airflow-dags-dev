from __future__ import annotations

import requests
from airflow.utils.log.logging_mixin import LoggingMixin
from mlops.config import cfg
from mlops.slack import notify

log = LoggingMixin().log


def task_fastapi_reload(**context):
    ti = context["ti"]
    base_url = cfg("FASTAPI_RELOAD_URL", required=True).rstrip("/")
    token = cfg("RELOAD_SECRET_TOKEN", required=True)

    alias = ti.xcom_pull(task_ids="train_and_evaluate", key="alias") or cfg("mlflow_alias", "A")
    url = f"{base_url}/variant/{alias}/reload"

    r = requests.post(url, headers={"x-token": token}, timeout=8)
    if r.status_code != 200:
        notify("FastAPI reload failed", url=url, status=r.status_code, body=r.text[:300])
        raise RuntimeError(f"FastAPI reload failed: {r.status_code} {r.text[:300]}")

    try:
        js = r.json()
    except Exception:
        js = {"raw": r.text[:200]}

    if js.get("status") != "success":
        notify("FastAPI reload bad response", url=url, resp=str(js)[:300])
        raise RuntimeError(f"FastAPI reload bad response: {js}")

    notify("FastAPI reload ok", alias=alias, url=url)
    log.info("[RELOAD] ok alias=%s resp=%s", alias, js)

