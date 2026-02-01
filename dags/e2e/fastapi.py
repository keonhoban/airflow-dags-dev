from __future__ import annotations
import os, requests
from airflow.utils.log.logging_mixin import LoggingMixin

log = LoggingMixin().log

def trigger_reload_task(**context):
    ti = context["ti"]
    alias = ti.xcom_pull(task_ids="train", key="alias") or "A"

    base = os.getenv("FASTAPI_RELOAD_URL")
    token = os.getenv("RELOAD_SECRET_TOKEN")
    if not base or not token:
        raise RuntimeError("[FASTAPI] FASTAPI_RELOAD_URL or RELOAD_SECRET_TOKEN missing")

    url = f"{base}/variant/{alias}/reload"
    r = requests.post(url, headers={"x-token": token}, timeout=5)
    if r.status_code != 200:
        raise RuntimeError(f"[FASTAPI] reload failed: {r.status_code} {r.text[:300]}")
    log.info("[FASTAPI] reload OK alias=%s resp=%s", alias, r.text[:300])

