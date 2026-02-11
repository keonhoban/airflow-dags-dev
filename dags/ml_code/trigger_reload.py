# dags/ml_code/trigger_reload.py
from __future__ import annotations

from typing import Optional
import requests

from airflow.utils.log.logging_mixin import LoggingMixin
from ml_code.config import get_fastapi_reload_url, get_reload_token

logger = LoggingMixin().log


def trigger_reload(variant: str = "A", *, deploy_version: Optional[int] = None, run_id: Optional[str] = None):
    """
    FastAPI /reload 호출 래퍼

    - 운영(SSOT): deploy_version로 동기화 (권장)
    - 검증(shadow): run_id query로 동기화
    - 하위호환: 둘 다 없으면 FastAPI가 alias 메타로 reload
    """
    base_url = get_fastapi_reload_url()
    token = get_reload_token()

    url = f"{base_url}/variant/{variant}/reload"

    params = {}
    payload = None

    if run_id:
        params["run_id"] = run_id
    elif deploy_version is not None:
        payload = {"deploy_version": int(deploy_version)}

    res = requests.post(
        url,
        headers={"x-token": token, "Content-Type": "application/json"},
        params=params if params else None,
        json=payload,
        timeout=5,
    )

    if res.status_code != 200:
        raise RuntimeError(f"FastAPI reload failed: {res.status_code} {res.text}")

    j = res.json()
    if j.get("status") != "success":
        raise RuntimeError(f"FastAPI reload bad response: {j}")

    logger.info("[Reload] OK variant=%s deploy_version=%s run_id=%s resp=%s", variant, deploy_version, run_id, j)
    return j

