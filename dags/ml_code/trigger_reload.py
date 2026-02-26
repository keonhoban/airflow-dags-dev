# dags/ml_code/trigger_reload.py
from __future__ import annotations

from typing import Any, Dict, Optional

from airflow.utils.log.logging_mixin import LoggingMixin

from ml_code.config import get_fastapi_reload_url, get_reload_token
from mlops_lib.infra.http import request_json

log = LoggingMixin().log


def trigger_reload(
    alias: str,
    *,
    deploy_version: Optional[int] = None,
    run_id: Optional[str] = None,
) -> Dict[str, Any]:
    """
    ✅ promotion: deploy_version 모드 (Triton SSOT 동기화)
    ✅ shadow: run_id 모드
    """
    alias = (alias or "").strip() or "A"

    base = get_fastapi_reload_url().rstrip("/")
    url = f"{base}/variant/{alias}/reload"

    token = get_reload_token()
    headers = {"x-token": token}

    params: Dict[str, Any] = {}
    body: Dict[str, Any] = {}

    if run_id:
        params["run_id"] = str(run_id)
    elif deploy_version is not None:
        body["deploy_version"] = int(deploy_version)

    payload = request_json(
        "POST",
        url,
        headers=headers,
        params=params or None,
        json_body=body or None,
        timeout=10,
    )

    log.info(
        "[Reload] OK variant=%s deploy_version=%s run_id=%s resp=%s",
        alias,
        str(deploy_version) if deploy_version is not None else None,
        run_id,
        str(payload)[:500],
    )
    return payload
