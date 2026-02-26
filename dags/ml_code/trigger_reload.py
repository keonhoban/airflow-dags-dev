# dags/ml_code/trigger_reload.py
from __future__ import annotations

from typing import Any, Dict, Optional

from airflow.utils.log.logging_mixin import LoggingMixin

from ml_code.config import get_fastapi_reload_url, get_reload_token, T_FASTAPI_RELOAD
from mlops_lib.infra.http import request_json

log = LoggingMixin().log


def _norm_alias(alias: Optional[str]) -> str:
    return (str(alias) if alias is not None else "").strip() or "A"


def trigger_reload(
    alias: str,
    *,
    deploy_version: Optional[int] = None,
    run_id: Optional[str] = None,
) -> Dict[str, Any]:
    """
    ✅ promotion: deploy_version 모드 (Triton SSOT 동기화)
    ✅ shadow: run_id 모드

    규칙:
      - run_id 와 deploy_version 둘 중 정확히 하나만 지정해야 함.
    """
    alias = _norm_alias(alias)

    has_run = bool(str(run_id).strip()) if run_id is not None else False
    has_ver = deploy_version is not None

    if has_run == has_ver:  # 둘 다 True 이거나 둘 다 False
        raise ValueError(
            f"[Reload] invalid args: exactly one of (run_id, deploy_version) required. "
            f"alias={alias!r} run_id={run_id!r} deploy_version={deploy_version!r}"
        )

    base = get_fastapi_reload_url().rstrip("/")
    url = f"{base}/variant/{alias}/reload"

    token = str(get_reload_token() or "").strip()
    if not token:
        raise RuntimeError("[Reload] missing reload token")

    headers = {"x-token": token}

    params: Dict[str, Any] = {}
    body: Dict[str, Any] = {}

    if has_run:
        params["run_id"] = str(run_id).strip()
        mode = "shadow"
    else:
        body["deploy_version"] = int(deploy_version)  # type: ignore[arg-type]
        mode = "promote"

    payload = request_json(
        "POST",
        url,
        headers=headers,
        params=params or None,
        json_body=body or None,
        timeout=T_FASTAPI_RELOAD,
    )

    log.info(
        "[Reload] OK mode=%s variant=%s deploy_version=%s run_id=%s resp=%s",
        mode,
        alias,
        str(deploy_version) if deploy_version is not None else None,
        str(run_id).strip() if run_id is not None else None,
        str(payload)[:500],
    )
    return payload
