# dags/ml_code/trigger_reload.py
from __future__ import annotations

import os
from typing import Optional, Dict, Any

import requests
from airflow.models import Variable
from airflow.utils.log.logging_mixin import LoggingMixin

log = LoggingMixin().log


def _cfg(key: str, default=None, *, required: bool = False):
    v = os.getenv(key)
    if v is not None and str(v).strip() != "":
        return v
    try:
        if default is None:
            v = Variable.get(key)
        else:
            v = Variable.get(key, default_var=str(default))
        if v is not None and str(v).strip() != "":
            return v
    except Exception:
        pass
    if required:
        raise RuntimeError(f"[Config] missing required key: {key}")
    return default


def _build_fastapi_base_url() -> str:
    """
    1) FASTAPI_RELOAD_URL이 있으면 그걸 사용 (예: http://fastapi-dev.../variant/A/reload)
       -> 이 경우엔 base_url을 뽑기 애매해서, 아래에서 그대로 쓰도록 처리하지 않고
          "BASE"로 고정된 변수를 쓰게 설계합니다.

    2) 없으면 service/ns/port로 조립
    """
    base = _cfg("FASTAPI_BASE_URL", None)
    if base:
        return base.rstrip("/")

    svc = _cfg("FASTAPI_SERVICE", "fastapi-dev")
    ns = _cfg("FASTAPI_NAMESPACE", "fastapi-dev")
    port = _cfg("FASTAPI_PORT", "80")
    return f"http://{svc}.{ns}.svc.cluster.local:{port}"


def trigger_reload(
    alias: str,
    *,
    deploy_version: Optional[int] = None,
    run_id: Optional[str] = None,
) -> Dict[str, Any]:
    """
    ✅ promotion: deploy_version 모드 (Triton SSOT 동기화)
    ✅ shadow: run_id 모드 (MLflow version 조회/DB 타입 이슈 회피)
    """
    alias = (alias or "").strip() or "A"

    token = _cfg("RELOAD_SECRET_TOKEN", None) or _cfg("FASTAPI_RELOAD_SECRET_TOKEN", None)
    if not token:
        # (기존 호환) 예전 변수명도 읽어줌
        token = _cfg("FASTAPI_RELOAD_TOKEN", None)

    if not token:
        raise RuntimeError("[Reload] missing RELOAD_SECRET_TOKEN (or FASTAPI_RELOAD_SECRET_TOKEN)")

    base = _build_fastapi_base_url()
    url = f"{base}/variant/{alias}/reload"

    headers = {"x-token": token}

    params = {}
    body = None
    if run_id:
        params["run_id"] = str(run_id)
    elif deploy_version is not None:
        body = {"deploy_version": int(deploy_version)}
    else:
        # alias 메타 기반 reload (필요 시)
        body = {}

    res = requests.post(url, headers=headers, params=params, json=body, timeout=10)

    if res.status_code >= 400:
        raise RuntimeError(f"FastAPI reload failed: {res.status_code} {res.text}")

    try:
        payload = res.json()
    except Exception:
        payload = {"raw": res.text}

    log.info(
        "[Reload] OK variant=%s deploy_version=%s run_id=%s resp=%s",
        alias,
        str(deploy_version) if deploy_version is not None else None,
        run_id,
        payload,
    )
    return payload

