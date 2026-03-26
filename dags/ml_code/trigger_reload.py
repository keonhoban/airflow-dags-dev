# dags/ml_code/trigger_reload.py
from __future__ import annotations

import random
import re
import time
from typing import Any, Dict, Optional

from airflow.utils.log.logging_mixin import LoggingMixin

from ml_code.config import get_fastapi_reload_url, get_reload_token, T_FASTAPI_RELOAD
from mlops_lib.core.policy import (
    RELOAD_RETRY_MAX,
    RELOAD_RETRY_BACKOFF_BASE_SEC,
    RELOAD_RETRY_BACKOFF_CAP_SEC,
)
from mlops_lib.infra.http import request_json

log = LoggingMixin().log

_TRANSIENT_STATUS = {429, 500, 502, 503, 504}
_BAD_STATUS_RE = re.compile(r"\[HTTP] bad status: (\d+)")


def _is_transient(err: RuntimeError) -> bool:
    """네트워크 에러 또는 일시적 HTTP 상태인지 판별."""
    msg = str(err)
    if "[HTTP] request failed:" in msg:
        return True
    m = _BAD_STATUS_RE.search(msg)
    return m is not None and int(m.group(1)) in _TRANSIENT_STATUS


def _sleep_backoff(attempt: int) -> float:
    backoff = min(RELOAD_RETRY_BACKOFF_CAP_SEC, RELOAD_RETRY_BACKOFF_BASE_SEC * (2 ** attempt))
    jitter = random.uniform(0, RELOAD_RETRY_BACKOFF_BASE_SEC / 2)
    delay = backoff + jitter
    time.sleep(delay)
    return delay


def _request_with_retry(
    method: str,
    url: str,
    **kwargs: Any,
) -> Dict[str, Any]:
    last_err: RuntimeError | None = None
    for attempt in range(RELOAD_RETRY_MAX + 1):
        try:
            return request_json(method, url, **kwargs)
        except RuntimeError as e:
            last_err = e
            if attempt < RELOAD_RETRY_MAX and _is_transient(e):
                delay = _sleep_backoff(attempt)
                log.warning(
                    "[Reload] retry %d/%d after %.1fs — %s",
                    attempt + 1, RELOAD_RETRY_MAX, delay, e,
                )
                continue
            raise
    raise last_err  # type: ignore[misc]


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

    payload = _request_with_retry(
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
