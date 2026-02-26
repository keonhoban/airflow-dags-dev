# dags/mlops_lib/infra/http.py
from __future__ import annotations

import logging
from typing import Any, Dict, Optional

import requests

log = logging.getLogger(__name__)


def request_json(
    method: str,
    url: str,
    *,
    headers: Optional[Dict[str, str]] = None,
    params: Optional[Dict[str, Any]] = None,
    json_body: Optional[Dict[str, Any]] = None,
    timeout: int = 10,
    ok_status: tuple[int, ...] = (200,),
) -> Dict[str, Any]:
    """
    표준 HTTP 호출:
    - ok_status 외 status면 RuntimeError
    - response json 파싱 실패 시 raw 텍스트로 반환
    """
    try:
        r = requests.request(
            method=method,
            url=url,
            headers=headers,
            params=params,
            json=json_body,
            timeout=timeout,
        )
    except requests.RequestException as e:
        raise RuntimeError(f"[HTTP] request failed: {method} {url} err={e}") from e

    if r.status_code not in ok_status:
        body = (r.text or "")[:500]
        raise RuntimeError(f"[HTTP] bad status: {r.status_code} {method} {url} body={body}")

    try:
        return r.json()  # type: ignore[return-value]
    except Exception:
        return {"raw": (r.text or "")[:2000]}


def request_ok(
    method: str,
    url: str,
    *,
    headers: Optional[Dict[str, str]] = None,
    params: Optional[Dict[str, Any]] = None,
    json_body: Optional[Dict[str, Any]] = None,
    timeout: int = 10,
    ok_status: tuple[int, ...] = (200,),
) -> None:
    """본문 필요 없을 때 쓰는 표준 호출."""
    _ = request_json(
        method,
        url,
        headers=headers,
        params=params,
        json_body=json_body,
        timeout=timeout,
        ok_status=ok_status,
    )
