# dags/mlops_lib/rollback/fastapi.py
from __future__ import annotations

import json
import time
from typing import Any, Optional, Tuple
from urllib import request as urlrequest
from urllib.error import HTTPError, URLError

from airflow.utils.log.logging_mixin import LoggingMixin

from mlops_lib.core.policy import T_FASTAPI_RELOAD_HTTP, T_FASTAPI_MODELS_HTTP
from mlops_lib.rollback.types import Ctx

log = LoggingMixin().log


def _http_json(
    method: str,
    url: str,
    payload: Optional[dict[str, Any]] = None,
    headers: Optional[dict[str, str]] = None,
    timeout: int = 10,
) -> Tuple[int, str]:
    body = None
    hdrs = {"Content-Type": "application/json"}
    if headers:
        hdrs.update(headers)
    if payload is not None:
        body = json.dumps(payload).encode("utf-8")

    req = urlrequest.Request(url, data=body, method=method, headers=hdrs)
    try:
        with urlrequest.urlopen(req, timeout=timeout) as resp:
            raw = resp.read().decode("utf-8", errors="replace")
            return resp.status, raw
    except HTTPError as e:
        raw = e.read().decode("utf-8", errors="replace") if hasattr(e, "read") else str(e)
        return int(e.code), raw
    except URLError as e:
        return 0, f"URLError: {e}"
    except (OSError, ValueError) as e:
        return 0, f"Exception: {e}"


def _ctx(ctx_like: dict[str, Any] | Ctx) -> Ctx:
    return ctx_like if isinstance(ctx_like, Ctx) else Ctx.from_dict(ctx_like)


def fastapi_reload_service(ctx_like: dict[str, Any] | Ctx) -> dict[str, Any]:
    ctx = _ctx(ctx_like)
    headers = {"x-token": ctx.fastapi_token}
    url = f"{ctx.fastapi_base_url}/variant/{ctx.alias}/reload"

    st, raw = _http_json(
        "POST",
        url,
        payload={"deploy_version": int(ctx.deploy_version)},
        headers=headers,
        timeout=T_FASTAPI_RELOAD_HTTP,
    )
    if st != 200:
        raise RuntimeError(f"[fastapi] reload(service) failed: status={st} body={raw[:800]}")

    try:
        return json.loads(raw)
    except (json.JSONDecodeError, ValueError):
        return {"_status": st, "_raw": raw}


def _fastapi_models(ctx: Ctx) -> dict[str, Any]:
    url = f"{ctx.fastapi_base_url}/models"
    st, raw = _http_json("GET", url, payload=None, headers=None, timeout=T_FASTAPI_MODELS_HTTP)
    if st != 200:
        return {"_status": st, "_raw": raw}
    try:
        return json.loads(raw)
    except (json.JSONDecodeError, ValueError):
        return {"_status": st, "_raw": raw}


def _get_served_version(models_json: dict[str, Any]) -> Optional[int]:
    try:
        ssot = models_json.get("ssot") or {}
        v = ssot.get("served_version")
        if v is None:
            return None
        return int(v)
    except (TypeError, ValueError, AttributeError):
        return None


def fastapi_wait_ssot_converged(ctx_like: dict[str, Any] | Ctx) -> None:
    ctx = _ctx(ctx_like)
    deadline = time.time() + int(ctx.fastapi_wait_timeout_sec)

    # pod 이름별로 마지막 관측 결과를 캐싱
    seen: dict[str, dict[str, Any]] = {}

    while time.time() < deadline:
        m = _fastapi_models(ctx)

        pod = str(m.get("pod") or "unknown")
        seen[pod] = m

        ok_pods = 0
        for _p, mm in seen.items():
            if _get_served_version(mm) == int(ctx.deploy_version):
                ok_pods += 1

        log.info(
            "[fastapi] observe pod=%s served=%s target=%s ok_pods=%s seen=%s",
            pod,
            _get_served_version(m),
            ctx.deploy_version,
            ok_pods,
            list(seen.keys()),
        )

        if ok_pods >= int(ctx.fastapi_converge_min_pods):
            log.warning(
                "[fastapi] SSOT CONVERGED across >=%s pods: target=%s pods=%s",
                ctx.fastapi_converge_min_pods,
                ctx.deploy_version,
                list(seen.keys()),
            )
            return

        time.sleep(int(ctx.fastapi_wait_interval_sec))

    last = seen.get(pod, {})
    raise RuntimeError(
        f"[fastapi] SSOT wait timeout. seen={json.dumps(list(seen.keys()))} last={json.dumps(last)[:800]}"
    )
