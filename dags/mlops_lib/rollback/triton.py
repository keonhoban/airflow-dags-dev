# dags/mlops_lib/rollback/triton.py
from __future__ import annotations

import json
import time
from typing import Any, Optional, Tuple
from urllib import request as urlrequest
from urllib.error import HTTPError, URLError

from airflow.utils.log.logging_mixin import LoggingMixin

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
    except Exception as e:
        return 0, f"Exception: {e}"


def _ctx(ctx_like: dict[str, Any] | Ctx) -> Ctx:
    return ctx_like if isinstance(ctx_like, Ctx) else Ctx.from_dict(ctx_like)


def triton_unload(ctx_like: dict[str, Any] | Ctx) -> None:
    ctx = _ctx(ctx_like)
    url = f"{ctx.triton_http_url}/v2/repository/models/{ctx.model}/unload"
    status, raw = _http_json("POST", url, payload={})
    if status not in (200, 201, 204):
        log.warning("[triton] unload non-200 status=%s body=%s", status, raw[:500])
    else:
        log.info("[triton] unload ok status=%s", status)


def triton_load(ctx_like: dict[str, Any] | Ctx) -> None:
    ctx = _ctx(ctx_like)
    url = f"{ctx.triton_http_url}/v2/repository/models/{ctx.model}/load"
    status, raw = _http_json("POST", url, payload={})
    if status not in (200, 201, 204):
        raise RuntimeError(f"[triton] load failed status={status} body={raw[:800]}")
    log.warning("[triton] load ok status=%s", status)


def triton_wait_ready(ctx_like: dict[str, Any] | Ctx) -> None:
    ctx = _ctx(ctx_like)
    deadline = time.time() + int(ctx.triton_ready_timeout_sec)
    url = f"{ctx.triton_http_url}/v2/models/{ctx.model}"

    last = ""
    while time.time() < deadline:
        status, raw = _http_json("GET", url, payload=None)
        last = raw
        if status == 200:
            try:
                data = json.loads(raw)
                versions = data.get("versions") or []
                if str(ctx.deploy_version) in [str(v) for v in versions]:
                    log.warning("[triton] READY: versions=%s", versions)
                    return
                log.info("[triton] not ready yet. versions=%s (target=%s)", versions, ctx.deploy_version)
            except Exception:
                log.info("[triton] not json yet: %s", raw[:200])
        else:
            log.info("[triton] status=%s body=%s", status, raw[:200])

        time.sleep(int(ctx.triton_ready_interval_sec))

    raise RuntimeError(f"[triton] not ready: target={ctx.deploy_version} last={last[:800]}")
