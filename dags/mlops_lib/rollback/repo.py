# dags/mlops_lib/rollback/repo.py
from __future__ import annotations

import json
import os
import re
import time
from typing import Any

from airflow.utils.log.logging_mixin import LoggingMixin

from mlops_lib.core.triton_config import atomic_write
from mlops_lib.rollback.types import Ctx

log = LoggingMixin().log


def _utc_ts() -> str:
    return time.strftime("%Y%m%dT%H%M%S", time.gmtime())


def _ctx(ctx_like: dict[str, Any] | Ctx) -> Ctx:
    return ctx_like if isinstance(ctx_like, Ctx) else Ctx.from_dict(ctx_like)


def require_version_dir(ctx_like: dict[str, Any] | Ctx) -> None:
    ctx = _ctx(ctx_like)
    vdir = os.path.join(ctx.model_dir, str(int(ctx.deploy_version)))
    if not os.path.isdir(vdir):
        raise RuntimeError(f"missing version dir: {vdir}")
    onnx = os.path.join(vdir, "model.onnx")
    if not os.path.exists(onnx):
        raise RuntimeError(f"missing model.onnx: {onnx}")


def write_current_json(ctx_like: dict[str, Any] | Ctx) -> None:
    ctx = _ctx(ctx_like)
    path = os.path.join(ctx.model_dir, "current.json")
    payload = {
        "active_version": int(ctx.deploy_version),
        "run_id": "",
        "alias": ctx.alias,
        "deploy_mode": "rollback_manual",
        "updated_at_utc": _utc_ts(),
    }
    atomic_write(path, json.dumps(payload, indent=2))
    log.warning("[rollback_manual] current.json updated: %s", path)
    log.warning("[rollback_manual] current.json payload=%s", payload)


def _backup_file(path: str) -> str:
    ts = _utc_ts()
    bak = f"{path}.bak.{ts}"
    try:
        with open(path, "r", encoding="utf-8") as f:
            raw = f.read()
        atomic_write(bak, raw)
        return bak
    except Exception as e:
        log.warning("[rollback_manual] backup failed: %s", e)
        return ""


def _sanity_check_pbtxt(text: str) -> None:
    if text.count("{") != text.count("}"):
        raise RuntimeError(
            f"config.pbtxt sanity check failed: brace mismatch {{={text.count('{')} }}={text.count('}')}"
        )


def _render_version_policy_specific(config_text: str, version: int) -> str:
    block = (
        "version_policy {\n"
        "  specific {\n"
        f"    versions: [ {int(version)} ]\n"
        "  }\n"
        "}\n"
    )

    m = re.search(r"(?m)^\s*version_policy\s*(?::\s*)?\{", config_text)
    if m:
        start = m.start()
        brace_open = config_text.find("{", m.end() - 1)
        if brace_open == -1:
            raise RuntimeError("config.pbtxt parse error: missing '{' after version_policy")

        depth = 0
        end = None
        for i in range(brace_open, len(config_text)):
            ch = config_text[i]
            if ch == "{":
                depth += 1
            elif ch == "}":
                depth -= 1
                if depth == 0:
                    end = i + 1
                    break
        if end is None:
            raise RuntimeError("config.pbtxt parse error: version_policy brace not closed")

        while end < len(config_text) and config_text[end] in (" ", "\t"):
            end += 1
        while end < len(config_text) and config_text[end] == "\n":
            end += 1

        return config_text[:start] + block + "\n" + config_text[end:]

    # 없으면 적절한 위치에 삽입(기존 로직 유지)
    m2 = re.search(r"(?m)^\s*max_batch_size\s*:\s*.*$", config_text)
    if m2:
        insert_at = m2.end()
        return config_text[:insert_at] + "\n\n" + block + "\n" + config_text[insert_at:]

    m3 = re.search(r"(?m)^\s*platform\s*:\s*.*$", config_text)
    if m3:
        insert_at = m3.end()
        return config_text[:insert_at] + "\n\n" + block + "\n" + config_text[insert_at:]

    return block + "\n" + config_text


def update_config_pbtxt_specific_version(ctx_like: dict[str, Any] | Ctx) -> None:
    ctx = _ctx(ctx_like)
    path = os.path.join(ctx.model_dir, "config.pbtxt")
    if not os.path.exists(path):
        raise RuntimeError(f"missing config.pbtxt: {path}")

    with open(path, "r", encoding="utf-8") as f:
        original = f.read()

    updated = _render_version_policy_specific(original, ctx.deploy_version)
    _sanity_check_pbtxt(updated)

    if updated != original:
        bak = _backup_file(path)
        atomic_write(path, updated)
        log.warning("[rollback_manual] config.pbtxt updated: version_policy specific=%s", ctx.deploy_version)
        if bak:
            log.warning("[rollback_manual] config.pbtxt backup created: %s", bak)
    else:
        log.warning("[rollback_manual] config.pbtxt unchanged (already specific?)")
