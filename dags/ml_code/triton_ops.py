from __future__ import annotations

import json
import os
from typing import Any, Dict

from airflow.utils.log.logging_mixin import LoggingMixin

from ml_code.config import cfg
from mlops_lib.core.triton_config import atomic_write
from ml_code.triton_actions import rebuild_config_for_version, run_id_by_version, triton_unload, triton_load, utc_ts

log = LoggingMixin().log


def rollback_manual(model: str | None = None, deploy_version: int | None = None) -> None:
    model = str(model or cfg("triton_model_name", required=True))
    repo = cfg("triton_repo_base", "/models")
    model_dir = os.path.join(str(repo), model)
    os.makedirs(model_dir, exist_ok=True)

    path = os.path.join(model_dir, "current.json")
    cur: Dict[str, Any] = {}
    if os.path.exists(path):
        try:
            with open(path, "r", encoding="utf-8") as f:
                cur = json.load(f) or {}
        except Exception as e:
            log.warning("[rollback_manual] read current.json failed: %s", e)

    if deploy_version is not None:
        dv = int(deploy_version)
        cur["active_version"] = dv
        cur["run_id"] = run_id_by_version(model, dv)
        cur["deploy_mode"] = "rollback_manual"
        cur["updated_at_utc"] = utc_ts()

        atomic_write(path, json.dumps(cur, indent=2))

        cfg_text = rebuild_config_for_version(model, dv)
        atomic_write(os.path.join(model_dir, "config.pbtxt"), cfg_text)
        log.warning("[ROLLBACK_MANUAL] forced dv=%s run_id=%s", dv, cur.get("run_id"))
    else:
        log.warning("[ROLLBACK_MANUAL] no deploy_version -> reload only")

    triton_unload(model)
    triton_load(model)
    log.warning("[ROLLBACK_MANUAL] reload OK")
