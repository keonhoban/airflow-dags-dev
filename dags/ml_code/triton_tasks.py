# dags/ml_code/triton_tasks.py
from __future__ import annotations

import json
import os
from typing import Any, Dict, Optional, Sequence

from airflow.utils.log.logging_mixin import LoggingMixin

from ml_code.config import cfg
from mlops_lib.core.triton_config import atomic_write, write_or_update_config_policy
from ml_code.triton_actions import (
    utc_ts,
    decide_deploy_target,
    materialize_repo,
    triton_unload,
    triton_load,
    triton_ready,
    triton_infer_smoke,
    rebuild_config_for_version,
    run_id_by_version,
)

log = LoggingMixin().log

# XCom keys (SSOT)
K_MODEL = "model"
K_MODEL_DIR = "model_dir"
K_DEPLOY_VERSION = "deploy_version"
K_RUN_ID = "run_id"
K_ALIAS = "alias"
K_DEPLOY_MODE = "deploy_mode"
K_N_FEATURES = "n_features"
K_N_CLASSES = "n_classes"
K_ONNX_INPUT_NAME = "onnx_input_name"

# Task IDs (TaskGroup 고려: SSOT)
T_MAT = ("deploy.materialize_repo", "materialize_repo")
T_SNAPSHOT = ("deploy.snapshot_current", "snapshot_current")


def _xcom_pull_any(ti, *, key: str, task_ids: Sequence[str]) -> Optional[Any]:
    for tid in task_ids:
        v = ti.xcom_pull(task_ids=tid, key=key)
        if v is not None:
            return v
    return None


def _require_xcom(ti, *, key: str, task_ids: Sequence[str], hint: str) -> Any:
    v = _xcom_pull_any(ti, key=key, task_ids=task_ids)
    if v is None or v == "":
        raise RuntimeError(f"XCom missing: key='{key}' from task_ids={list(task_ids)}. Hint: {hint}")
    return v


def snapshot_current(ti, **_) -> None:
    base_model = cfg("triton_model_name", required=True)
    repo = cfg("triton_repo_base", "/models")
    model_dir = os.path.join(str(repo), str(base_model))
    path = os.path.join(model_dir, "current.json")

    prev = None
    if os.path.exists(path):
        try:
            with open(path, "r") as f:
                prev = json.load(f)
        except Exception as e:
            log.warning("[snapshot] read current.json failed: %s", e)

    ti.xcom_push(key="prev_current", value=prev)
    log.info("[snapshot] model=%s prev=%s", base_model, prev)


def materialize(ti, alias: str = "A", *, run_id: str | None = None, shadow: bool = False, **_) -> None:
    base_model = cfg("triton_model_name", required=True)

    model, deploy_version, chosen_run_id, used_alias, mode = decide_deploy_target(
        base_model=str(base_model),
        alias=str(alias),
        run_id=str(run_id) if run_id else None,
        shadow=bool(shadow),
    )

    meta = materialize_repo(model=model, deploy_version=int(deploy_version), run_id=str(chosen_run_id))

    # ✅ materialize_repo task에서 downstream이 쓰는 값들 전부 push
    ti.xcom_push(key=K_MODEL, value=meta[K_MODEL])
    ti.xcom_push(key=K_MODEL_DIR, value=meta[K_MODEL_DIR])
    ti.xcom_push(key=K_DEPLOY_VERSION, value=int(meta[K_DEPLOY_VERSION]))
    ti.xcom_push(key=K_RUN_ID, value=meta[K_RUN_ID])
    ti.xcom_push(key=K_ALIAS, value=used_alias)
    ti.xcom_push(key=K_DEPLOY_MODE, value=mode)
    ti.xcom_push(key=K_N_FEATURES, value=int(meta[K_N_FEATURES]))
    ti.xcom_push(key=K_N_CLASSES, value=int(meta[K_N_CLASSES]))
    ti.xcom_push(key=K_ONNX_INPUT_NAME, value=meta[K_ONNX_INPUT_NAME])

    log.info(
        "[materialize] mode=%s model=%s alias=@%s version=%s run_id=%s",
        mode,
        meta[K_MODEL],
        used_alias,
        meta[K_DEPLOY_VERSION],
        meta[K_RUN_ID],
    )


def triton_load_task(ti, **_) -> None:
    model = _require_xcom(
        ti,
        key=K_MODEL,
        task_ids=T_MAT,
        hint="materialize_repo가 TaskGroup deploy 아래면 task_id는 'deploy.materialize_repo' 입니다.",
    )
    triton_unload(str(model))
    triton_load(str(model))


def triton_ready_task(ti, **_) -> None:
    model = _require_xcom(ti, key=K_MODEL, task_ids=T_MAT, hint="Check TaskGroup prefix deploy.*")
    triton_ready(str(model))


def triton_infer_smoke_task(ti, **_) -> None:
    model = _require_xcom(ti, key=K_MODEL, task_ids=T_MAT, hint="Check TaskGroup prefix deploy.*")
    n_features = int(_xcom_pull_any(ti, key=K_N_FEATURES, task_ids=T_MAT) or 0)
    in_name = _xcom_pull_any(ti, key=K_ONNX_INPUT_NAME, task_ids=T_MAT) or "input"
    _ = triton_infer_smoke(str(model), in_name=str(in_name), n_features=int(n_features))


def commit_current(ti, **_) -> None:
    model = _require_xcom(ti, key=K_MODEL, task_ids=T_MAT, hint="Check TaskGroup prefix deploy.*")
    base_model = cfg("triton_model_name", required=True)

    deploy_mode = _xcom_pull_any(ti, key=K_DEPLOY_MODE, task_ids=T_MAT)
    if str(deploy_mode) == "shadow":
        log.warning("[commit] skip current.json for shadow model=%s", model)
        return

    if str(model) != str(base_model):
        raise RuntimeError(f"[commit] unexpected promote model={model} (base_model={base_model})")

    model_dir = _require_xcom(ti, key=K_MODEL_DIR, task_ids=T_MAT, hint="materialize must push model_dir")
    deploy_version = int(_require_xcom(ti, key=K_DEPLOY_VERSION, task_ids=T_MAT, hint="materialize must push deploy_version"))
    run_id = _xcom_pull_any(ti, key=K_RUN_ID, task_ids=T_MAT)
    alias = _xcom_pull_any(ti, key=K_ALIAS, task_ids=T_MAT)

    payload = {
        "active_version": deploy_version,
        "run_id": run_id,
        "alias": alias,
        "deploy_mode": deploy_mode,
        "updated_at_utc": utc_ts(),
    }

    path = os.path.join(str(model_dir), "current.json")
    atomic_write(path, json.dumps(payload, indent=2))

    write_or_update_config_policy(str(model_dir), version=deploy_version)
    log.info("[commit] OK version=%s path=%s", deploy_version, path)


def rollback_minimal(ti, **_) -> None:
    model = _require_xcom(ti, key=K_MODEL, task_ids=T_MAT, hint="Check TaskGroup prefix deploy.*")
    model_dir = _xcom_pull_any(ti, key=K_MODEL_DIR, task_ids=T_MAT)
    deploy_v = _xcom_pull_any(ti, key=K_DEPLOY_VERSION, task_ids=T_MAT)
    deploy_mode = _xcom_pull_any(ti, key=K_DEPLOY_MODE, task_ids=T_MAT)

    prev = _xcom_pull_any(ti, key="prev_current", task_ids=T_SNAPSHOT)

    log.warning("[ROLLBACK] start model=%s deploy_version=%s mode=%s", model, deploy_v, deploy_mode)

    base_model = cfg("triton_model_name", required=True)

    if str(deploy_mode) != "shadow" and str(model) == str(base_model) and prev is not None and model_dir:
        path = os.path.join(str(model_dir), "current.json")
        atomic_write(path, json.dumps(prev, indent=2))
        log.warning("[ROLLBACK] restored current.json (promote)")

    if deploy_v is not None and model_dir:
        ver_dir = os.path.join(str(model_dir), str(deploy_v))
        if os.path.isdir(ver_dir):
            failed_dir = ver_dir + f".failed_{utc_ts()}"
            os.rename(ver_dir, failed_dir)
            log.warning("[ROLLBACK] moved failed dir: %s -> %s", ver_dir, failed_dir)

    try:
        triton_unload(str(model))
        triton_load(str(model))
    except Exception as e:
        log.warning("[ROLLBACK] reload failed: %s", e)


def rollback_manual(model: str | None = None, deploy_version: int | None = None) -> None:
    """
    수동 롤백: base_model(best_model) 전용 권장
    """
    model = str(model or cfg("triton_model_name", required=True))
    repo = cfg("triton_repo_base", "/models")
    model_dir = os.path.join(str(repo), model)
    os.makedirs(model_dir, exist_ok=True)

    path = os.path.join(model_dir, "current.json")
    cur: Dict[str, Any] = {}
    if os.path.exists(path):
        try:
            with open(path, "r") as f:
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
