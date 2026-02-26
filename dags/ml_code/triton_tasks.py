# dags/ml_code/triton_tasks.py
from __future__ import annotations

import json
import os
from typing import Any, Dict, Optional

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

    ti.xcom_push(key=K_MODEL, value=meta[K_MODEL])
    ti.xcom_push(key=K_MODEL_DIR, value=meta[K_MODEL_DIR])
    ti.xcom_push(key=K_DEPLOY_VERSION, value=int(meta[K_DEPLOY_VERSION]))
    ti.xcom_push(key=K_RUN_ID, value=meta[K_RUN_ID])
    ti.xcom_push(key=K_ALIAS, value=used_alias)
    ti.xcom_push(key=K_DEPLOY_MODE, value=mode)
    ti.xcom_push(key=K_N_FEATURES, value=int(meta[K_N_FEATURES]))
    ti.xcom_push(key=K_N_CLASSES, value=int(meta[K_N_CLASSES]))
    ti.xcom_push(key=K_ONNX_INPUT_NAME, value=meta[K_ONNX_INPUT_NAME])

    log.info("[materialize] mode=%s model=%s alias=@%s version=%s run_id=%s", mode, model, used_alias, deploy_version, chosen_run_id)


def triton_load_task(ti, **_) -> None:
    model = ti.xcom_pull(task_ids="materialize_repo", key=K_MODEL)
    # 정석: unload -> load (대기 없음; ready는 Sensor/다음 task에서)
    triton_unload(str(model))
    triton_load(str(model))


def triton_ready_task(ti, **_) -> None:
    model = ti.xcom_pull(task_ids="materialize_repo", key=K_MODEL)
    triton_ready(str(model))


def triton_infer_smoke_task(ti, **_) -> None:
    model = ti.xcom_pull(task_ids="materialize_repo", key=K_MODEL)
    n_features = int(ti.xcom_pull(task_ids="materialize_repo", key=K_N_FEATURES) or 0)
    in_name = ti.xcom_pull(task_ids="materialize_repo", key=K_ONNX_INPUT_NAME) or "input"
    _ = triton_infer_smoke(str(model), in_name=str(in_name), n_features=int(n_features))


def commit_current(ti, **_) -> None:
    model = ti.xcom_pull(task_ids="materialize_repo", key=K_MODEL)
    base_model = cfg("triton_model_name", required=True)

    deploy_mode = ti.xcom_pull(task_ids="materialize_repo", key=K_DEPLOY_MODE)
    if str(deploy_mode) == "shadow":
        log.warning("[commit] skip current.json for shadow model=%s", model)
        return

    if str(model) != str(base_model):
        raise RuntimeError(f"[commit] unexpected promote model={model} (base_model={base_model})")

    model_dir = ti.xcom_pull(task_ids="materialize_repo", key=K_MODEL_DIR)
    deploy_version = int(ti.xcom_pull(task_ids="materialize_repo", key=K_DEPLOY_VERSION))
    run_id = ti.xcom_pull(task_ids="materialize_repo", key=K_RUN_ID)
    alias = ti.xcom_pull(task_ids="materialize_repo", key=K_ALIAS)

    payload = {
        "active_version": deploy_version,
        "run_id": run_id,
        "alias": alias,
        "deploy_mode": deploy_mode,
        "updated_at_utc": utc_ts(),
    }

    path = os.path.join(str(model_dir), "current.json")
    atomic_write(path, json.dumps(payload, indent=2))

    # policy 정화 포함 재적용
    write_or_update_config_policy(str(model_dir), version=deploy_version)
    log.info("[commit] OK version=%s path=%s", deploy_version, path)


def rollback_minimal(ti, **_) -> None:
    model = ti.xcom_pull(task_ids="materialize_repo", key=K_MODEL)
    model_dir = ti.xcom_pull(task_ids="materialize_repo", key=K_MODEL_DIR)
    deploy_v = ti.xcom_pull(task_ids="materialize_repo", key=K_DEPLOY_VERSION)
    deploy_mode = ti.xcom_pull(task_ids="materialize_repo", key=K_DEPLOY_MODE)
    prev = ti.xcom_pull(task_ids="snapshot_current", key="prev_current")

    log.warning("[ROLLBACK] start model=%s deploy_version=%s mode=%s", model, deploy_v, deploy_mode)

    base_model = cfg("triton_model_name", required=True)

    if str(deploy_mode) != "shadow" and str(model) == str(base_model) and prev is not None:
        path = os.path.join(str(model_dir), "current.json")
        atomic_write(path, json.dumps(prev, indent=2))
        log.warning("[ROLLBACK] restored current.json (promote)")

    if deploy_v is not None:
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
