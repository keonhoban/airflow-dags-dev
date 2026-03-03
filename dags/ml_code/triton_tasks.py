# dags/ml_code/triton_tasks.py
from __future__ import annotations

import json
import os
from typing import Any

from airflow.utils.log.logging_mixin import LoggingMixin

from ml_code.config import cfg
from mlops_lib.core.triton_config import atomic_write
from mlops_lib.core.ids import (
    # xcom keys
    K_MODEL,
    K_MODEL_DIR,
    K_DEPLOY_VERSION,
    K_RUN_ID,
    K_ALIAS,
    K_DEPLOY_MODE,
    K_N_FEATURES,
    K_N_CLASSES,
    K_ONNX_INPUT_NAME,
    K_PREV_CURRENT,
)

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

from ml_code.triton_xcom import (
    mat_task_ids,
    snapshot_task_ids,
    xcom_pull_any,
    require_xcom,
)

from ml_code.triton_repo_safety import (
    sanitize_numeric_failed_dirs,
    quarantine_failed_version_dir,
)

log = LoggingMixin().log


# -----------------------
# Tasks (public API 유지)
# -----------------------
def snapshot_current(ti, **_) -> None:
    base_model = cfg("triton_model_name", required=True)
    repo = cfg("triton_repo_base", "/models")
    model_dir = os.path.join(str(repo), str(base_model))
    path = os.path.join(model_dir, "current.json")

    prev = None
    if os.path.exists(path):
        try:
            with open(path, "r", encoding="utf-8") as f:
                prev = json.load(f)
        except Exception as e:
            log.warning("[snapshot] read current.json failed: %s", e)

    ti.xcom_push(key=K_PREV_CURRENT, value=prev)
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

    log.info(
        "[materialize] mode=%s model=%s alias=@%s version=%s run_id=%s",
        mode,
        meta[K_MODEL],
        used_alias,
        meta[K_DEPLOY_VERSION],
        meta[K_RUN_ID],
    )


def triton_load_task(ti, **_) -> None:
    model = require_xcom(
        ti,
        key=K_MODEL,
        task_ids=mat_task_ids(),
        hint="materialize_repo는 deploy TaskGroup 아래 task_id='deploy.materialize_repo'가 표준입니다.",
    )
    triton_unload(str(model))
    triton_load(str(model))


def triton_ready_task(ti, **_) -> None:
    model = require_xcom(
        ti,
        key=K_MODEL,
        task_ids=mat_task_ids(),
        hint="deploy TaskGroup prefix('deploy.*')가 맞는지 확인하세요.",
    )
    triton_ready(str(model))


def triton_infer_smoke_task(ti, **_) -> None:
    model = require_xcom(
        ti,
        key=K_MODEL,
        task_ids=mat_task_ids(),
        hint="deploy TaskGroup prefix('deploy.*')가 맞는지 확인하세요.",
    )
    n_features = int(xcom_pull_any(ti, key=K_N_FEATURES, task_ids=mat_task_ids()) or 0)
    in_name = xcom_pull_any(ti, key=K_ONNX_INPUT_NAME, task_ids=mat_task_ids()) or "input"
    _ = triton_infer_smoke(str(model), in_name=str(in_name), n_features=int(n_features))


def commit_current(ti, **_) -> None:
    model = require_xcom(
        ti,
        key=K_MODEL,
        task_ids=mat_task_ids(),
        hint="deploy.materialize_repo가 선행되어야 합니다.",
    )
    base_model = cfg("triton_model_name", required=True)

    deploy_mode = xcom_pull_any(ti, key=K_DEPLOY_MODE, task_ids=mat_task_ids())
    if str(deploy_mode) == "shadow":
        log.warning("[commit] skip current.json for shadow model=%s", model)
        return

    if str(model) != str(base_model):
        raise RuntimeError(f"[commit] unexpected promote model={model} (base_model={base_model})")

    model_dir = require_xcom(ti, key=K_MODEL_DIR, task_ids=mat_task_ids(), hint="materialize must push model_dir")
    deploy_version = int(
        require_xcom(ti, key=K_DEPLOY_VERSION, task_ids=mat_task_ids(), hint="materialize must push deploy_version")
    )
    run_id = xcom_pull_any(ti, key=K_RUN_ID, task_ids=mat_task_ids())
    alias = xcom_pull_any(ti, key=K_ALIAS, task_ids=mat_task_ids())

    payload = {
        "active_version": deploy_version,
        "run_id": run_id,
        "alias": alias,
        "deploy_mode": deploy_mode,
        "updated_at_utc": utc_ts(),
    }

    path = os.path.join(str(model_dir), "current.json")
    atomic_write(path, json.dumps(payload, indent=2))
    log.info("[commit] OK version=%s path=%s", deploy_version, path)


def rollback_minimal(ti, **_) -> None:
    """
    롤백 최소 단위:
    - current.json snapshot 복원(가능하면)
    - 실패 버전 dir 격리(중요: Triton version 파싱 안전)
    - unload/load 재시도
    """
    model = xcom_pull_any(ti, key=K_MODEL, task_ids=mat_task_ids())
    model_dir = xcom_pull_any(ti, key=K_MODEL_DIR, task_ids=mat_task_ids())
    deploy_v = xcom_pull_any(ti, key=K_DEPLOY_VERSION, task_ids=mat_task_ids())
    deploy_mode = xcom_pull_any(ti, key=K_DEPLOY_MODE, task_ids=mat_task_ids())

    prev = xcom_pull_any(ti, key=K_PREV_CURRENT, task_ids=snapshot_task_ids())

    base_model = cfg("triton_model_name", required=True)
    repo = cfg("triton_repo_base", "/models")
    base_model_dir = os.path.join(str(repo), str(base_model))

    if not model:
        log.warning(
            "[ROLLBACK] missing XCom(model). deploy_v=%s mode=%s -> fallback only (base_model=%s)",
            str(deploy_v),
            str(deploy_mode),
            str(base_model),
        )
        if prev is not None:
            try:
                atomic_write(os.path.join(base_model_dir, "current.json"), json.dumps(prev, indent=2))
                log.warning("[ROLLBACK] restored current.json using snapshot (fallback)")
            except Exception as e:
                log.warning("[ROLLBACK] fallback restore failed: %s", e)
        else:
            log.warning("[ROLLBACK] no snapshot(prev_current) -> nothing to restore")
        return

    log.warning("[ROLLBACK] start model=%s deploy_version=%s mode=%s", model, deploy_v, deploy_mode)

    # 1) current.json 복원 (promote 케이스만)
    if str(deploy_mode) != "shadow" and str(model) == str(base_model) and prev is not None and model_dir:
        path = os.path.join(str(model_dir), "current.json")
        try:
            atomic_write(path, json.dumps(prev, indent=2))
            log.warning("[ROLLBACK] restored current.json (promote)")
        except Exception as e:
            log.warning("[ROLLBACK] failed to restore current.json: %s", e)

    # 2) 실패 버전 격리 (가장 중요)
    if deploy_v is not None:
        md = str(model_dir) if model_dir else os.path.join(str(repo), str(model))

        # 과거의 "숫자 failed dir"이 남아있으면 먼저 제거
        sanitize_numeric_failed_dirs(md)

        # 이번 deploy_v 버전을 안전한 형태로 격리
        quarantine_failed_version_dir(model_dir=md, deploy_v=int(deploy_v))

    # 3) Triton reload
    try:
        triton_unload(str(model))
        triton_load(str(model))
    except Exception as e:
        log.warning("[ROLLBACK] reload failed: %s", e)


def rollback_manual(model: str | None = None, deploy_version: int | None = None) -> None:
    """
    수동 롤백:
    - active_version 강제 지정
    - config.pbtxt 재생성
    - unload/load
    """
    model = str(model or cfg("triton_model_name", required=True))
    repo = cfg("triton_repo_base", "/models")
    model_dir = os.path.join(str(repo), model)
    os.makedirs(model_dir, exist_ok=True)

    # 안전 장치: 숫자 failed dir 남아있으면 제거
    sanitize_numeric_failed_dirs(model_dir)

    path = os.path.join(model_dir, "current.json")
    cur: dict[str, Any] = {}
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
