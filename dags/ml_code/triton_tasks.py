# dags/ml_code/triton_tasks.py
from __future__ import annotations

import json
import os
from typing import Any, Optional, Sequence

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
    # task ids
    TRITON_MAT_TASK_ID,
    TRITON_SNAPSHOT_TASK_ID,
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

log = LoggingMixin().log

# -----------------------
# Legacy support (optional)
# -----------------------
_LEGACY_MAT_TASK_IDS: Sequence[str] = (TRITON_MAT_TASK_ID, "materialize_repo")
_LEGACY_SNAPSHOT_TASK_IDS: Sequence[str] = (TRITON_SNAPSHOT_TASK_ID, "snapshot_current")


def _allow_legacy_task_ids() -> bool:
    return str(cfg("allow_legacy_task_ids", "false")).lower() in ("1", "true", "yes", "y")


def _task_id_candidates(primary: str, legacy: Sequence[str]) -> Sequence[str]:
    return legacy if _allow_legacy_task_ids() else (primary,)


def _xcom_pull_any(ti, *, key: str, task_ids: Sequence[str]) -> Optional[Any]:
    for tid in task_ids:
        v = ti.xcom_pull(task_ids=tid, key=key)
        if v is not None:
            return v
    return None


def _require_xcom(ti, *, key: str, task_ids: Sequence[str], hint: str) -> Any:
    v = _xcom_pull_any(ti, key=key, task_ids=task_ids)
    if v is None or v == "":
        raise RuntimeError(
            "XCom missing: "
            f"key='{key}' from task_ids={list(task_ids)}. "
            f"Hint: {hint}"
        )
    return v


def _mat_task_ids() -> Sequence[str]:
    return _task_id_candidates(TRITON_MAT_TASK_ID, _LEGACY_MAT_TASK_IDS)


def _snapshot_task_ids() -> Sequence[str]:
    return _task_id_candidates(TRITON_SNAPSHOT_TASK_ID, _LEGACY_SNAPSHOT_TASK_IDS)


# -----------------------
# Quarantine dir helpers
# -----------------------
def _ensure_writable_quarantine_dir(model_dir: str) -> str:
    """
    NFS에서 _quarantine 디렉토리가 root:root 755로 남아있으면
    airflow(uid=50000)가 그 아래에 write/rename을 못 해서 rollback이 깨질 수 있습니다.

    정책:
    1) model_dir/_quarantine 를 우선 사용 (가능하면 chmod 2775 시도)
    2) 쓰기 불가하면 model_dir/_quarantine_airflow 로 폴백 (airflow가 직접 생성)

    반환: 실제로 사용할 quarantine dir 경로
    """
    q1 = os.path.join(model_dir, "_quarantine")
    q2 = os.path.join(model_dir, "_quarantine_airflow")

    def _mkdir(p: str) -> None:
        os.makedirs(p, exist_ok=True)

    def _chmod_2775(p: str) -> None:
        # best effort: 실패해도 폴백으로 해결 가능
        try:
            os.chmod(p, 0o2775)  # setgid + group write
        except Exception as e:
            log.warning("[QUARANTINE] chmod 2775 failed: path=%s err=%s", p, e)

    def _is_writable_dir(p: str) -> bool:
        try:
            return os.path.isdir(p) and os.access(p, os.W_OK | os.X_OK)
        except Exception:
            return False

    # 1) prefer _quarantine
    try:
        _mkdir(q1)
        _chmod_2775(q1)
        if _is_writable_dir(q1):
            return q1
        log.warning("[QUARANTINE] not writable: %s (will fallback)", q1)
    except Exception as e:
        log.warning("[QUARANTINE] prepare failed: %s err=%s (will fallback)", q1, e)

    # 2) fallback: _quarantine_airflow
    try:
        _mkdir(q2)
        _chmod_2775(q2)
        if _is_writable_dir(q2):
            log.warning("[QUARANTINE] using fallback dir: %s", q2)
            return q2
    except Exception as e:
        log.warning("[QUARANTINE] fallback prepare failed: %s err=%s", q2, e)

    # 마지막: 그래도 안되면 q1 반환(기존 동작 유지) -> rename 실패 로그로 남게 됨
    return q1


# -----------------------
# Triton repo safety helpers
# -----------------------
def _sanitize_numeric_failed_dirs(model_dir: str) -> None:
    """
    과거에 생성된 "56.failed_..." 같은 디렉토리가 남아있으면
    Triton이 version=56 으로 오인 → /56/model.onnx stat 실패 → UNAVAILABLE로 남아서
    model load / readiness를 계속 깨뜨릴 수 있습니다.

    따라서 숫자로 시작하고 '.failed_'를 포함한 디렉토리는
    _quarantine/ 아래로 이동해 Triton의 버전 스캔에서 완전히 제외합니다.
    """
    try:
        if not os.path.isdir(model_dir):
            return

        qdir = _ensure_writable_quarantine_dir(model_dir)

        for name in os.listdir(model_dir):
            if not name:
                continue
            # ex) "56.failed_2026..." 형태
            if name[0].isdigit() and ".failed_" in name:
                src = os.path.join(model_dir, name)
                if os.path.isdir(src):
                    dst = os.path.join(qdir, f"failed_{name}")
                    log.warning("[SANITIZE] move numeric failed dir: %s -> %s", src, dst)
                    try:
                        os.rename(src, dst)
                    except Exception as e:
                        log.warning("[SANITIZE] rename failed: %s", e)
    except Exception as e:
        log.warning("[SANITIZE] failed: %s", e)


def _quarantine_failed_version_dir(*, model_dir: str, deploy_v: int) -> None:
    """
    실패 버전 디렉토리를 안전하게 격리합니다.

    ❌ 절대 금지:
      - "56.failed_..." 같은 형태 (숫자로 시작) => Triton이 version=56으로 해석 가능

    ✅ 권장:
      - _quarantine/failed_v{version}_{ts}
      - 숫자로 시작하지 않는 이름
      - 루트 버전 디렉토리 스캔 영역 밖
    """
    ver_dir = os.path.join(model_dir, str(deploy_v))
    if not os.path.isdir(ver_dir):
        return

    quarantine_dir = _ensure_writable_quarantine_dir(model_dir)

    failed_name = f"failed_v{deploy_v}_{utc_ts()}"
    failed_dir = os.path.join(quarantine_dir, failed_name)

    try:
        os.rename(ver_dir, failed_dir)
        log.warning("[ROLLBACK] moved failed dir: %s -> %s", ver_dir, failed_dir)
    except Exception as e:
        log.warning("[ROLLBACK] failed to move dir: %s -> %s err=%s", ver_dir, failed_dir, e)


# -----------------------
# Tasks
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
    model = _require_xcom(
        ti,
        key=K_MODEL,
        task_ids=_mat_task_ids(),
        hint="materialize_repo는 deploy TaskGroup 아래 task_id='deploy.materialize_repo'가 표준입니다.",
    )
    triton_unload(str(model))
    triton_load(str(model))


def triton_ready_task(ti, **_) -> None:
    model = _require_xcom(
        ti,
        key=K_MODEL,
        task_ids=_mat_task_ids(),
        hint="deploy TaskGroup prefix('deploy.*')가 맞는지 확인하세요.",
    )
    triton_ready(str(model))


def triton_infer_smoke_task(ti, **_) -> None:
    model = _require_xcom(
        ti,
        key=K_MODEL,
        task_ids=_mat_task_ids(),
        hint="deploy TaskGroup prefix('deploy.*')가 맞는지 확인하세요.",
    )
    n_features = int(_xcom_pull_any(ti, key=K_N_FEATURES, task_ids=_mat_task_ids()) or 0)
    in_name = _xcom_pull_any(ti, key=K_ONNX_INPUT_NAME, task_ids=_mat_task_ids()) or "input"
    _ = triton_infer_smoke(str(model), in_name=str(in_name), n_features=int(n_features))


def commit_current(ti, **_) -> None:
    model = _require_xcom(
        ti,
        key=K_MODEL,
        task_ids=_mat_task_ids(),
        hint="deploy.materialize_repo가 선행되어야 합니다.",
    )
    base_model = cfg("triton_model_name", required=True)

    deploy_mode = _xcom_pull_any(ti, key=K_DEPLOY_MODE, task_ids=_mat_task_ids())
    if str(deploy_mode) == "shadow":
        log.warning("[commit] skip current.json for shadow model=%s", model)
        return

    if str(model) != str(base_model):
        raise RuntimeError(f"[commit] unexpected promote model={model} (base_model={base_model})")

    model_dir = _require_xcom(ti, key=K_MODEL_DIR, task_ids=_mat_task_ids(), hint="materialize must push model_dir")
    deploy_version = int(
        _require_xcom(ti, key=K_DEPLOY_VERSION, task_ids=_mat_task_ids(), hint="materialize must push deploy_version")
    )
    run_id = _xcom_pull_any(ti, key=K_RUN_ID, task_ids=_mat_task_ids())
    alias = _xcom_pull_any(ti, key=K_ALIAS, task_ids=_mat_task_ids())

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
    model = _xcom_pull_any(ti, key=K_MODEL, task_ids=_mat_task_ids())
    model_dir = _xcom_pull_any(ti, key=K_MODEL_DIR, task_ids=_mat_task_ids())
    deploy_v = _xcom_pull_any(ti, key=K_DEPLOY_VERSION, task_ids=_mat_task_ids())
    deploy_mode = _xcom_pull_any(ti, key=K_DEPLOY_MODE, task_ids=_mat_task_ids())

    prev = _xcom_pull_any(ti, key=K_PREV_CURRENT, task_ids=_snapshot_task_ids())

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

        # 혹시 과거의 "숫자 failed dir"이 남아있으면 먼저 제거
        _sanitize_numeric_failed_dirs(md)

        # 이번 deploy_v 버전을 안전한 형태로 격리
        _quarantine_failed_version_dir(model_dir=md, deploy_v=int(deploy_v))

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
    _sanitize_numeric_failed_dirs(model_dir)

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
