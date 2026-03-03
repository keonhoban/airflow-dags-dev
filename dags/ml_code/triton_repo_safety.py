# dags/ml_code/triton_repo_safety.py
from __future__ import annotations

import os

from airflow.utils.log.logging_mixin import LoggingMixin

from ml_code.triton_actions import utc_ts

log = LoggingMixin().log


def ensure_writable_quarantine_dir(model_dir: str) -> str:
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

    # 마지막: 그래도 안되면 q1 반환(기존 동작 유지)
    return q1


def sanitize_numeric_failed_dirs(model_dir: str) -> None:
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

        qdir = ensure_writable_quarantine_dir(model_dir)

        for name in os.listdir(model_dir):
            if not name:
                continue
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


def quarantine_failed_version_dir(*, model_dir: str, deploy_v: int) -> None:
    """
    실패 버전 디렉토리를 안전하게 격리합니다.

    ❌ 금지: "56.failed_..." 같은 형태 (숫자로 시작) => Triton이 version=56으로 해석 가능
    ✅ 권장: _quarantine/failed_v{version}_{ts}
    """
    ver_dir = os.path.join(model_dir, str(deploy_v))
    if not os.path.isdir(ver_dir):
        return

    quarantine_dir = ensure_writable_quarantine_dir(model_dir)

    failed_name = f"failed_v{deploy_v}_{utc_ts()}"
    failed_dir = os.path.join(quarantine_dir, failed_name)

    try:
        os.rename(ver_dir, failed_dir)
        log.warning("[ROLLBACK] moved failed dir: %s -> %s", ver_dir, failed_dir)
    except Exception as e:
        log.warning("[ROLLBACK] failed to move dir: %s -> %s err=%s", ver_dir, failed_dir, e)
