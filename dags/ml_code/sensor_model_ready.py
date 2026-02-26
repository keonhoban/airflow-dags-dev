# dags/ml_code/sensor_model_ready.py
from __future__ import annotations

from dataclasses import dataclass

from airflow.utils.log.logging_mixin import LoggingMixin

from ml_code.config import get_mlflow_client

logger = LoggingMixin().log


@dataclass(frozen=True)
class ModelReadyResult:
    model_name: str
    version: str
    status: str


def _normalize(s: str) -> str:
    return (str(s) if s is not None else "").strip()


def check_model_ready(model_name: str, version: str) -> bool:
    """
    MLflow ModelVersion 상태가 READY가 될 때까지 기다리는 Sensor용 함수.

    반환:
      - READY: True
      - 대기 상태(PENDING_*): False
      - 실패/알 수 없는 상태: 예외 (즉시 fail)
    """
    model_name = _normalize(model_name)
    version = _normalize(version)
    if not model_name or not version:
        raise ValueError(f"[Sensor] invalid args: model_name={model_name!r}, version={version!r}")

    client = get_mlflow_client()
    mv = client.get_model_version(name=model_name, version=version)

    status = _normalize(getattr(mv, "status", ""))
    logger.info("[Sensor] %s v%s status=%s", model_name, version, status)

    # ✅ success
    if status == "READY":
        return True

    # ✅ fail-fast: 명확히 실패 상태
    if status in ("FAILED_REGISTRATION", "FAILED"):
        raise RuntimeError(f"[Sensor] model registration failed: model={model_name} v={version} status={status}")

    # ✅ wait: registration 진행 중
    if status in ("PENDING_REGISTRATION", "RUNNING_REGISTRATION", "PENDING"):
        return False

    # ✅ unknown status는 운영 리스크: 제출용/운영용 기준으로 fail 권장
    raise RuntimeError(f"[Sensor] unknown model status: model={model_name} v={version} status={status}")
