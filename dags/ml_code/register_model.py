# dags/ml_code/register_model.py
from __future__ import annotations

from typing import Optional

import mlflow
from airflow.utils.log.logging_mixin import LoggingMixin

from ml_code.config import get_mlflow_client

logger = LoggingMixin().log


def _normalize(s: Optional[str]) -> str:
    return (str(s) if s is not None else "").strip()


def _ensure_registered_model(client, model_name: str) -> None:
    try:
        client.create_registered_model(model_name)
        logger.info("[Register] model created: %s", model_name)
    except mlflow.exceptions.RestException as e:
        # MLflow backend마다 메시지/코드가 살짝 달라서 문자열도 같이 봅니다.
        if "RESOURCE_ALREADY_EXISTS" in str(e) or "already exists" in str(e).lower():
            logger.info("[Register] model exists: %s", model_name)
            return
        raise


def _find_version_by_run_id(client, model_name: str, run_id: str) -> Optional[str]:
    """
    같은 run_id가 이미 model version으로 존재하면 그 version을 재사용합니다.
    (중복 버전 생성 방지: 운영에서 꽤 중요)
    """
    try:
        versions = client.search_model_versions(f"name='{model_name}'")
    except Exception as e:
        logger.warning("[Register] search_model_versions failed: %s", e)
        return None

    for mv in versions or []:
        try:
            if str(getattr(mv, "run_id", "")).strip() == run_id:
                v = str(getattr(mv, "version", "")).strip()
                if v:
                    return v
        except Exception:
            continue
    return None


def register_model(
    run_id: str,
    model_name: str,
    mlflow_alias: str,
    *,
    tags_env: Optional[str] = None,
    tags_fs_version: Optional[str] = None,
    tags_schema_hash: Optional[str] = None,
) -> str:
    """
    - Registered Model이 없으면 생성
    - Model Version 생성(단, 동일 run_id가 이미 등록돼 있으면 재사용)
    - Alias를 해당 version으로 이동
    Returns: version (str)
    """
    client = get_mlflow_client()

    model_name = _normalize(model_name)
    mlflow_alias = _normalize(mlflow_alias)
    run_id = _normalize(run_id)

    if not model_name or not mlflow_alias or not run_id:
        raise ValueError(f"invalid args: run_id={run_id!r}, model_name={model_name!r}, alias={mlflow_alias!r}")

    _ensure_registered_model(client, model_name)

    # 1) 중복 등록 방지: 같은 run_id 버전이 있으면 재사용
    existing_version = _find_version_by_run_id(client, model_name, run_id)
    if existing_version:
        version = existing_version
        logger.info("[Register] reuse existing version: %s v%s (run_id=%s)", model_name, version, run_id)
    else:
        # 2) 신규 버전 생성
        result = client.create_model_version(
            name=model_name,
            source=f"runs:/{run_id}/model",
            run_id=run_id,
        )
        version = str(result.version).strip()
        logger.info("[Register] version created: %s v%s (run_id=%s)", model_name, version, run_id)

        # 3) (선택) 모델버전 태그로 lineage 남기기: 운영/디버깅에 도움
        try:
            if tags_env:
                client.set_model_version_tag(model_name, version, "env", str(tags_env))
            if tags_fs_version:
                client.set_model_version_tag(model_name, version, "fs_version", str(tags_fs_version))
            if tags_schema_hash:
                client.set_model_version_tag(model_name, version, "schema_hash", str(tags_schema_hash))
            client.set_model_version_tag(model_name, version, "source_run_id", run_id)
        except Exception as e:
            logger.warning("[Register] set_model_version_tag failed (ignore): %s", e)

    # 4) Alias 이동: delete -> set 패턴은 불필요한 예외를 만들 수 있어 set 중심으로 처리
    try:
        client.set_registered_model_alias(model_name, mlflow_alias, version)
        logger.info("[Alias] %s v%s -> @%s", model_name, version, mlflow_alias)
    except Exception as e:
        # 일부 MLflow backend/버전에서 alias API 지원이 다를 수 있어 로그를 남기고 실패
        raise RuntimeError(f"[Alias] set_registered_model_alias failed model={model_name} alias={mlflow_alias} v={version} err={e}") from e

    return version
