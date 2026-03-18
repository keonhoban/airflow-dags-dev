# dags/rollback_manual.py
from __future__ import annotations

import re
from datetime import datetime, timezone

from airflow.decorators import dag, task
from airflow.models import DagRun

from ml_code.config import cfg
from mlops_lib.rollback.types import build_ctx_from_airflow
from mlops_lib.rollback.repo import (
    require_version_dir,
    write_current_json,
    update_config_pbtxt_specific_version,
)
from mlops_lib.rollback.triton import (
    triton_unload,
    triton_load,
    triton_wait_ready,
)
from mlops_lib.rollback.fastapi import (
    fastapi_reload_service,
    fastapi_wait_ssot_converged,
)


@dag(
    dag_id="rollback_manual",
    start_date=datetime(2026, 1, 1, tzinfo=timezone.utc),
    schedule=None,
    catchup=False,
    max_active_runs=1,
    default_args={"retries": 0},
    tags=["ops", "rollback"],
)
def rollback_manual_dag():
    @task
    def validate_conf(dag_run: DagRun | None = None) -> None:
        """
        dag_run.conf 필드와 필수 Variable을 조기에 검증한다.

        이유:
          - 잘못된 deploy_version/alias 입력이 SSOT 파일(current.json, config.pbtxt)을
            손상시키기 전에 DAG를 즉시 실패시킨다.
          - path traversal 방지: model_name에 '/', '..' 포함 시 차단.
          - 부분 실행 방지: 검증이 통과해야만 write_ssot_files가 실행된다.

        검증 항목:
          1. deploy_version: conf 또는 Variable에서 양의 정수여야 함
          2. alias: 영숫자·하이픈·언더스코어만 허용 (Triton 모델 alias 제약)
          3. triton_model_name: path separator 및 '..' 포함 금지
        """
        conf = dict(getattr(dag_run, "conf", None) or {})
        errors: list[str] = []

        # --- 1. deploy_version ---
        raw_version = conf.get("deploy_version")
        if raw_version is not None:
            try:
                v = int(str(raw_version).strip())
                if v <= 0:
                    errors.append(
                        f"deploy_version must be a positive integer, got: {raw_version!r}"
                    )
            except (ValueError, TypeError):
                errors.append(
                    f"deploy_version must be castable to int, got: {raw_version!r}"
                )

        # --- 2. alias ---
        alias = conf.get("alias")
        if alias is not None:
            alias_str = str(alias).strip()
            if not alias_str or not re.fullmatch(r"[A-Za-z0-9_\-]+", alias_str):
                errors.append(
                    f"alias must match [A-Za-z0-9_-] (non-empty), got: {alias!r}"
                )

        # --- 3. triton_model_name: path traversal 방지 ---
        # conf에 model_name 오버라이드가 없어도 Variable 값 자체를 검증한다.
        model_name = str(cfg("triton_model_name", "") or "").strip()
        if not model_name:
            errors.append(
                "Airflow Variable 'triton_model_name' is not set or empty. "
                "Set it before triggering rollback_manual."
            )
        elif "/" in model_name or "\\" in model_name or ".." in model_name:
            errors.append(
                f"triton_model_name must be a simple identifier (no path separators or '..'), "
                f"got: {model_name!r}"
            )

        if errors:
            raise ValueError(
                "rollback_manual: conf/Variable validation failed\n"
                + "\n".join(f"  [{i+1}] {e}" for i, e in enumerate(errors))
            )

    @task
    def load_ctx(dag_run: DagRun | None = None) -> dict:
        # dict로 반환해서 XCom 안정성 확보
        ctx = build_ctx_from_airflow(dag_run=dag_run)
        return ctx.to_dict()

    @task
    def guard_repo(ctx: dict) -> dict:
        require_version_dir(ctx)
        return ctx

    @task
    def write_ssot_files(ctx: dict) -> dict:
        write_current_json(ctx)
        update_config_pbtxt_specific_version(ctx)
        return ctx

    @task
    def rollback_triton(ctx: dict) -> dict:
        triton_unload(ctx)
        triton_load(ctx)
        triton_wait_ready(ctx)
        return ctx

    @task
    def reload_fastapi(ctx: dict) -> dict:
        resp = fastapi_reload_service(ctx)
        # 필요하면 resp를 XCom으로 남겨 증거로 활용 가능
        return {"ctx": ctx, "fastapi_reload_resp": resp}

    @task
    def verify_convergence(payload: dict) -> None:
        ctx = payload["ctx"]
        fastapi_wait_ssot_converged(ctx)

    validated = validate_conf()
    ctx = load_ctx()
    ctx.set_upstream(validated)   # validate 통과 후에만 load_ctx 실행
    ctx = guard_repo(ctx)
    ctx = write_ssot_files(ctx)
    ctx = rollback_triton(ctx)
    payload = reload_fastapi(ctx)
    verify_convergence(payload)


rollback_manual_dag()
