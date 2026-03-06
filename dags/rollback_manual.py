# dags/rollback_manual.py
from __future__ import annotations

from datetime import datetime, timezone

from airflow.decorators import dag, task
from airflow.models import DagRun

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

    ctx = load_ctx()
    ctx = guard_repo(ctx)
    ctx = write_ssot_files(ctx)
    ctx = rollback_triton(ctx)
    payload = reload_fastapi(ctx)
    verify_convergence(payload)


rollback_manual_dag()
