from __future__ import annotations

from airflow import DAG
from airflow.operators.python import PythonOperator, BranchPythonOperator
from airflow.operators.empty import EmptyOperator
from airflow.utils.trigger_rule import TriggerRule
from datetime import datetime, timedelta

from pipelines.e2e_pipeline import (
    dp_build_and_store,
    train_and_log,
    branch_on_accuracy,
    register_and_alias,
    wait_model_ready,
    triton_snapshot_current,
    triton_materialize_and_smoke,
    triton_commit_current,
    triton_rollback_minimal,
    fastapi_reload,
    notify_shadow_skip,
)

default_args = {
    "owner": "mlops",
    "retries": 1,
    "retry_delay": timedelta(minutes=3),
}

with DAG(
    dag_id="e2e_full",
    start_date=datetime(2025, 1, 1),
    schedule=None,
    catchup=False,
    default_args=default_args,
    dagrun_timeout=timedelta(hours=2),
    tags=["mlops", "e2e"],
) as dag:
    start = EmptyOperator(task_id="start")

    dp = PythonOperator(
        task_id="dp_build_and_store",
        python_callable=dp_build_and_store,
    )

    train = PythonOperator(
        task_id="train_and_log",
        python_callable=train_and_log,
    )

    branch = BranchPythonOperator(
        task_id="branch_on_accuracy",
        python_callable=branch_on_accuracy,
    )

    shadow = PythonOperator(
        task_id="shadow_start",
        python_callable=notify_shadow_skip,
    )

    register = PythonOperator(
        task_id="register_and_alias",
        python_callable=register_and_alias,
    )

    sensor = PythonOperator(
        task_id="wait_model_ready",
        python_callable=wait_model_ready,
    )

    snap = PythonOperator(
        task_id="triton_snapshot_current",
        python_callable=triton_snapshot_current,
    )

    deploy_smoke = PythonOperator(
        task_id="triton_materialize_and_smoke",
        python_callable=triton_materialize_and_smoke,
    )

    commit = PythonOperator(
        task_id="triton_commit_current",
        python_callable=triton_commit_current,
    )

    reload_api = PythonOperator(
        task_id="fastapi_reload",
        python_callable=fastapi_reload,
    )

    rollback = PythonOperator(
        task_id="triton_rollback_minimal",
        python_callable=triton_rollback_minimal,
        trigger_rule=TriggerRule.ONE_FAILED,
    )

    end = EmptyOperator(task_id="end", trigger_rule=TriggerRule.NONE_FAILED_MIN_ONE_SUCCESS)

    start >> dp >> train >> branch
    branch >> register >> sensor >> snap >> deploy_smoke >> commit >> reload_api >> end

    branch >> shadow >> end

    # 실패 시 롤백(프로모션 경로에서만 의미 있음)
    [snap, deploy_smoke, commit, reload_api] >> rollback >> end

