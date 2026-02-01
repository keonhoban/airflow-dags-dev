# dags/e2e_full.py
from __future__ import annotations

from datetime import datetime, timedelta
from pendulum import timezone

from airflow import DAG
from airflow.operators.python import PythonOperator, BranchPythonOperator
from airflow.sensors.python import PythonSensor
from airflow.operators.empty import EmptyOperator
from airflow.utils.trigger_rule import TriggerRule

from utils.slack_alerts import alert_slack
from pipelines import full_e2e as p

kst = timezone("Asia/Seoul")
default_args = {
    "start_date": datetime(2025, 1, 1, tzinfo=kst),
    "retries": 1,
    "retry_delay": timedelta(minutes=2),
}

with DAG(
    dag_id="e2e_full",
    default_args=default_args,
    schedule=None,
    catchup=False,
    max_active_runs=1,
    tags=["e2e", "mlops", "triton", "mlflow"],
    on_failure_callback=alert_slack,
) as dag:

    # -----------------------
    # Data pipeline (Feature Store-lite)
    # -----------------------
    extract_raw_data = PythonOperator(
        task_id="extract_raw_data",
        python_callable=p.dp_extract,
    )

    validate_data = PythonOperator(
        task_id="validate_data",
        python_callable=p.dp_validate,
    )

    build_features = PythonOperator(
        task_id="build_features",
        python_callable=p.dp_build,
    )

    store_features = PythonOperator(
        task_id="store_features",
        python_callable=p.dp_store,
    )

    summarize_run = PythonOperator(
        task_id="summarize_run",
        python_callable=p.dp_summary,
        trigger_rule=TriggerRule.ALL_DONE,
    )

    # -----------------------
    # Train / Branch
    # -----------------------
    train = PythonOperator(
        task_id="train_and_evaluate",
        python_callable=p.train_and_evaluate,
    )

    branch = BranchPythonOperator(
        task_id="check_result",
        python_callable=p.check_result,  # pass -> register_path / fail -> shadow_path
    )

    # -----------------------
    # Promotion path
    # -----------------------
    register = PythonOperator(
        task_id="register_model_task",
        python_callable=p.register_model_task,
    )

    sensor = PythonSensor(
        task_id="check_model_ready",
        python_callable=p.sensor_ready_func,
        poke_interval=10,
        timeout=180,
        mode="reschedule",
    )

    promotion_start = EmptyOperator(task_id="promotion_start")
    shadow_start = EmptyOperator(task_id="shadow_start")

    notify_failure = PythonOperator(
        task_id="notify_failure",
        python_callable=p.notify_failure,
        trigger_rule=TriggerRule.NONE_FAILED_MIN_ONE_SUCCESS,
    )

    # -----------------------
    # Common deploy chain (both paths)
    # -----------------------
    snap = PythonOperator(
        task_id="snapshot_current",
        python_callable=p.snapshot_current,
        trigger_rule=TriggerRule.NONE_FAILED_MIN_ONE_SUCCESS,
    )

    mat = PythonOperator(
        task_id="materialize_repo",
        python_callable=p.triton_materialize_task,
        trigger_rule=TriggerRule.NONE_FAILED_MIN_ONE_SUCCESS,
    )

    load = PythonOperator(
        task_id="triton_load",
        python_callable=p.triton_load,
        trigger_rule=TriggerRule.ALL_SUCCESS,
    )

    ready = PythonOperator(
        task_id="triton_ready",
        python_callable=p.triton_ready,
        trigger_rule=TriggerRule.ALL_SUCCESS,
    )

    smoke = PythonOperator(
        task_id="triton_infer_smoke",
        python_callable=p.triton_infer_smoke,
        trigger_rule=TriggerRule.ALL_SUCCESS,
    )

    rb = PythonOperator(
        task_id="rollback_minimal",
        python_callable=p.triton_rollback_task,
        trigger_rule=TriggerRule.ONE_FAILED,
    )

    # -----------------------
    # Promotion-only state changes
    # -----------------------
    commit = PythonOperator(
        task_id="commit_current",
        python_callable=p.commit_current,
        trigger_rule=TriggerRule.ALL_SUCCESS,
    )

    fastapi_reload = PythonOperator(
        task_id="fastapi_reload",
        python_callable=p.fastapi_reload_task,
        trigger_rule=TriggerRule.ALL_SUCCESS,
    )

    # -----------------------
    # Dependencies
    # -----------------------
    extract_raw_data >> validate_data >> build_features >> store_features
    store_features >> train >> branch

    branch >> register >> promotion_start
    branch >> shadow_start >> notify_failure

    promotion_start >> sensor >> snap
    shadow_start >> snap

    snap >> mat >> load >> ready >> smoke

    smoke >> commit >> fastapi_reload

    [mat, load, ready, smoke, commit] >> rb
    [store_features, fastapi_reload, rb, notify_failure] >> summarize_run

