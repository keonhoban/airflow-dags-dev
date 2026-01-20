from __future__ import annotations

from datetime import datetime, timedelta
from pendulum import timezone

from airflow import DAG
from airflow.operators.python import PythonOperator, BranchPythonOperator
from airflow.operators.bash import BashOperator
from airflow.sensors.python import PythonSensor
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
    dag_id="full_e2e_dev",
    default_args=default_args,
    schedule=None,
    catchup=False,
    max_active_runs=1,
    tags=["e2e", "dev", "feast", "triton", "mlops"],
    on_failure_callback=alert_slack,
) as dag:

    # -----------------------
    # Data pipeline
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

    feast_apply = BashOperator(
        task_id="feast_apply",
        bash_command="""
        set -euo pipefail
        cd /opt/airflow/dags/repo/dags/feast_repo
        feast apply
        """.strip(),
    )

    feast_materialize = BashOperator(
        task_id="feast_materialize",
        bash_command="""
        set -euo pipefail
        cd /opt/airflow/dags/repo/dags/feast_repo
        feast materialize "{{ macros.ds_add(ds, -1) }}T00:00:00" "{{ ds }}T23:59:59"
        """.strip(),
    )

    summarize_run = PythonOperator(
        task_id="summarize_run",
        python_callable=p.dp_summary,
        trigger_rule=TriggerRule.ALL_DONE,
    )

    # -----------------------
    # Train/Register
    # -----------------------
    train = PythonOperator(
        task_id="train_and_evaluate",
        python_callable=p.train_and_evaluate,
    )

    branch = BranchPythonOperator(
        task_id="check_result",
        python_callable=p.check_result,
    )

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

    failure = PythonOperator(
        task_id="notify_failure",
        python_callable=p.notify_failure,
        trigger_rule=TriggerRule.NONE_FAILED_MIN_ONE_SUCCESS,
    )

    # -----------------------
    # Triton deploy
    # -----------------------
    snap = PythonOperator(
        task_id="snapshot_current",
        python_callable=p.snapshot_current,
    )

    mat = PythonOperator(
        task_id="materialize_repo",
        python_callable=p.triton_materialize_task,
    )

    load = PythonOperator(
        task_id="triton_load",
        python_callable=p.triton_load,
    )

    ready = PythonOperator(
        task_id="triton_ready",
        python_callable=p.triton_ready,
    )

    smoke = PythonOperator(
        task_id="triton_infer_smoke",
        python_callable=p.triton_infer_smoke,
    )

    commit = PythonOperator(
        task_id="commit_current",
        python_callable=p.commit_current,
    )

    rb = PythonOperator(
        task_id="rollback_minimal",
        python_callable=p.triton_rollback_task,
        trigger_rule=TriggerRule.ONE_FAILED,
    )

    fastapi_reload = PythonOperator(
        task_id="fastapi_reload",
        python_callable=p.fastapi_reload_task,
        trigger_rule=TriggerRule.ALL_SUCCESS,
    )

    # -----------------------
    # Dependencies
    # -----------------------
    extract_raw_data >> validate_data >> build_features >> store_features >> feast_apply >> feast_materialize
    feast_materialize >> train >> branch
    branch >> [register, failure]
    register >> sensor >> snap >> mat >> load >> ready >> smoke >> commit >> fastapi_reload
    [mat, load, ready, smoke, commit] >> rb
    [feast_materialize, fastapi_reload, rb, failure] >> summarize_run
