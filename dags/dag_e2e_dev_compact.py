# dags/dag_e2e_dev_compact.py
from __future__ import annotations

from datetime import datetime, timedelta
from pendulum import timezone

from airflow import DAG
from airflow.operators.python import PythonOperator, BranchPythonOperator
from airflow.operators.empty import EmptyOperator
from airflow.sensors.python import PythonSensor
from airflow.utils.trigger_rule import TriggerRule

from utils.slack_alerts import alert_slack
from pipelines import e2e_compact as p

kst = timezone("Asia/Seoul")
default_args = {
    "start_date": datetime(2025, 1, 1, tzinfo=kst),
    "retries": 1,
    "retry_delay": timedelta(minutes=2),
}

with DAG(
    dag_id="e2e_dev_compact",
    default_args=default_args,
    schedule=None,
    catchup=False,
    max_active_runs=1,
    tags=["e2e", "compact", "dev"],
    on_failure_callback=alert_slack,
) as dag:

    # -----------------------
    # Data Pipeline (DP)
    # -----------------------
    extract_raw = PythonOperator(
        task_id="extract_raw_data",
        python_callable=p.dp_extract,
    )

    validate = PythonOperator(
        task_id="validate_data",
        python_callable=p.dp_validate,
    )

    build = PythonOperator(
        task_id="build_features",
        python_callable=p.dp_build,
    )

    store = PythonOperator(
        task_id="store_features",
        python_callable=p.dp_store,
    )

    # -----------------------
    # Train / Branch
    # -----------------------
    train = PythonOperator(
        task_id="train_and_eval",
        python_callable=p.train_and_eval,
    )

    branch = BranchPythonOperator(
        task_id="check_result",
        python_callable=p.check_result,  # pass -> promo_start / fail -> shadow_start
    )

    promo_start = EmptyOperator(task_id="promo_start")
    shadow_start = EmptyOperator(task_id="shadow_start")

    notify_fail = PythonOperator(
        task_id="notify_failure",
        python_callable=p.notify_failure,
        trigger_rule=TriggerRule.NONE_FAILED_MIN_ONE_SUCCESS,
    )

    # -----------------------
    # Deploy common
    # -----------------------
    snapshot = PythonOperator(
        task_id="snapshot_current",
        python_callable=p.snapshot_current,
        trigger_rule=TriggerRule.NONE_FAILED_MIN_ONE_SUCCESS,
    )

    mat_promo = PythonOperator(
        task_id="triton_materialize_promo",
        python_callable=p.triton_materialize_promo,
        trigger_rule=TriggerRule.NONE_FAILED_MIN_ONE_SUCCESS,
    )

    mat_shadow = PythonOperator(
        task_id="triton_materialize_shadow",
        python_callable=p.triton_materialize_shadow,
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

    rollback = PythonOperator(
        task_id="rollback_minimal",
        python_callable=p.rollback_minimal,
        trigger_rule=TriggerRule.ONE_FAILED,
    )

    # -----------------------
    # Promotion-only
    # -----------------------
    commit = PythonOperator(
        task_id="commit_current",
        python_callable=p.commit_current,
        trigger_rule=TriggerRule.ALL_SUCCESS,
    )

    fastapi_reload = PythonOperator(
        task_id="fastapi_reload",
        python_callable=p.fastapi_reload,
        trigger_rule=TriggerRule.ALL_SUCCESS,
    )

    # -----------------------
    # Summary (always)
    # -----------------------
    summarize = PythonOperator(
        task_id="summarize_run",
        python_callable=p.summarize_run,
        trigger_rule=TriggerRule.ALL_DONE,
    )

    # -----------------------
    # Dependencies
    # -----------------------
    extract_raw >> validate >> build >> store >> train >> branch

    branch >> promo_start
    branch >> shadow_start >> notify_fail

    # 공통: snapshot 이후 promo/shadow 중 하나만 materialize 실행
    promo_start >> snapshot >> mat_promo
    shadow_start >> snapshot >> mat_shadow

    # deploy chain
    [mat_promo, mat_shadow] >> load >> ready >> smoke

    # promotion-only
    smoke >> commit >> fastapi_reload

    # rollback on failure
    [mat_promo, mat_shadow, load, ready, smoke, commit] >> rollback

    # summarize
    [fastapi_reload, rollback, notify_fail, train] >> summarize

