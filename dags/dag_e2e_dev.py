# dags/dag_e2e_dev.py
from __future__ import annotations

from datetime import datetime, timedelta
from pendulum import timezone

from airflow import DAG
from airflow.operators.python import PythonOperator, ShortCircuitOperator
from airflow.operators.bash import BashOperator
from airflow.sensors.python import PythonSensor
from airflow.operators.empty import EmptyOperator
from airflow.utils.trigger_rule import TriggerRule

from utils.slack import alert_slack
from mlops_lib.dp.tasks import (
    task_extract_raw_data,
    task_validate_data,
    task_build_features,
    task_store_features,
    task_summarize_run,
)
from pipelines.e2e import (
    task_train_and_eval,
    task_gate_promotion,
    task_register_if_promoted,
    task_wait_model_ready_if_promoted,
    task_snapshot_current,
    task_triton_materialize_shadow,
    task_triton_load,
    task_triton_ready,
    task_triton_infer_smoke,
    task_triton_rollback,
    task_commit_current_if_promoted,
    task_fastapi_reload_if_promoted,
)

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
    tags=["e2e", "dev", "compact", "mlflow", "triton", "feast"],
    on_failure_callback=alert_slack,
) as dag:

    # -----------------------
    # 1) Data pipeline (DP)
    # -----------------------
    extract_raw_data = PythonOperator(
        task_id="extract_raw_data",
        python_callable=task_extract_raw_data,
    )

    validate_data = PythonOperator(
        task_id="validate_data",
        python_callable=task_validate_data,
    )

    build_features = PythonOperator(
        task_id="build_features",
        python_callable=task_build_features,
    )

    store_features = PythonOperator(
        task_id="store_features",
        python_callable=task_store_features,
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

    # -----------------------
    # 2) Train
    # -----------------------
    train_and_eval = PythonOperator(
        task_id="train_and_eval",
        python_callable=task_train_and_eval,
    )

    # -----------------------
    # 3) Deploy shadow (항상 수행)
    # -----------------------
    snapshot_current = PythonOperator(
        task_id="snapshot_current",
        python_callable=task_snapshot_current,
        trigger_rule=TriggerRule.ALL_DONE,  # "배포 시작 직전 스냅샷"은 최대한 실행
    )

    triton_materialize_shadow = PythonOperator(
        task_id="triton_materialize_shadow",
        python_callable=task_triton_materialize_shadow,
    )

    triton_load = PythonOperator(
        task_id="triton_load",
        python_callable=task_triton_load,
    )

    triton_ready = PythonOperator(
        task_id="triton_ready",
        python_callable=task_triton_ready,
    )

    triton_infer_smoke = PythonOperator(
        task_id="triton_infer_smoke",
        python_callable=task_triton_infer_smoke,
    )

    # deploy 구간 실패시에만 rollback
    rollback = PythonOperator(
        task_id="rollback_minimal",
        python_callable=task_triton_rollback,
        trigger_rule=TriggerRule.ONE_FAILED,
    )

    # -----------------------
    # 4) Promotion gate (조건부)
    # -----------------------
    gate_promotion = ShortCircuitOperator(
        task_id="gate_promotion",
        python_callable=task_gate_promotion,  # True면 아래 promotion 수행, False면 skip
        trigger_rule=TriggerRule.ALL_SUCCESS,
    )

    register_model = PythonOperator(
        task_id="register_model",
        python_callable=task_register_if_promoted,
    )

    wait_ready = PythonSensor(
        task_id="wait_model_ready",
        python_callable=task_wait_model_ready_if_promoted,
        poke_interval=10,
        timeout=180,
        mode="reschedule",
    )

    commit_current = PythonOperator(
        task_id="commit_current",
        python_callable=task_commit_current_if_promoted,
    )

    fastapi_reload = PythonOperator(
        task_id="fastapi_reload",
        python_callable=task_fastapi_reload_if_promoted,
    )

    end = EmptyOperator(task_id="end", trigger_rule=TriggerRule.ALL_DONE)

    summarize = PythonOperator(
        task_id="summarize_run",
        python_callable=task_summarize_run,
        trigger_rule=TriggerRule.ALL_DONE,
    )

    # -----------------------
    # Dependencies
    # -----------------------
    extract_raw_data >> validate_data >> build_features >> store_features >> feast_apply >> feast_materialize
    feast_materialize >> train_and_eval

    # 배포 검증(항상)
    train_and_eval >> snapshot_current >> triton_materialize_shadow >> triton_load >> triton_ready >> triton_infer_smoke

    # rollback은 배포 구간 실패만 연결
    [triton_materialize_shadow, triton_load, triton_ready, triton_infer_smoke] >> rollback

    # promotion은 "배포 검증 성공" 후 게이트
    triton_infer_smoke >> gate_promotion >> register_model >> wait_ready >> commit_current >> fastapi_reload >> end

    # 요약은 모든 경로 끝나고
    [end, rollback] >> summarize

