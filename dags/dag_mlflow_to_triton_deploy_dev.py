# dags/dag_mlflow_to_triton_deploy_dev.py

from airflow import DAG
from airflow.operators.python import PythonOperator
from airflow.utils.trigger_rule import TriggerRule
from datetime import datetime
from pendulum import timezone

from ml_code.triton_deploy import (
    snapshot_current,
    materialize,
    triton_load,
    triton_ready,
    triton_infer_smoke,
    commit_current,
    rollback_minimal,
)

kst = timezone("Asia/Seoul")

with DAG(
    dag_id="mlflow_to_triton_min_dev",
    start_date=datetime(2025, 1, 1, tzinfo=kst),
    schedule=None,
    catchup=False,
    max_active_runs=1,
    tags=["w6", "triton", "dev"],
    params={"alias": "A"},
) as dag:

    POOL = "triton_deploy"

    t0 = PythonOperator(
        task_id="snapshot_current",
        python_callable=snapshot_current,
        pool=POOL,
    )

    t1 = PythonOperator(
        task_id="materialize_repo",
        python_callable=materialize,
        op_kwargs={"alias": "{{ params.alias }}"},
        pool=POOL,
    )

    t2 = PythonOperator(
        task_id="triton_load",
        python_callable=triton_load,
        pool=POOL,
    )

    t_ready = PythonOperator(
        task_id="triton_ready",
        python_callable=triton_ready,
        pool=POOL,
    )

    t_smoke = PythonOperator(
        task_id="triton_infer_smoke",
        python_callable=triton_infer_smoke,
        pool=POOL,
    )

    t3 = PythonOperator(
        task_id="commit_current",
        python_callable=commit_current,
        trigger_rule=TriggerRule.ALL_SUCCESS,
        pool=POOL,
    )

    t_rb = PythonOperator(
        task_id="rollback_minimal",
        python_callable=rollback_minimal,
        trigger_rule=TriggerRule.ONE_FAILED,
        pool=POOL,
    )

    # success path
    t0 >> t1 >> t2 >> t_ready >> t_smoke >> t3

    # rollback path (snapshot도 upstream에 포함)
    [t0, t1, t2, t_ready, t_smoke] >> t_rb
