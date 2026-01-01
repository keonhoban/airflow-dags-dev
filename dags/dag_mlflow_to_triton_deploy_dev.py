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
    tags=["w6", "triton", "dev"],
    params={"alias": "A"},
) as dag:

    t0 = PythonOperator(
        task_id="snapshot_current",
        python_callable=snapshot_current,
    )

    t1 = PythonOperator(
        task_id="materialize_repo",
        python_callable=materialize,
        op_kwargs={"alias": "{{ params.alias }}"},
    )

    t2 = PythonOperator(
        task_id="triton_load",
        python_callable=triton_load,
    )

    t_ready = PythonOperator(
        task_id="triton_ready",
        python_callable=triton_ready,
    )

    t_smoke = PythonOperator(
        task_id="triton_infer_smoke",
        python_callable=triton_infer_smoke,
    )

    t3 = PythonOperator(
        task_id="commit_current",
        python_callable=commit_current,
        trigger_rule=TriggerRule.ALL_SUCCESS,
    )

    t_rb = PythonOperator(
        task_id="rollback_minimal",
        python_callable=rollback_minimal,
        trigger_rule=TriggerRule.ONE_FAILED,
    )

    # success path
    t0 >> t1 >> t2 >> t_ready >> t_smoke >> t3

    # rollback path (snapshot도 upstream에 포함)
    [t0, t1, t2, t_ready, t_smoke] >> t_rb
