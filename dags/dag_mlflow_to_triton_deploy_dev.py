# dags/dag_mlflow_to_triton_deploy_dev.py

from airflow import DAG
from airflow.operators.python import PythonOperator
from datetime import datetime
from pendulum import timezone

from ml_code.triton_deploy import (
    materialize,
    triton_load,
    triton_ready,
    triton_infer_smoke,
    commit_current,
)

kst = timezone("Asia/Seoul")

with DAG(
    dag_id="mlflow_to_triton_min_dev",
    start_date=datetime(2025, 1, 1, tzinfo=kst),
    schedule=None,
    catchup=False,
    tags=["w6", "triton", "dev"],
    params={
        "alias": "A",
    },
) as dag:

    t1 = PythonOperator(
        task_id="materialize_repo",
        python_callable=materialize,
        op_kwargs={"alias": "{{ params.alias }}"},
    )

    t2 = PythonOperator(
        task_id="triton_load",
        python_callable=triton_load,
    )

    t3 = PythonOperator(
        task_id="triton_ready",
        python_callable=triton_ready,
    )

    t4 = PythonOperator(
        task_id="triton_infer_smoke",
        python_callable=triton_infer_smoke,
    )

    t5 = PythonOperator(
        task_id="commit_current",
        python_callable=commit_current,
    )

    t1 >> t2 >> t3 >> t4 >> t5
