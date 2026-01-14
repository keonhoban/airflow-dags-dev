from datetime import datetime, timedelta
from pendulum import timezone
import os
import subprocess

from airflow import DAG
from airflow.operators.python import PythonOperator

kst = timezone("Asia/Seoul")

default_args = {
    "start_date": datetime(2025, 1, 1, tzinfo=kst),
    "retries": 1,
    "retry_delay": timedelta(minutes=3),
}

FEAST_REPO = "/opt/airflow/dags/repo/dags/mlops_lib/feast_repo"

def _run(cmd: list[str]):
    p = subprocess.run(cmd, cwd=FEAST_REPO, capture_output=True, text=True)
    if p.returncode != 0:
        raise RuntimeError(f"cmd failed: {cmd}\nstdout:\n{p.stdout}\nstderr:\n{p.stderr}")
    return p.stdout

def feast_apply():
    _run(["feast", "apply"])

def feast_materialize_incremental():
    # 최근 1일만 적재 (원하면 기간 늘리기)
    _run(["feast", "materialize-incremental", (datetime.utcnow() - timedelta(days=1)).isoformat()])

with DAG(
    dag_id="feast_materialize_dev",
    default_args=default_args,
    schedule="*/30 * * * *",  # 30분마다
    catchup=False,
    max_active_runs=1,
    tags=["feast", "feature-store", "dev"],
) as dag:
    t_apply = PythonOperator(task_id="feast_apply", python_callable=feast_apply)
    t_mat = PythonOperator(task_id="feast_materialize", python_callable=feast_materialize_incremental)
    t_apply >> t_mat
