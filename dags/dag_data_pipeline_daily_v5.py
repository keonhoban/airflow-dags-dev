# dags/dag_data_pipeline_daily_v5.py
from __future__ import annotations

from datetime import datetime, timedelta
from pendulum import timezone

from airflow import DAG
from airflow.operators.python import PythonOperator
from airflow.operators.bash import BashOperator
from airflow.utils.trigger_rule import TriggerRule

from utils.slack_alerts import alert_slack
from mlops_lib.dp.tasks import (
    task_extract_raw_data,
    task_validate_data,
    task_build_features,
    task_store_features,
    task_summarize_run,
)

kst = timezone("Asia/Seoul")

default_args = {
    "start_date": datetime(2025, 1, 1, tzinfo=kst),
    "retries": 1,
    "retry_delay": timedelta(minutes=3),
}

with DAG(
    dag_id="data_pipeline_daily_dev_v5",   # ✅ v5로 변경
    default_args=default_args,
    schedule=None,  # 수동 실행
    catchup=False,
    max_active_runs=1,
    tags=["data-pipeline", "dev", "mlops", "feast"],
    on_failure_callback=alert_slack,
) as dag:

    t_extract = PythonOperator(
        task_id="extract_raw_data",
        python_callable=task_extract_raw_data,
    )

    t_validate = PythonOperator(
        task_id="validate_data",
        python_callable=task_validate_data,
    )

    t_build = PythonOperator(
        task_id="build_features",
        python_callable=task_build_features,
    )

    t_store = PythonOperator(
        task_id="store_features",
        python_callable=task_store_features,
    )

    # ✅ Feast repo는 Airflow DAG 폴더에 둔 것으로 가정:
    # /opt/airflow/dags/feast_repo/feature_store.yaml, repo.py
    t_feast_apply = BashOperator(
        task_id="feast_apply",
        bash_command="""
set -euo pipefail
cd /opt/airflow/dags/feast_repo
feast apply
""".strip(),
    )

    t_feast_materialize = BashOperator(
        task_id="feast_materialize_incremental",
        bash_command="""
set -euo pipefail
cd /opt/airflow/dags/feast_repo
feast materialize-incremental "{{ macros.ds_add(ds, -1) }}T00:00:00" "{{ ds }}T23:59:59"
""".strip(),
    )

    t_summary = PythonOperator(
        task_id="summarize_run",
        python_callable=task_summarize_run,
        trigger_rule=TriggerRule.ALL_DONE,
    )

    # ✅ dependency 업데이트
    t_extract >> t_validate >> t_build >> t_store >> t_feast_apply >> t_feast_materialize >> t_summary
