# dags/dp_feature_pipeline.py
from __future__ import annotations

from datetime import datetime, timedelta
from pendulum import timezone

from airflow import DAG
from airflow.operators.python import PythonOperator
from airflow.utils.trigger_rule import TriggerRule

from utils.slack_alerts import alert_slack
from pipelines import full_e2e as p
from mlops_lib.core.policy import E2E_START_DATE_YMD, E2E_RETRIES, E2E_RETRY_DELAY_MIN

kst = timezone("Asia/Seoul")
default_args = {
    "start_date": datetime(*E2E_START_DATE_YMD, tzinfo=kst),
    "retries": E2E_RETRIES,
    "retry_delay": timedelta(minutes=E2E_RETRY_DELAY_MIN),
}

with DAG(
    dag_id="dp_feature_pipeline",
    default_args=default_args,
    schedule=None,  # 원하면 cron으로 바꿔도 됨
    catchup=False,
    max_active_runs=1,
    tags=["dp", "feature-store", "mlops"],
    on_failure_callback=alert_slack,
) as dag:

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

    extract_raw_data >> validate_data >> build_features >> store_features >> summarize_run
