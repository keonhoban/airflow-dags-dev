# dags/dag_data_pipeline_daily_v3.py

from __future__ import annotations

from datetime import datetime, timedelta
from pendulum import timezone

from airflow import DAG
from airflow.operators.python import PythonOperator
from airflow.utils.trigger_rule import TriggerRule
from airflow.sdk import Variable

from utils.slack_alerts import alert_slack
from ml_code.data_pipeline_v3 import (
    extract_raw_data,
    validate_data,
    build_features,
    store_features,
    summarize_run,
)

kst = timezone("Asia/Seoul")

default_args = {
    "start_date": datetime(2025, 1, 1, tzinfo=kst),
    "retries": 1,
    "retry_delay": timedelta(minutes=3),
}


def _get_pipeline_config() -> dict:
    """
    Airflow Variable 기반 파이프라인 설정.

    Variables:
      - dp_raw_path: s3://<bucket>/<key>.csv
      - dp_feature_base: s3://<bucket>/<prefix>/feature-store
          (v3는 feature_set/version을 내부에서 붙여서 저장)
      - dp_feature_set: e.g., user_features
      - dp_pipeline_name: e.g., daily_user_events
    """
    try:
        raw_path = Variable.get(
            "dp_raw_path",
            default_var="s3://datapipeline-raw-data-keonho/daily/user_events_20251119.csv",
        )

        # ✅ v3의 핵심: "base" 경로 (여기에 feature_set/version을 붙여 저장)
        feature_base = Variable.get(
            "dp_feature_base",
            default_var="s3://datapipeline-raw-data-keonho/feature-store",
        )

        # ✅ schema의 feature_set과 반드시 일치해야 함
        feature_set = Variable.get(
            "dp_feature_set",
            default_var="user_features",
        )

        pipeline_name = Variable.get(
            "dp_pipeline_name",
            default_var="daily_user_events",
        )

    except Exception:
        raw_path = "s3://datapipeline-raw-data-keonho/daily/user_events_20251119.csv"
        feature_base = "s3://datapipeline-raw-data-keonho/feature-store"
        feature_set = "user_features"
        pipeline_name = "daily_user_events"

    return {
        "raw_path": raw_path,
        "feature_base": feature_base,
        "feature_set": feature_set,
        "pipeline_name": pipeline_name,
    }


# --------------------------
# Task wrappers
# --------------------------
def task_extract_raw_data(**context):
    cfg = _get_pipeline_config()
    extract_raw_data(
        raw_path=cfg["raw_path"],
        pipeline_name=cfg["pipeline_name"],
        ti=context["ti"],
    )


def task_validate_data(**context):
    cfg = _get_pipeline_config()
    validate_data(
        raw_path=cfg["raw_path"],
        pipeline_name=cfg["pipeline_name"],
        ti=context["ti"],
    )


def task_build_features(**context):
    cfg = _get_pipeline_config()
    build_features(
        raw_path=cfg["raw_path"],
        feature_base=cfg["feature_base"],
        pipeline_name=cfg["pipeline_name"],
        feature_set=cfg["feature_set"],
        ti=context["ti"],
    )


def task_store_features(**context):
    cfg = _get_pipeline_config()
    store_features(
        feature_base=cfg["feature_base"],
        pipeline_name=cfg["pipeline_name"],
        feature_set=cfg["feature_set"],
        ti=context["ti"],
    )


def task_summarize_run(**context):
    cfg = _get_pipeline_config()
    summarize_run(
        pipeline_name=cfg["pipeline_name"],
        feature_set=cfg["feature_set"],
        ti=context["ti"],
    )


with DAG(
    dag_id="data_pipeline_daily_dev_v3",
    default_args=default_args,
    schedule=None,  # 수동 실행
    catchup=False,
    max_active_runs=1,
    tags=["data-pipeline", "dev", "mlops"],
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

    t_summary = PythonOperator(
        task_id="summarize_run",
        python_callable=task_summarize_run,
        trigger_rule=TriggerRule.ALL_DONE,
    )

    t_extract >> t_validate >> t_build >> t_store >> t_summary
