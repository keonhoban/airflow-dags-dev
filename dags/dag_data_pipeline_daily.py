# dags/dag_data_pipeline_daily.py

from datetime import datetime, timedelta
from pendulum import timezone

from airflow import DAG
from airflow.operators.python import PythonOperator
from airflow.utils.trigger_rule import TriggerRule

from airflow.sdk import Variable

from utils.slack_alerts import alert_slack, send_slack_alert
from ml_code.data_pipeline import (
    extract_raw_data,
    validate_data,
    build_features,
    store_features,
    summarize_run,
)

# KST 타임존
kst = timezone("Asia/Seoul")

default_args = {
    "start_date": datetime(2025, 1, 1, tzinfo=kst),
    "retries": 1,
    "retry_delay": timedelta(minutes=3),
}


def _get_pipeline_config():
    """
    파이프라인 설정 로딩:
      - dp_raw_path      : RAW 데이터 S3 경로
      - dp_feature_path  : Feature S3 경로
      - dp_pipeline_name : 파이프라인 이름

    Airflow Variable에 없으면 기본값 사용.

    기본값 예시:
      RAW:     s3://datapipeline-raw-data-keonho/daily/user_events_{{ ds_nodash }}.csv
      FEATURE: s3://datapipeline-feature-data-keonho/daily/user_events_feat_{{ ds_nodash }}.csv
    """
    try:
        raw_path = Variable.get(
            "dp_raw_path",
            default_var="s3://datapipeline-raw-data-keonho/daily/user_events_{{ ds_nodash }}.csv",
        )
        feature_path = Variable.get(
            "dp_feature_path",
            default_var="s3://datapipeline-feature-data-keonho/daily/user_events_feat_{{ ds_nodash }}.csv",
        )
        pipeline_name = Variable.get(
            "dp_pipeline_name",
            default_var="daily_user_events",
        )
    except Exception:
        # Variable에서 뭔가 문제가 생겨도 DAG는 돌아가도록 기본값 사용
        raw_path = "s3://datapipeline-raw-data-keonho/daily/user_events_{{ ds_nodash }}.csv"
        feature_path = "s3://datapipeline-feature-data-keonho/daily/user_events_feat_{{ ds_nodash }}.csv"
        pipeline_name = "daily_user_events"

    return {
        "raw_path": raw_path,
        "feature_path": feature_path,
        "pipeline_name": pipeline_name,
    }


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
        feature_path=cfg["feature_path"],
        pipeline_name=cfg["pipeline_name"],
        ti=context["ti"],
    )


def task_store_features(**context):
    cfg = _get_pipeline_config()
    store_features(
        feature_path=cfg["feature_path"],
        pipeline_name=cfg["pipeline_name"],
        ti=context["ti"],
    )


def task_summarize_run(**context):
    cfg = _get_pipeline_config()
    summarize_run(
        pipeline_name=cfg["pipeline_name"],
        ti=context["ti"],
    )


with DAG(
    dag_id="data_pipeline_daily_dev",
    default_args=default_args,
    schedule="0 3 * * *",  # 매일 새벽 3시
    catchup=False,
    tags=["data-pipeline", "dev", "mlops"],
    description="데일리 데이터 파이프라인 (raw → validate → feature → store)",
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
        trigger_rule=TriggerRule.ALL_DONE,  # 중간에 실패해도 요약은 시도
    )

    t_extract >> t_validate >> t_build >> t_store >> t_summary
