# dags/dag_data_pipeline_daily.py

from datetime import datetime, timedelta
from pendulum import timezone

from airflow import DAG
from airflow.operators.python import PythonOperator
from airflow.utils.trigger_rule import TriggerRule

from airflow.sdk import Variable

from utils.slack_alerts import alert_slack
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


def _get_pipeline_config(context):
    """
    실행 시점(context)을 받아서 S3 경로를 동적으로 생성합니다.
    👉 여기서는 템플릿 문자열({{ ds_nodash }})을 쓰지 않습니다.

    - RAW 버킷:    datapipeline-raw-data-keonho
    - FEATURE 버킷: datapipeline-feature-data-keonho

    파일명 규칙:
      user_events_<ds_nodash>.csv
      user_events_feat_<ds_nodash>.csv
    """
    ds_nodash = context["ds_nodash"]  # 예: 20251119

    # Variable에서 버킷/프리픽스를 바꿀 수 있게 해둠 (없으면 기본값 사용)
    raw_bucket = Variable.get("dp_raw_bucket", default_var="datapipeline-raw-data-keonho")
    raw_prefix = Variable.get("dp_raw_prefix", default_var="")  # 예: "daily"
    feat_bucket = Variable.get("dp_feature_bucket", default_var="datapipeline-feature-data-keonho")
    feat_prefix = Variable.get("dp_feature_prefix", default_var="daily")

    def build_s3_uri(bucket, prefix, filename):
        prefix = prefix.strip("/")
        if prefix:
            key = f"{prefix}/{filename}"
        else:
            key = filename
        return f"s3://{bucket}/{key}"

    raw_filename = f"user_events_{ds_nodash}.csv"
    feat_filename = f"user_events_feat_{ds_nodash}.csv"

    raw_path = build_s3_uri(raw_bucket, raw_prefix, raw_filename)
    feature_path = build_s3_uri(feat_bucket, feat_prefix, feat_filename)

    pipeline_name = Variable.get("dp_pipeline_name", default_var="daily_user_events")

    return {
        "raw_path": raw_path,
        "feature_path": feature_path,
        "pipeline_name": pipeline_name,
    }


def task_extract_raw_data(**context):
    cfg = _get_pipeline_config(context)
    extract_raw_data(
        raw_path=cfg["raw_path"],
        pipeline_name=cfg["pipeline_name"],
        ti=context["ti"],
    )


def task_validate_data(**context):
    cfg = _get_pipeline_config(context)
    validate_data(
        raw_path=cfg["raw_path"],
        pipeline_name=cfg["pipeline_name"],
        ti=context["ti"],
    )


def task_build_features(**context):
    cfg = _get_pipeline_config(context)
    build_features(
        raw_path=cfg["raw_path"],
        feature_path=cfg["feature_path"],
        pipeline_name=cfg["pipeline_name"],
        ti=context["ti"],
    )


def task_store_features(**context):
    cfg = _get_pipeline_config(context)
    store_features(
        feature_path=cfg["feature_path"],
        pipeline_name=cfg["pipeline_name"],
        ti=context["ti"],
    )


def task_summarize_run(**context):
    cfg = _get_pipeline_config(context)
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
        trigger_rule=TriggerRule.ALL_DONE,
    )

    t_extract >> t_validate >> t_build >> t_store >> t_summary
