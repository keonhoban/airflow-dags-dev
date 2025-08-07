# dags/dag_model_rollback.py

from airflow import DAG
from airflow.operators.python import PythonOperator
from datetime import datetime, timedelta
from pendulum import timezone
from ml_code.rollback_model import rollback_model
from ml_code.trigger_reload import trigger_reload
from utils.slack_alerts import alert_slack
from airflow.sdk import Variable

kst = timezone("Asia/Seoul")

default_args = {
    'start_date': datetime(2024, 1, 1, tzinfo=kst),
    'retries': 1,
    'retry_delay': timedelta(minutes=1),
}

def dag_rollback():
    model_name = Variable.get("rollback_model_name")
    version = Variable.get("rollback_version")
    alias = Variable.get("rollback_alias")
    rollback_model(model_name=model_name, version=version, alias=alias)

def dag_reload():
    alias = Variable.get("rollback_alias")
    trigger_reload(alias)

with DAG(
    dag_id="manual_rollback_model_dev",
    default_args=default_args,
    schedule=None,
    catchup=False,
    tags=["mlops", "rollback"],
    description="MLflow 모델 롤백 DAG (핫스왑 포함)",
    on_failure_callback=alert_slack,
) as dag:

    rollback = PythonOperator(
        task_id="rollback_model_alias",
        python_callable=dag_rollback,
        on_failure_callback=alert_slack,
    )

    reload = PythonOperator(
        task_id="reload_fastapi_model",
        python_callable=dag_reload,
        on_failure_callback=alert_slack,
    )

    rollback >> reload
