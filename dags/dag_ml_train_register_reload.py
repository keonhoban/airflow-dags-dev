# dags/dag_ml_train_register_reload.py

from airflow import DAG
from airflow.operators.python import PythonOperator, BranchPythonOperator
from airflow.sensors.python import PythonSensor
from airflow.utils.trigger_rule import TriggerRule
from airflow.exceptions import AirflowSkipException
from airflow.sdk import Variable
from datetime import datetime, timedelta
from pendulum import timezone

from ml_code.train_model import train_model
from ml_code.register_model import register_model
from ml_code.rollback_model import rollback_model
from ml_code.trigger_reload import trigger_reload
from ml_code.sensor_model_ready import check_model_ready
from utils.slack_alerts import send_slack_alert, alert_slack

from mlflow.tracking import MlflowClient

# DAG 설정
kst = timezone("Asia/Seoul")
default_args = {
    'start_date': datetime(2025, 1, 1, tzinfo=kst),
    'retries': 1,
    'retry_delay': timedelta(minutes=2),
}

# ✅ 파라미터 로딩 유틸
def get_param(key, default, cast_func, validate_func=None):
    try:
        value = cast_func(Variable.get(key, default=str(default)))
        if validate_func and not validate_func(value):
            raise ValueError("Validation failed")
        return value
    except Exception as e:
        send_slack_alert(f"[Param] {key} 로딩 실패: {e} → 기본값 {default} 사용")
        return default

# ✅ alias 기반 버전 조회
def get_version_by_alias(model_name, alias):
    try:
        return MlflowClient().get_model_version_by_alias(model_name, alias).version
    except Exception:
        return None

# ✅ 학습 Task
def train_and_evaluate(ti, **_):
    C = get_param("logreg_C", 1.0, float, lambda x: 0.001 <= x <= 10.0)
    max_iter = get_param("logreg_max_iter", 200, int, lambda x: x > 50)
    threshold = get_param("accuracy_threshold", 0.9, float, lambda x: 0.5 <= x <= 0.99)
    model_name = Variable.get("model_name")
    alias = Variable.get("mlflow_alias")

    if not (model_name and alias):
        raise ValueError("필수 Variable 누락: model_name 또는 mlflow_alias")

    acc, run_id = train_model(C=C, max_iter=max_iter)
    if not run_id:
        raise ValueError("run_id 없음 → 학습 실패")

    ti.xcom_push(key="run_id", value=run_id)
    ti.xcom_push(key="model_name", value=model_name)
    ti.xcom_push(key="alias", value=alias)
    ti.xcom_push(key="acc", value=acc)
    ti.xcom_push(key="threshold", value=threshold)

# ✅ 분기 결정
def check_result(ti, **_):
    acc = ti.xcom_pull(task_ids="train_and_evaluate", key="acc")
    threshold = ti.xcom_pull(task_ids="train_and_evaluate", key="threshold")

    if acc is None or threshold is None:
        send_slack_alert("❌ check_result → XCom 누락")
        raise AirflowSkipException()

    return "register_model" if acc >= threshold else "notify_failure"

# ✅ 모델 등록 Task
def register_model_task(ti, **_):
    run_id = ti.xcom_pull(task_ids="train_and_evaluate", key="run_id")
    model_name = ti.xcom_pull(task_ids="train_and_evaluate", key="model_name")
    alias = ti.xcom_pull(task_ids="train_and_evaluate", key="alias")

    prev_version = get_version_by_alias(model_name, alias)

    try:
        version = register_model(run_id, model_name, alias)
        ti.xcom_push(key="version", value=version)
        send_slack_alert(f"✅ 모델 등록 완료: {model_name} v{version} → @{alias}")
    except Exception as e:
        msg = f"❌ 모델 등록 실패: {e}"
        if prev_version:
            rollback_model(model_name, prev_version, alias)
            msg += f" → 롤백 완료: v{prev_version}"
        else:
            msg += " → 롤백 생략"
        send_slack_alert(msg)
        raise

# ✅ 모델 준비 확인
def sensor_ready_func(ti, **_):
    model_name = ti.xcom_pull(task_ids="train_and_evaluate", key="model_name")
    version = ti.xcom_pull(task_ids="register_model", key="version")
    return check_model_ready(model_name, version)

# ✅ 핫스왑 트리거
def trigger_reload_task(ti, **_):
    alias = ti.xcom_pull(task_ids="train_and_evaluate", key="alias")
    try:
        trigger_reload(alias)
        send_slack_alert(f"🔁 핫스왑 완료: @{alias}")
    except Exception as e:
        send_slack_alert(f"❌ 핫스왑 실패: {e}")
        raise

# ✅ 실패 알림
def notify_failure():
    send_slack_alert("⚠️ 기준 미달 → 등록 및 핫스왑 생략")

# ✅ DAG 정의
with DAG(
    dag_id="ml_train_register_and_reload_dev",
    default_args=default_args,
    schedule=None,
    catchup=False,
    tags=["mlops", "train", "sensor"],
    on_failure_callback=alert_slack,
) as dag:

    train = PythonOperator(
        task_id="train_and_evaluate",
        python_callable=train_and_evaluate,
    )

    branch = BranchPythonOperator(
        task_id="check_result",
        python_callable=check_result,
    )

    register = PythonOperator(
        task_id="register_model",
        python_callable=register_model_task,
    )

    sensor = PythonSensor(
        task_id="check_model_ready",
        python_callable=sensor_ready_func,
        poke_interval=10,
        timeout=180,
        mode="reschedule",
    )

    reload = PythonOperator(
        task_id="trigger_reload",
        python_callable=trigger_reload_task,
        trigger_rule=TriggerRule.ALL_SUCCESS,
    )

    failure = PythonOperator(
        task_id="notify_failure",
        python_callable=notify_failure,
        trigger_rule=TriggerRule.NONE_FAILED_MIN_ONE_SUCCESS,
    )

    train >> branch
    branch >> [register, failure]
    register >> sensor >> reload
