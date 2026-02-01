from __future__ import annotations

from datetime import datetime, timedelta
from pendulum import timezone

from airflow import DAG
from airflow.operators.python import PythonOperator, BranchPythonOperator
from airflow.operators.bash import BashOperator
from airflow.sensors.python import PythonSensor
from airflow.operators.empty import EmptyOperator
from airflow.utils.trigger_rule import TriggerRule

from e2e.slack import alert_slack, notify_info, notify_success, notify_fail, notify_skip
from e2e import dp, train, registry, triton, fastapi
from e2e.config import cfg

kst = timezone("Asia/Seoul")

default_args = {
    "start_date": datetime(2025, 1, 1, tzinfo=kst),
    "retries": 1,
    "retry_delay": timedelta(minutes=2),
}

with DAG(
    dag_id="e2e_full",
    default_args=default_args,
    schedule=None,
    catchup=False,
    max_active_runs=1,
    tags=["e2e", "mlops", "feast", "triton", "mlflow"],
    on_failure_callback=alert_slack,
) as dag:
    # -----------------------
    # Data / Feature
    # -----------------------
    t_extract = PythonOperator(task_id="extract_raw", python_callable=dp.extract_raw)
    t_validate = PythonOperator(task_id="validate_raw", python_callable=dp.validate_raw)
    t_build = PythonOperator(task_id="build_features", python_callable=dp.build_features)
    t_store = PythonOperator(task_id="store_features", python_callable=dp.store_features)

    t_feast_apply = BashOperator(
        task_id="feast_apply",
        bash_command=(
            "set -euo pipefail\n"
            f"cd {cfg('FEAST_REPO_PATH', '/opt/airflow/dags/repo/dags/feast_repo')}\n"
            "feast apply\n"
        ),
    )
    t_feast_materialize = BashOperator(
        task_id="feast_materialize",
        bash_command=(
            "set -euo pipefail\n"
            f"cd {cfg('FEAST_REPO_PATH', '/opt/airflow/dags/repo/dags/feast_repo')}\n"
            'feast materialize "{{ macros.ds_add(ds, -1) }}T00:00:00" "{{ ds }}T23:59:59"\n'
        ),
    )

    # -----------------------
    # Train
    # -----------------------
    t_train = PythonOperator(task_id="train", python_callable=train.train_and_eval)

    # -----------------------
    # Branch: promote vs shadow
    # -----------------------
    def _branch(**context):
        """
        return: 'promote_start' or 'shadow_start'
        """
        ti = context["ti"]
        acc = ti.xcom_pull(task_ids="train", key="accuracy") or 0.0
        thr = float(cfg("accuracy_threshold", "0.50"))
        if acc >= thr:
            return "promote_start"
        return "shadow_start"

    t_branch = BranchPythonOperator(task_id="branch_result", python_callable=_branch)

    promote_start = EmptyOperator(task_id="promote_start")
    shadow_start = EmptyOperator(task_id="shadow_start")

    # -----------------------
    # Promotion path
    # -----------------------
    t_register = PythonOperator(task_id="register_model", python_callable=registry.register_alias)
    t_registry_ready = PythonSensor(
        task_id="wait_registry_ready",
        python_callable=registry.sensor_model_ready,
        poke_interval=10,
        timeout=180,
        mode="reschedule",
    )

    # -----------------------
    # Deploy group (common logic but invoked separately)
    # -----------------------
    t_snapshot = PythonOperator(task_id="snapshot_current", python_callable=triton.snapshot_current)

    # (A) promotion deploy
    t_materialize_promote = PythonOperator(
        task_id="materialize_promote",
        python_callable=triton.materialize_promote,  # alias 기반
    )

    # (B) shadow deploy
    t_materialize_shadow = PythonOperator(
        task_id="materialize_shadow",
        python_callable=triton.materialize_shadow,  # run_id 기반
    )

    t_load = PythonOperator(task_id="triton_load", python_callable=triton.triton_load)
    t_ready = PythonOperator(task_id="triton_ready", python_callable=triton.triton_ready)
    t_smoke = PythonOperator(task_id="triton_smoke", python_callable=triton.triton_infer_smoke)

    # ✅ rollback은 “deploy 실패”에서만
    t_rollback = PythonOperator(
        task_id="rollback_deploy",
        python_callable=triton.rollback_minimal,
        trigger_rule=TriggerRule.ONE_FAILED,
    )

    # promotion-only: commit + reload (실패해도 rollback 하지 않음)
    t_commit = PythonOperator(
        task_id="commit_current",
        python_callable=triton.commit_current,
        trigger_rule=TriggerRule.ALL_SUCCESS,
    )
    t_reload = PythonOperator(
        task_id="fastapi_reload",
        python_callable=fastapi.trigger_reload_task,
        trigger_rule=TriggerRule.ALL_SUCCESS,
    )

    # notify
    t_notify_shadow = PythonOperator(
        task_id="notify_shadow",
        python_callable=lambda **_: notify_skip("Accuracy below threshold", next_action="tune features/label/model"),
        trigger_rule=TriggerRule.ALL_DONE,
    )

    t_summary = PythonOperator(
        task_id="summarize",
        python_callable=dp.summarize_run,
        trigger_rule=TriggerRule.ALL_DONE,
    )

    # -----------------------
    # Dependencies
    # -----------------------
    t_extract >> t_validate >> t_build >> t_store >> t_feast_apply >> t_feast_materialize
    t_feast_materialize >> t_train >> t_branch

    # branch
    t_branch >> promote_start >> t_register >> t_registry_ready >> t_snapshot >> t_materialize_promote
    t_branch >> shadow_start >> t_notify_shadow >> t_snapshot >> t_materialize_shadow

    # deploy chain (materialize_* -> load/ready/smoke)
    [t_materialize_promote, t_materialize_shadow] >> t_load >> t_ready >> t_smoke

    # rollback only if deploy parts fail
    [t_materialize_promote, t_materialize_shadow, t_load, t_ready, t_smoke] >> t_rollback

    # promotion-only state update
    t_smoke >> t_commit >> t_reload

    # summary
    [t_feast_materialize, t_reload, t_rollback, t_notify_shadow] >> t_summary

