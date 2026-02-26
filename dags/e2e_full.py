# dags/e2e_full.py
from __future__ import annotations

from datetime import datetime, timedelta

from pendulum import timezone

from airflow import DAG
from airflow.operators.empty import EmptyOperator
from airflow.operators.python import BranchPythonOperator, PythonOperator
from airflow.sensors.python import PythonSensor
from airflow.utils.task_group import TaskGroup
from airflow.utils.trigger_rule import TriggerRule

from pipelines import full_e2e as p
from utils.slack_alerts import alert_slack

kst = timezone("Asia/Seoul")

DEFAULT_ARGS = {
    "start_date": datetime(2025, 1, 1, tzinfo=kst),
    "retries": 1,
    "retry_delay": timedelta(minutes=2),
}

DAG_ID = "e2e_full"

with DAG(
    dag_id=DAG_ID,
    default_args=DEFAULT_ARGS,
    schedule=None,
    catchup=False,
    max_active_runs=1,
    tags=["e2e", "mlops", "triton", "mlflow"],
    on_failure_callback=alert_slack,
    dagrun_timeout=timedelta(minutes=30),
) as dag:

    def mk_py(task_id: str, fn, *, trigger_rule: str = TriggerRule.ALL_SUCCESS) -> PythonOperator:
        return PythonOperator(task_id=task_id, python_callable=fn, trigger_rule=trigger_rule)

    # 1) Data pipeline
    with TaskGroup(group_id="dp") as dp:
        extract_raw_data = mk_py("extract_raw_data", p.dp_extract)
        validate_data = mk_py("validate_data", p.dp_validate)
        build_features = mk_py("build_features", p.dp_build)
        store_features = mk_py("store_features", p.dp_store)

        extract_raw_data >> validate_data >> build_features >> store_features

    summarize_run = mk_py("summarize_run", p.dp_summary, trigger_rule=TriggerRule.ALL_DONE)

    # 2) Train / Branch
    train = mk_py("train_and_evaluate", p.train_and_evaluate)

    branch = BranchPythonOperator(
        task_id="check_result",
        python_callable=p.branch_by_accuracy,  # -> register_model_task OR shadow_start
    )

    promotion_start = EmptyOperator(task_id="promotion_start")
    shadow_start = EmptyOperator(task_id="shadow_start")

    # 3) Promotion path
    register = mk_py("register_model_task", p.register_model_task)

    check_model_ready = PythonSensor(
        task_id="check_model_ready",
        python_callable=p.sensor_ready_func,
        poke_interval=10,
        timeout=180,
        mode="reschedule",
    )

    # train skipped / below threshold 알림 (shadow_start 경로)
    notify_failure = mk_py(
        "notify_failure",
        p.notify_shadow_reason,
        trigger_rule=TriggerRule.NONE_FAILED_MIN_ONE_SUCCESS,
    )

    # 4) Deploy chain
    with TaskGroup(group_id="deploy") as deploy:
        snapshot_current = mk_py(
            "snapshot_current",
            p.snapshot_current,
            trigger_rule=TriggerRule.NONE_FAILED_MIN_ONE_SUCCESS,
        )
        materialize_repo = mk_py(
            "materialize_repo",
            p.triton_materialize_task,
            trigger_rule=TriggerRule.NONE_FAILED_MIN_ONE_SUCCESS,
        )

        triton_load = mk_py("triton_load", p.triton_load_task)
        triton_ready = mk_py("triton_ready", p.triton_ready_task)
        triton_infer_smoke = mk_py("triton_infer_smoke", p.triton_infer_smoke_task)

        snapshot_current >> materialize_repo >> triton_load >> triton_ready >> triton_infer_smoke

    # deploy/commit 실패 시 최소 롤백
    rollback_minimal = mk_py("rollback_minimal", p.triton_rollback_task, trigger_rule=TriggerRule.ONE_FAILED)

    # 5) Promotion-only
    commit_current = mk_py("commit_current", p.commit_current)

    # ✅ 정책: FastAPI reload 실패는 자동 롤백하지 않음(모델 repo 되돌림은 위험)
    fastapi_reload = mk_py("fastapi_reload", p.fastapi_reload_task)

    # Dependencies
    dp >> train >> branch

    branch >> register >> promotion_start
    branch >> shadow_start >> notify_failure

    promotion_start >> check_model_ready >> deploy
    shadow_start >> deploy

    deploy >> commit_current >> fastapi_reload

    # rollback policy
    [deploy, commit_current] >> rollback_minimal

    # summarize (always)
    store_features >> summarize_run
    fastapi_reload >> summarize_run
    rollback_minimal >> summarize_run
    notify_failure >> summarize_run
