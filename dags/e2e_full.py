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

from mlops_lib.core.ids import (
    DAG_ID_E2E_FULL,
    TG_DP,
    TG_DEPLOY,
)

from mlops_lib.core.policy import (
    E2E_START_DATE_YMD,
    E2E_RETRIES,
    E2E_RETRY_DELAY_MIN,
    E2E_MAX_ACTIVE_RUNS,
    E2E_DAGRUN_TIMEOUT_MIN,
    MODEL_READY_POKE_INTERVAL_SEC,
    MODEL_READY_TIMEOUT_SEC,
    MODEL_READY_MODE,
)

kst = timezone("Asia/Seoul")

DEFAULT_ARGS = {
    "start_date": datetime(*E2E_START_DATE_YMD, tzinfo=kst),
    "retries": E2E_RETRIES,
    "retry_delay": timedelta(minutes=E2E_RETRY_DELAY_MIN),
}

with DAG(
    dag_id=DAG_ID_E2E_FULL,
    default_args=DEFAULT_ARGS,
    schedule=None,
    catchup=False,
    max_active_runs=E2E_MAX_ACTIVE_RUNS,
    tags=["e2e", "mlops", "triton", "mlflow"],
    on_failure_callback=alert_slack,
    dagrun_timeout=timedelta(minutes=E2E_DAGRUN_TIMEOUT_MIN),
) as dag:

    def mk_py(task_id: str, fn, *, trigger_rule: str = TriggerRule.ALL_SUCCESS) -> PythonOperator:
        return PythonOperator(task_id=task_id, python_callable=fn, trigger_rule=trigger_rule)

    # 1) Data pipeline
    with TaskGroup(group_id=TG_DP) as dp:
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
        poke_interval=MODEL_READY_POKE_INTERVAL_SEC,
        timeout=MODEL_READY_TIMEOUT_SEC,
        mode=MODEL_READY_MODE,
    )

    # train skipped / below threshold 알림 (shadow_start 경로)
    notify_failure = mk_py(
        "notify_failure",
        p.notify_shadow_reason,
        trigger_rule=TriggerRule.NONE_FAILED_MIN_ONE_SUCCESS,
    )

    # 4) Deploy chain
    with TaskGroup(group_id=TG_DEPLOY) as deploy:
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

    # deploy/commit/observe 실패 시 최소 롤백
    rollback_minimal = mk_py("rollback_minimal", p.triton_rollback_task, trigger_rule=TriggerRule.ONE_FAILED)

    # ✅ FastAPI reload 실패는 자동 롤백하지 않음(정책 유지)
    fastapi_reload = mk_py("fastapi_reload", p.fastapi_reload_task)

    # ✅ 배포 후 관측(임계치 초과 시 실패 -> rollback 트리거)
    observe_metrics = mk_py("observe_metrics", p.observe_post_deploy_metrics)

    # 5) commit (promotion-only 의미지만, 파이프라인 SSOT 관점에서 공통 체인으로 둠)
    commit_current = mk_py("commit_current", p.commit_current)

    # Dependencies
    dp >> train >> branch

    branch >> register >> promotion_start
    branch >> shadow_start >> notify_failure

    promotion_start >> check_model_ready >> deploy
    shadow_start >> deploy

    # ✅ deploy 후: reload -> observe -> commit
    deploy >> fastapi_reload >> observe_metrics >> commit_current

    # rollback policy (fastapi_reload는 제외)
    [deploy, observe_metrics, commit_current] >> rollback_minimal

    # summarize (always)
    store_features >> summarize_run
    fastapi_reload >> summarize_run
    observe_metrics >> summarize_run
    commit_current >> summarize_run
    rollback_minimal >> summarize_run
    notify_failure >> summarize_run
