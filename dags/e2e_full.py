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
    # Task IDs (SSOT)
    DP_EXTRACT_TASK_ID,
    DP_VALIDATE_TASK_ID,
    DP_BUILD_TASK_ID,
    DP_STORE_TASK_ID,
    SUMMARIZE_TASK_ID,
    TRAIN_TASK_ID,
    BRANCH_TASK_ID,
    REGISTER_TASK_ID,
    PROMOTION_START_TASK_ID,
    SHADOW_START_TASK_ID,
    NOTIFY_FAILURE_TASK_ID,
    SENSOR_MODEL_READY_TASK_ID,
    COMMIT_CURRENT_TASK_ID,
    FASTAPI_RELOAD_TASK_ID,
    ROLLBACK_MINIMAL_TASK_ID,
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
    """
    NOTE (운영/제출 기준 SSOT):
    - TaskGroup 내부에서 operator를 만들 때는 짧은 task_id(예: "store_features")로 생성합니다.
    - 하지만 실제 DAG의 full task_id는 "{group_id}.{task_id}" (예: "dp.store_features") 입니다.
    - XCom pull / task_ids 참조는 반드시 ids.py의 "full task_id" SSOT 값을 사용합니다.
    """

    def mk_py(task_id: str, fn, *, trigger_rule=TriggerRule.ALL_SUCCESS) -> PythonOperator:
        return PythonOperator(task_id=task_id, python_callable=fn, trigger_rule=trigger_rule)

    # 1) Data pipeline
    with TaskGroup(group_id=TG_DP) as dp:
        # ids.py는 full task_id("dp.extract_raw_data")를 제공하지만,
        # operator 생성은 TG 내부이므로 task_id는 suffix만 사용합니다.
        extract_raw_data = mk_py("extract_raw_data", p.dp_extract)
        validate_data = mk_py("validate_data", p.dp_validate)
        build_features = mk_py("build_features", p.dp_build)
        store_features = mk_py("store_features", p.dp_store)

        extract_raw_data >> validate_data >> build_features >> store_features

    summarize_run = mk_py(SUMMARIZE_TASK_ID, p.dp_summary, trigger_rule=TriggerRule.ALL_DONE)

    # 2) Train / Branch
    train = mk_py(TRAIN_TASK_ID, p.train_and_evaluate)

    branch = BranchPythonOperator(
        task_id=BRANCH_TASK_ID,
        python_callable=p.branch_by_accuracy,  # -> REGISTER_TASK_ID OR SHADOW_START_TASK_ID
    )

    promotion_start = EmptyOperator(task_id=PROMOTION_START_TASK_ID)
    shadow_start = EmptyOperator(task_id=SHADOW_START_TASK_ID)

    # 3) Promotion path
    register = mk_py(REGISTER_TASK_ID, p.register_model_task)

    check_model_ready = PythonSensor(
        task_id=SENSOR_MODEL_READY_TASK_ID,
        python_callable=p.sensor_ready_func,
        poke_interval=MODEL_READY_POKE_INTERVAL_SEC,
        timeout=MODEL_READY_TIMEOUT_SEC,
        mode=MODEL_READY_MODE,
    )

    # shadow path reason notify (train skipped / below threshold / invalid accuracy)
    notify_failure = mk_py(
        NOTIFY_FAILURE_TASK_ID,
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

    # 5) Promotion-only
    commit_current = mk_py(COMMIT_CURRENT_TASK_ID, p.commit_current)

    # 정책: FastAPI reload 실패는 자동 롤백하지 않음 (repo 되돌림은 위험)
    fastapi_reload = mk_py(FASTAPI_RELOAD_TASK_ID, p.fastapi_reload_task)

    # deploy/commit 실패 시 최소 롤백
    rollback_minimal = mk_py(
        ROLLBACK_MINIMAL_TASK_ID,
        p.triton_rollback_task,
        trigger_rule=TriggerRule.ONE_FAILED,
    )

    # -----------------------
    # Dependencies
    # -----------------------
    dp >> train >> branch

    # branch targets
    branch >> register >> promotion_start
    branch >> shadow_start >> notify_failure

    # promotion path must wait for model ready
    promotion_start >> check_model_ready >> deploy

    # shadow path goes directly to deploy (shadow materialize)
    shadow_start >> deploy

    # commit + reload only after deploy
    deploy >> commit_current >> fastapi_reload

    # rollback policy
    [deploy, commit_current] >> rollback_minimal

    # summarize (always)
    store_features >> summarize_run
    fastapi_reload >> summarize_run
    rollback_minimal >> summarize_run
    notify_failure >> summarize_run
