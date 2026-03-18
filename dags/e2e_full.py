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
from utils.slack_alerts import alert_slack, alert_sla_miss

# ✅ Import 압축: DAG는 E2E alias만 봄
from mlops_lib.core.ids import E2E as I

from mlops_lib.core.policy import (
    E2E_START_DATE_YMD,
    E2E_RETRIES,
    E2E_RETRY_DELAY_MIN,
    E2E_MAX_ACTIVE_RUNS,
    E2E_DAGRUN_TIMEOUT_MIN,
    E2E_SLA_HOUR,
    MODEL_READY_POKE_INTERVAL_SEC,
    MODEL_READY_TIMEOUT_SEC,
    MODEL_READY_MODE,
)

kst = timezone("Asia/Seoul")

DEFAULT_ARGS = {
    "start_date": datetime(*E2E_START_DATE_YMD, tzinfo=kst),
    "retries": E2E_RETRIES,
    "retry_delay": timedelta(minutes=E2E_RETRY_DELAY_MIN),
    # SLA: 태스크 단위. 트리거 기준으로 E2E_SLA_HOUR 내에 완료되지 않으면
    # DAG 레벨 sla_miss_callback(alert_sla_miss)이 호출된다.
    "sla": timedelta(hours=E2E_SLA_HOUR),
}


def mk_py(task_id: str, fn, *, trigger_rule: str = TriggerRule.ALL_SUCCESS) -> PythonOperator:
    return PythonOperator(
        task_id=task_id,
        python_callable=fn,
        trigger_rule=trigger_rule,
    )


with DAG(
    dag_id=I.DAG_ID,
    default_args=DEFAULT_ARGS,
    # schedule=None: 수동 트리거 전용 DAG.
    # 이유: dp_feature_pipeline(upstream)의 완료 시점이 데이터 볼륨에 따라 가변적이므로
    #       cron으로 고정하면 race condition이 발생할 수 있다.
    #       운영 환경에서는 dp_feature_pipeline의 on_success_callback 또는
    #       Airflow Dataset 트리거로 연결한다(mlops-infra-gitops 참고).
    schedule=None,
    catchup=False,
    max_active_runs=E2E_MAX_ACTIVE_RUNS,
    tags=["e2e", "mlops", "triton", "mlflow"],
    on_failure_callback=alert_slack,
    sla_miss_callback=alert_sla_miss,
    dagrun_timeout=timedelta(minutes=E2E_DAGRUN_TIMEOUT_MIN),
) as dag:

    # =========================================================
    # 1) Data Pipeline
    # =========================================================
    with TaskGroup(group_id=I.TG_DP) as dp:
        extract_raw_data = mk_py(I.DP_EXTRACT_S, p.dp_extract)
        validate_data = mk_py(I.DP_VALIDATE_S, p.dp_validate)
        build_features = mk_py(I.DP_BUILD_S, p.dp_build)
        store_features = mk_py(I.DP_STORE_S, p.dp_store)

        extract_raw_data >> validate_data >> build_features >> store_features

    summarize_run = mk_py(I.SUMMARIZE, p.dp_summary, trigger_rule=TriggerRule.ALL_DONE)

    # =========================================================
    # 1.5) Drift Gate
    # =========================================================
    drift_gate = mk_py(I.DRIFT_GATE, p.drift_gate_task)

    # =========================================================
    # 2) Train / Branch
    # =========================================================
    train = mk_py(I.TRAIN, p.train_and_evaluate)

    branch = BranchPythonOperator(
        task_id=I.BRANCH,
        python_callable=p.branch_by_accuracy,  # -> I.REGISTER OR I.SHADOW_START
    )

    promotion_start = EmptyOperator(task_id=I.PROMOTION_START)
    shadow_start = EmptyOperator(task_id=I.SHADOW_START)

    # =========================================================
    # 3) Promotion path
    # =========================================================
    register = mk_py(I.REGISTER, p.register_model_task)

    check_model_ready = PythonSensor(
        task_id=I.SENSOR_MODEL_READY,
        python_callable=p.sensor_ready_func,
        poke_interval=MODEL_READY_POKE_INTERVAL_SEC,
        timeout=MODEL_READY_TIMEOUT_SEC,
        mode=MODEL_READY_MODE,
    )

    notify_failure = mk_py(
        I.NOTIFY_FAILURE,
        p.notify_shadow_reason,
        trigger_rule=TriggerRule.NONE_FAILED_MIN_ONE_SUCCESS,
    )

    # =========================================================
    # 4) Deploy chain
    # =========================================================
    with TaskGroup(group_id=I.TG_DEPLOY) as deploy:
        snapshot_current = mk_py(
            I.DEPLOY_SNAPSHOT_S,
            p.snapshot_current,
            trigger_rule=TriggerRule.NONE_FAILED_MIN_ONE_SUCCESS,
        )
        materialize_repo = mk_py(
            I.DEPLOY_MATERIALIZE_S,
            p.triton_materialize_task,
            trigger_rule=TriggerRule.NONE_FAILED_MIN_ONE_SUCCESS,
        )

        triton_load = mk_py(I.DEPLOY_TRITON_LOAD_S, p.triton_load_task)
        triton_ready = mk_py(I.DEPLOY_TRITON_READY_S, p.triton_ready_task)
        triton_infer_smoke = mk_py(I.DEPLOY_TRITON_SMOKE_S, p.triton_infer_smoke_task)

        snapshot_current >> materialize_repo >> triton_load >> triton_ready >> triton_infer_smoke

    # =========================================================
    # 5) Commit
    # =========================================================
    commit_current = mk_py(I.COMMIT, p.commit_current)

    # =========================================================
    # 6) FastAPI reload
    # 정책:
    # - promotion: 실행
    # - shadow: p_reload.py 내부 정책으로 skip
    # =========================================================
    fastapi_reload = mk_py(I.FASTAPI_RELOAD, p.fastapi_reload_task)

    # =========================================================
    # 7) Post-deploy Observability
    # shadow에서는 fastapi_reload가 skipped 되어도 observe는 계속 진행
    # =========================================================
    observe_metrics = mk_py(
        I.OBSERVE,
        p.observe_post_deploy_metrics,
        trigger_rule=TriggerRule.NONE_FAILED_MIN_ONE_SUCCESS,
    )

    # =========================================================
    # 8) Rollback (deploy/commit/observe 실패 시)
    # =========================================================
    rollback_minimal = mk_py(
        I.ROLLBACK_MINIMAL,
        p.triton_rollback_task,
        trigger_rule=TriggerRule.ONE_FAILED,
    )

    # =========================================================
    # Dependencies
    # =========================================================
    dp >> drift_gate >> train >> branch

    branch >> register >> promotion_start
    branch >> shadow_start >> notify_failure

    promotion_start >> check_model_ready >> deploy
    shadow_start >> deploy

    # 운영형 순서:
    # - promotion: deploy -> commit -> reload -> observe
    # - shadow: deploy -> commit -> (reload skipped) -> observe
    deploy >> commit_current >> fastapi_reload >> observe_metrics

    # rollback policy (reload는 제외)
    [deploy, commit_current, observe_metrics] >> rollback_minimal

    # summarize fan-in: 실제 terminal 태스크만 연결한다.
    # - fastapi_reload  : promotion의 마지막 I/O 태스크 (shadow는 skip됨)
    # - observe_metrics : 파이프라인 최종 품질 판정
    # - rollback_minimal: 장애 복구 경로의 끝
    # - notify_failure  : shadow 분기 알림의 끝
    # (store_features는 초반 dp 태스크이므로 제외)
    [
        fastapi_reload,
        observe_metrics,
        rollback_minimal,
        notify_failure,
    ] >> summarize_run
