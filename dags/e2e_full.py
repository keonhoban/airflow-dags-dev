# dags/e2e_full.py
"""
E2E ML 플랫폼 파이프라인 (e2e_full)

개요
----
데이터 추출 → feature 빌드 → 학습 → 배포 → 모니터링 → (필요 시) 자동 롤백까지
전 과정을 단일 DAG로 오케스트레이션한다.
비즈니스 로직은 pipelines/, ml_code/, mlops_lib/ 에 위임하고,
이 파일은 태스크 의존성(DAG 구조)만 정의한다.

전체 플로우 요약
-----------------
[dp: 데이터 파이프라인]
  extract_raw_data → validate_data → build_features → store_features
    ↓
[drift_gate: 배포 전 데이터 드리프트 검사 (KS-stat)]
    ↓
[train: 모델 학습 + 정확도 평가]
    ↓
[branch: 정확도 & 드리프트 결과로 경로 분기]
    ├─ promotion 경로 (정확도 ≥ promote_threshold, drift 없음)
    │     register → check_model_ready → deploy → commit_current → fastapi_reload
    ├─ canary 경로 (canary_threshold ≤ 정확도 < promote_threshold, drift 없음)
    │     promotion과 동일 경로, XCOM_CANARY_TRAFFIC_PCT로 구분
    └─ shadow 경로 (정확도 < canary_threshold or drift 감지 or train 실패)
          shadow_start → notify_failure → deploy (commit/reload 생략)
    ↓
[observe_post_deploy_metrics: Prometheus 기반 자동 롤백 판정]
    ↓ (deploy/commit/observe 실패 시)
[rollback_minimal: current.json 복원 + 실패 버전 격리 + Triton 재로드]

트리거 방식 / schedule
-----------------------
schedule=None — 수동 트리거 전용.

이유: 이 DAG의 upstream인 dp_feature_pipeline 완료 시점이
     데이터 볼륨에 따라 가변적이므로 cron을 고정하면 race condition이 발생한다.
     운영 환경에서는 dp_feature_pipeline의 on_success_callback 또는
     Airflow Dataset 트리거로 자동 연결한다 (mlops-infra-gitops 참고).

SLA
----
default_args['sla'] = timedelta(hours=E2E_SLA_HOUR).
트리거 기준으로 SLA_HOUR(기본 1시간) 내에 완료되지 않으면
sla_miss_callback(alert_sla_miss)이 Slack으로 알림을 보낸다.
dagrun_timeout(30분)은 강제 종료, SLA는 경보용으로 역할이 다르다.

관련 문서
----------
- docs/02_deployment_flow.md : promotion / shadow / rollback 경로 상세
- docs/04_rollback_policy.md : 롤백 범위와 FastAPI 정책
- docs/06_runbook.md         : 장애 유형별 운영 절차
"""
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
    # TODO(W6): Airflow 2.10+ Dataset API로 dp_feature_pipeline → e2e_full 자동 트리거 구현.
    #   현재 미구현 사유:
    #   - 데모/포트폴리오 환경에서 수동 트리거가 디버깅에 유리
    #   - Dataset 트리거는 dp_feature_pipeline에 outlet 정의 필요 (양 DAG 동시 변경)
    #   - 운영 적용 시: dp_feature_pipeline에서 Dataset("s3://features/latest") outlet 추가,
    #     이 DAG의 schedule을 [Dataset("s3://features/latest")]로 변경
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
        python_callable=p.branch_by_accuracy,  # -> I.REGISTER (promotion/canary) OR I.SHADOW_START
    )

    promotion_start = EmptyOperator(task_id=I.PROMOTION_START)
    shadow_start = EmptyOperator(task_id=I.SHADOW_START)

    # =========================================================
    # 3) Promotion / Canary path
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
    # - promotion/canary: 실행
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
    # - promotion/canary: deploy -> commit -> reload -> observe
    # - shadow: deploy -> commit -> (reload skipped) -> observe
    deploy >> commit_current >> fastapi_reload >> observe_metrics

    # rollback policy (reload는 제외)
    [deploy, commit_current, observe_metrics] >> rollback_minimal

    # summarize fan-in: 실제 terminal 태스크만 연결한다.
    # - fastapi_reload  : promotion/canary의 마지막 I/O 태스크 (shadow는 skip됨)
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
