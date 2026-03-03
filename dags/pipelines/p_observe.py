# dags/pipelines/p_observe.py
from __future__ import annotations

from typing import Any
from airflow.exceptions import AirflowException
from airflow.utils.log.logging_mixin import LoggingMixin

from mlops_lib.core.policy import Settings
from mlops_lib.observability.auto_rollback import AutoRollback

log = LoggingMixin().log


def observe_post_deploy_metrics(**context: Any) -> None:
    """
    배포 후 관측 결과가 나쁘면 task 실패 -> e2e_full.py에서 rollback_minimal 트리거
    - 여기서는 '결정'만 내리고
    - 롤백 실행은 DAG(trigger_rule=ONE_FAILED)에게 맡긴다 (오케스트레이션 SSOT)
    """
    s = Settings.load()

    ar = AutoRollback()
    decision = ar.evaluate()

    log.info(
        "[observe_post_deploy_metrics] env=%s decision=%s signals=%s",
        s.env,
        decision.reason,
        getattr(decision, "signals", None),
    )

    if getattr(decision, "should_rollback", False):
        raise AirflowException(f"[AUTO-ROLLBACK] {decision.reason} | signals={decision.signals}")
