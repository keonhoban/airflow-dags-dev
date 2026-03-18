# dags/utils/slack_alerts.py
from __future__ import annotations

import logging
import os
from dataclasses import dataclass
from typing import Any, Dict, Optional

import requests

log = logging.getLogger(__name__)

_MAX_VALUE_LEN = 180
_DEFAULT_TIMEOUT_SEC = 5


# -----------------------
# small utils (stable)
# -----------------------
def _get_env(name: str, default: str = "") -> str:
    v = os.getenv(name, default)
    return v.strip() if isinstance(v, str) else str(v).strip()


def _truncate(v: Any, max_len: int = _MAX_VALUE_LEN) -> str:
    if v is None or v == "":
        return "-"
    s = str(v)
    return (s[:max_len] + "…") if len(s) > max_len else s


def slack_kv(**fields: Any) -> str:
    """Slack markdown용 key-value 포맷(폭발 방지)."""
    return "\n".join([f"- *{k}*: `{_truncate(v)}`" for k, v in fields.items()])


# -----------------------
# notifier
# -----------------------
@dataclass(frozen=True)
class SlackNotifier:
    webhook_url: Optional[str]
    timeout_sec: int = _DEFAULT_TIMEOUT_SEC

    @classmethod
    def from_env(cls) -> "SlackNotifier":
        url = _get_env("SLACK_WEBHOOK_URL")
        return cls(webhook_url=url or None)

    def post(self, text: str) -> bool:
        """
        - Slack 미설정이면 False 반환 (DAG 동작 영향 없음)
        - 실패 시에도 예외를 밖으로 던지지 않음 (운영 안정성)
        """
        if not self.webhook_url:
            return False

        try:
            r = requests.post(self.webhook_url, json={"text": text}, timeout=self.timeout_sec)
            if r.status_code >= 400:
                log.warning("slack webhook failed: status=%s body=%s", r.status_code, (r.text or "")[:200])
                return False
            return True
        except requests.RequestException as e:
            log.warning("slack webhook exception: %s", str(e))
            return False

    def notify(self, level: str, title: str, **fields: Any) -> bool:
        header = f"*{level}* - *{title}*"
        body = slack_kv(**fields) if fields else ""
        msg = f"{header}\n{body}" if body else header
        return self.post(msg)


# lazy singleton (import-time 고정 방지)
_NOTIFIER: Optional[SlackNotifier] = None


def _notifier() -> SlackNotifier:
    global _NOTIFIER
    if _NOTIFIER is None:
        _NOTIFIER = SlackNotifier.from_env()
    return _NOTIFIER


def reset_notifier() -> None:
    """
    (선택) 테스트/로컬에서 env 바뀐 경우 notifier 재생성.
    운영에선 보통 호출할 일 없음.
    """
    global _NOTIFIER
    _NOTIFIER = None


# -----------------------
# public helpers (호환 유지)
# -----------------------
def notify(level: str, title: str, **fields: Any) -> bool:
    """level 예: ℹ️ INFO | ✅ SUCCESS | ⏭️ SKIP | 🔥 FAIL"""
    return _notifier().notify(level, title, **fields)


def notify_info(title: str, **fields: Any) -> bool:
    return notify("ℹ️ INFO", title, **fields)


def notify_success(title: str, **fields: Any) -> bool:
    return notify("✅ SUCCESS", title, **fields)


def notify_skip(title: str, **fields: Any) -> bool:
    return notify("⏭️ SKIP", title, **fields)


def notify_fail(title: str, **fields: Any) -> bool:
    return notify("🔥 FAIL", title, **fields)


# -----------------------
# airflow callback (SSOT)
# -----------------------
def alert_sla_miss(dag, task_list, blocking_task_list, slas, blocking_tis) -> None:
    """
    Airflow sla_miss_callback 용.

    SLA를 초과한 태스크가 발생하면 Slack에 경고를 보낸다.
    콜백 실패가 DAG 실행에 영향을 주지 않도록 예외를 억제한다.

    파라미터 (Airflow 2.x 고정 시그니처):
        dag              : DAG 객체
        task_list        : SLA를 초과한 task_id 목록 문자열
        blocking_task_list: 완료를 막고 있는 task_id 목록 문자열
        slas             : SlaMiss 객체 목록
        blocking_tis     : 차단 중인 TaskInstance 목록
    """
    try:
        dag_id = getattr(dag, "dag_id", "-")
        base_url = _get_env("AIRFLOW__WEBSERVER__WEB_SERVER_BASE_URL").rstrip("/")
        dag_url = f"{base_url}/dags/{dag_id}/grid" if base_url else "-"

        notify(
            "⏰ SLA MISS",
            "E2E pipeline exceeded SLA",
            dag=dag_id,
            missed_tasks=_truncate(task_list),
            blocking_tasks=_truncate(blocking_task_list),
            dag_view=dag_url,
        )
    except Exception as exc:  # noqa: BLE001
        log.warning("alert_sla_miss: failed to send Slack notification: %s", exc)


def alert_slack(context: Dict[str, Any]) -> None:
    """
    Airflow on_failure_callback 용.
    - context 구조가 달라도 안전하게 동작
    - 운영 대응/면접 설명력: DAG Run / Task 링크 포함
    """
    dag = context.get("dag")
    ti = context.get("task_instance")

    dag_id = getattr(dag, "dag_id", "-")
    task_id = getattr(ti, "task_id", "-")
    run_id = context.get("run_id", "-")
    ts = context.get("ts", "-")
    try_number = getattr(ti, "try_number", "-")

    base_url = _get_env("AIRFLOW__WEBSERVER__WEB_SERVER_BASE_URL").rstrip("/")
    dag_run_url = f"{base_url}/dags/{dag_id}/grid?dag_run_id={run_id}" if base_url else "-"
    task_url = f"{base_url}/dags/{dag_id}/runs/{run_id}/tasks/{task_id}" if base_url else "-"

    notify_fail(
        "Airflow task failed",
        dag=dag_id,
        task=task_id,
        run_id=run_id,
        ts=ts,
        try_number=str(try_number),
        dag_run=dag_run_url,
        task_view=task_url,
    )
