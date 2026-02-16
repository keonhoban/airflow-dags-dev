# dags/utils/slack_alerts.py
import os
import logging
from typing import Any, Dict, Optional

import requests

log = logging.getLogger(__name__)


def _slack_url() -> Optional[str]:
    url = os.getenv("SLACK_WEBHOOK_URL", "").strip()
    return url or None


def _post_slack(text: str, timeout: int = 5) -> bool:
    """
    - Slack 미설정이면 False 반환 (DAG 동작에는 영향 없음)
    - 실패 시에도 예외를 밖으로 던지지 않음 (운영 안정성)
    - 대신 로그로 남겨 디버깅 가능하게 함 (현업 필수)
    """
    url = _slack_url()
    if not url:
        return False

    payload = {"text": text}
    try:
        r = requests.post(url, json=payload, timeout=timeout)
        if r.status_code >= 400:
            log.warning("slack webhook failed: status=%s body=%s", r.status_code, r.text[:200])
            return False
        return True
    except requests.RequestException as e:
        log.warning("slack webhook exception: %s", str(e))
        return False


def slack_kv(**fields: Any) -> str:
    """
    Slack markdown용 key-value 포맷.
    - 값이 길면 잘라서 메시지 폭발 방지 (운영에서 중요)
    """
    lines = []
    for k, v in fields.items():
        if v is None or v == "":
            vv = "-"
        else:
            vv = str(v)
            if len(vv) > 180:
                vv = vv[:180] + "…"
        lines.append(f"- *{k}*: `{vv}`")
    return "\n".join(lines)


def notify(level: str, title: str, **fields: Any) -> bool:
    """
    level: INFO | SUCCESS | SKIP | FAIL
    """
    header = f"*{level}* - *{title}*"
    body = slack_kv(**fields) if fields else ""
    msg = f"{header}\n{body}" if body else header
    return _post_slack(msg)


def notify_info(title: str, **fields: Any) -> bool:
    return notify("ℹ️ INFO", title, **fields)


def notify_success(title: str, **fields: Any) -> bool:
    return notify("✅ SUCCESS", title, **fields)


def notify_skip(title: str, **fields: Any) -> bool:
    return notify("⏭️ SKIP", title, **fields)


def notify_fail(title: str, **fields: Any) -> bool:
    return notify("🔥 FAIL", title, **fields)


def alert_slack(context: Dict[str, Any]) -> None:
    """
    Airflow on_failure_callback 용.
    - context 구조가 조금 달라도 안전하게 동작
    - Web UI 링크를 남겨 '면접에서 운영 대응' 설명 가능
    """
    dag = context.get("dag")
    ti = context.get("task_instance")
    dag_id = getattr(dag, "dag_id", "-")
    task_id = getattr(ti, "task_id", "-")
    run_id = context.get("run_id", "-")
    ts = context.get("ts", "-")

    base_url = os.getenv("AIRFLOW__WEBSERVER__WEB_SERVER_BASE_URL", "").rstrip("/")
    # Airflow UI 링크는 환경마다 다를 수 있어 base_url 없는 경우 "-" 처리
    log_url = f"{base_url}/dags/{dag_id}/runs/{run_id}/tasks/{task_id}" if base_url else "-"

    notify_fail(
        "Airflow task failed",
        dag=dag_id,
        task=task_id,
        run_id=run_id,
        ts=ts,
        log=log_url,
    )

