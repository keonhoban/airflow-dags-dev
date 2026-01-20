# dags/utils/slack_alerts.py

import os
import requests
from datetime import datetime

def send_slack_alert(text: str):
    """Slack 메시지 전송 (Airflow 외 모든 곳에서 호출 가능)"""
    slack_url = os.environ.get("SLACK_WEBHOOK_URL")
    if not slack_url:
        raise ValueError("SLACK_WEBHOOK_URL is not set")

    message = {"text": text}

    try:
        response = requests.post(slack_url, json=message, timeout=5)
        response.raise_for_status()
    except Exception as e:
        print(f"[Slack Alert Error] 전송 실패: {e}")


def slack_kv(*pairs):
    """("key","value") 튜플 리스트를 Slack markdown bullet 로 변환"""
    lines = []
    for k, v in pairs:
        if v is None:
            v = "-"
        lines.append(f"- *{k}*: `{v}`")
    return "\n".join(lines)


def notify_skip(title: str, **fields):
    text = f"*⏭️ SKIP* - *{title}*\n" + slack_kv(*fields.items())
    send_slack_alert(text)


def notify_info(title: str, **fields):
    text = f"*ℹ️ INFO* - *{title}*\n" + slack_kv(*fields.items())
    send_slack_alert(text)


def notify_success(title: str, **fields):
    text = f"*✅ SUCCESS* - *{title}*\n" + slack_kv(*fields.items())
    send_slack_alert(text)


def notify_fail(title: str, **fields):
    text = f"*❌ FAIL* - *{title}*\n" + slack_kv(*fields.items())
    send_slack_alert(text)


def alert_slack(context):
    """Airflow DAG 실패 알림 콜백 함수"""
    dag_id = context.get("dag").dag_id
    task_id = context.get("task_instance").task_id
    execution_date = context.get("ts")
    dag_run_id = context.get("run_id")

    base_url = os.environ.get("AIRFLOW__WEBSERVER__WEB_SERVER_BASE_URL")
    log_url = f"{base_url}/dags/{dag_id}/runs/{dag_run_id}/tasks/{task_id}"

    text = f"""
*🔥 Airflow DAG 실패 알림!*
- *DAG*: `{dag_id}`
- *Task*: `{task_id}`
- *Time*: `{execution_date}`
- *Log*: <{log_url}|로그 바로가기>
""".strip()

    send_slack_alert(text)
