from __future__ import annotations

import os
import requests


def _send(text: str):
    url = os.getenv("SLACK_WEBHOOK_URL")
    if not url:
        return
    try:
        requests.post(url, json={"text": text}, timeout=5).raise_for_status()
    except Exception:
        pass


def alert_slack(context):
    dag_id = context.get("dag").dag_id
    task_id = context.get("task_instance").task_id
    ts = context.get("ts")
    run_id = context.get("run_id")

    base_url = os.environ.get("AIRFLOW__WEBSERVER__WEB_SERVER_BASE_URL", "").rstrip("/")
    log_url = f"{base_url}/dags/{dag_id}/runs/{run_id}/tasks/{task_id}" if base_url else "(no base url)"

    text = (
        "*Airflow DAG 실패 알림*\n"
        f"- DAG: `{dag_id}`\n"
        f"- Task: `{task_id}`\n"
        f"- Time: `{ts}`\n"
        f"- Log: {log_url}"
    )
    _send(text)


def notify(title: str, **fields):
    lines = [f"*{title}*"]
    for k, v in fields.items():
        lines.append(f"- *{k}*: `{v}`")
    _send("\n".join(lines))

