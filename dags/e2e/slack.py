from __future__ import annotations
import os, requests

def _send(text: str):
    url = os.getenv("SLACK_WEBHOOK_URL")
    if not url:
        return
    requests.post(url, json={"text": text}, timeout=5).raise_for_status()

def _kv(**fields):
    lines = []
    for k, v in fields.items():
        lines.append(f"- *{k}*: `{v if v is not None else '-'}`")
    return "\n".join(lines)

def notify_info(title: str, **fields):
    _send(f"*ℹ️ INFO* - *{title}*\n{_kv(**fields)}")

def notify_success(title: str, **fields):
    _send(f"*✅ SUCCESS* - *{title}*\n{_kv(**fields)}")

def notify_fail(title: str, **fields):
    _send(f"*❌ FAIL* - *{title}*\n{_kv(**fields)}")

def notify_skip(title: str, **fields):
    _send(f"*⏭️ SKIP* - *{title}*\n{_kv(**fields)}")

def alert_slack(context):
    dag_id = context.get("dag").dag_id
    task_id = context.get("task_instance").task_id
    ts = context.get("ts")
    run_id = context.get("run_id")
    base_url = os.environ.get("AIRFLOW__WEBSERVER__WEB_SERVER_BASE_URL", "")
    log_url = f"{base_url}/dags/{dag_id}/runs/{run_id}/tasks/{task_id}" if base_url else "-"
    _send(
        "*🔥 Airflow DAG Failure*\n"
        f"- *dag*: `{dag_id}`\n"
        f"- *task*: `{task_id}`\n"
        f"- *ts*: `{ts}`\n"
        f"- *log*: {log_url}"
    )

