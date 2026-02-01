# dags/utils/slack_alerts.py
import os
import requests


def send_slack_alert(text: str):
    url = os.environ.get("SLACK_WEBHOOK_URL")
    if not url:
        return  # 제출용 최소: slack 없으면 조용히 무시

    try:
        requests.post(url, json={"text": text}, timeout=5).raise_for_status()
    except Exception:
        pass


def _kv(**fields):
    lines = []
    for k, v in fields.items():
        lines.append(f"- *{k}*: `{v}`")
    return "\n".join(lines)


def notify_info(title: str, **fields):
    send_slack_alert(f"*ℹ️ {title}*\n{_kv(**fields)}")


def notify_success(title: str, **fields):
    send_slack_alert(f"*✅ {title}*\n{_kv(**fields)}")


def notify_fail(title: str, **fields):
    send_slack_alert(f"*❌ {title}*\n{_kv(**fields)}")


def notify_skip(title: str, **fields):
    send_slack_alert(f"*⏭️ {title}*\n{_kv(**fields)}")


def alert_slack(context):
    dag_id = context.get("dag").dag_id
    task_id = context.get("task_instance").task_id
    ts = context.get("ts")
    run_id = context.get("run_id")

    base_url = os.environ.get("AIRFLOW__WEBSERVER__WEB_SERVER_BASE_URL", "")
    log_url = f"{base_url}/dags/{dag_id}/runs/{run_id}/tasks/{task_id}" if base_url else "(no webserver url)"

    send_slack_alert(
        "\n".join(
            [
                "*🔥 Airflow DAG 실패 알림*",
                f"- *DAG*: `{dag_id}`",
                f"- *Task*: `{task_id}`",
                f"- *Time*: `{ts}`",
                f"- *Log*: {log_url}",
            ]
        )
    )

