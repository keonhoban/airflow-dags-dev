# dags/utils/slack_alerts.py
import os
import requests


def send_slack_alert(text: str):
    slack_url = os.environ.get("SLACK_WEBHOOK_URL")
    if not slack_url:
        # Slack 없는 환경에서도 DAG는 돌아야 합니다.
        return
    try:
        requests.post(slack_url, json={"text": text}, timeout=5).raise_for_status()
    except Exception:
        pass


def slack_kv(**fields):
    lines = []
    for k, v in fields.items():
        lines.append(f"- *{k}*: `{v if v is not None else '-'}`")
    return "\n".join(lines)


def notify_info(title: str, **fields):
    send_slack_alert(f"*ℹ️ INFO* - *{title}*\n{slack_kv(**fields)}")


def notify_success(title: str, **fields):
    send_slack_alert(f"*✅ SUCCESS* - *{title}*\n{slack_kv(**fields)}")


def notify_skip(title: str, **fields):
    send_slack_alert(f"*⏭️ SKIP* - *{title}*\n{slack_kv(**fields)}")


def alert_slack(context):
    dag_id = context.get("dag").dag_id
    task_id = context.get("task_instance").task_id
    ts = context.get("ts")
    run_id = context.get("run_id")

    base_url = os.environ.get("AIRFLOW__WEBSERVER__WEB_SERVER_BASE_URL", "")
    log_url = f"{base_url}/dags/{dag_id}/runs/{run_id}/tasks/{task_id}" if base_url else "-"

    send_slack_alert(
        "\n".join(
            [
                "*🔥 Airflow DAG failed*",
                f"- *DAG*: `{dag_id}`",
                f"- *Task*: `{task_id}`",
                f"- *Time*: `{ts}`",
                f"- *Log*: {log_url}",
            ]
        )
    )

