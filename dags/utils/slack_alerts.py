# dags/utils/slack_alerts.py

import requests
import os

def send_slack_alert(text: str): # Slack 메시지 전송 (Airflow 외 모든 곳에서 호출 가능)
    slack_url = os.environ.get("SLACK_WEBHOOK_URL")
    if not slack_url:
        raise ValueError("SLACK_WEBHOOK_URL is not set")

    message = {"text": text}

    try:
        response = requests.post(slack_url, json=message)
        response.raise_for_status()
    except Exception as e:
        print(f"[Slack Alert Error] 전송 실패: {e}")

def alert_slack(context): # Airflow DAG 실패 알림 콜백 함수
    dag_id = context.get("dag").dag_id
    task_id = context.get("task_instance").task_id
    execution_date = context.get("ts")
    dag_run_id = context.get("run_id")  # ✅ 핵심: Airflow 2.x 이상은 run_id 기반


    base_url = os.environ.get("AIRFLOW__WEBSERVER__WEB_SERVER_BASE_URL")

    # ✅ 새로운 로그 URL 포맷 (Airflow 2.x+ 호환)
    log_url = f"{base_url}/dags/{dag_id}/runs/{dag_run_id}/tasks/{task_id}"

    text = f"""
*🔥 Airflow DAG 실패 알림!*
*DAG*: `{dag_id}`
*Task*: `{task_id}`
*Time*: `{execution_date}`
*Log*: <{log_url}|로그 바로가기>
    """

    send_slack_alert(text)
