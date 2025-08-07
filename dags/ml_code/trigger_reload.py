# ml_code/trigger_reload.py

import requests
from ml_code.config import get_fastapi_reload_url, get_reload_token
from airflow.utils.log.logging_mixin import LoggingMixin

logger = LoggingMixin().log

def trigger_reload(variant="A"):
    base_url = get_fastapi_reload_url()
    token = get_reload_token()

    try:
        url = f"{base_url}/variant/{variant}/reload"
        res = requests.post(url, headers={"x-token": token}, timeout=5)

        if res.status_code != 200:
            raise Exception(f"FastAPI reload 실패: {res.status_code} {res.text}")

        json_resp = res.json()
        if json_resp.get("status") != "success":
            raise Exception(f"FastAPI reload 실패 (응답 문제): {json_resp}")

        logger.info(f"[Reload 성공] variant={variant} → {json_resp}")
        return json_resp

    except Exception as e:
        raise RuntimeError(f"🔥 모델 reload 실패: {e}")
