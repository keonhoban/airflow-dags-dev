from __future__ import annotations
import os, json, requests

SLACK_WEBHOOK = os.getenv("SLACK_WEBHOOK_URL", "")

def _post(payload: dict):
    if not SLACK_WEBHOOK:
        return
    try:
        requests.post(SLACK_WEBHOOK, json=payload, timeout=5)
    except Exception:
        pass

def info(title: str, **fields):
    _post({"text": f"INFO - {title}\n" + "\n".join([f"- {k}: {v}" for k, v in fields.items()])})

def success(title: str, **fields):
    _post({"text": f"SUCCESS - {title}\n" + "\n".join([f"- {k}: {v}" for k, v in fields.items()])})

def skip(title: str, **fields):
    _post({"text": f"SKIP - {title}\n" + "\n".join([f"- {k}: {v}" for k, v in fields.items()])})

