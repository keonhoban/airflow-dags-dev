from __future__ import annotations
import requests
from utils.config import fastapi_base_url, fastapi_token
from utils.slack import success

def reload_variant(alias: str):
    base = fastapi_base_url().rstrip("/")
    url = f"{base}/variant/{alias}/reload"
    r = requests.post(url, headers={"x-token": fastapi_token()}, timeout=15, verify=False)
    if r.status_code >= 400:
        raise RuntimeError(f"FastAPI reload failed: {r.status_code} {r.text[:500]}")
    return True

def reload_and_notify(alias: str, env: str):
    reload_variant(alias)
    success("FastAPI reload completed", env=env, alias=alias)

