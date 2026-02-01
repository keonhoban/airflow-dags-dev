from __future__ import annotations
from urllib.parse import urlparse
import json, os
from datetime import datetime, timezone
from zoneinfo import ZoneInfo

def parse_s3_uri(uri: str) -> tuple[str, str]:
    p = urlparse(uri)
    if p.scheme != "s3":
        raise ValueError(f"invalid s3 uri: {uri}")
    bucket = p.netloc
    key = p.path.lstrip("/")
    if not bucket or not key:
        raise ValueError(f"invalid s3 uri: {uri}")
    return bucket, key

def kst_now_iso():
    return datetime.now(ZoneInfo("Asia/Seoul")).isoformat()

def utc_ts():
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S")

def atomic_json_write(path: str, obj: dict):
    tmp = path + ".tmp"
    with open(tmp, "w") as f:
        json.dump(obj, f, indent=2)
    os.replace(tmp, path)

