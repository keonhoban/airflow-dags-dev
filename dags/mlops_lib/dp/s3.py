# dags/mlops_lib/dp/s3.py
from __future__ import annotations

import io
import json
from urllib.parse import urlparse

import boto3
import pandas as pd


# -----------------------
# Client
# -----------------------
def s3_client():
    return boto3.client("s3")


# ✅ 제출용/호환용 별칭
def get_s3_client():
    return s3_client()


# -----------------------
# URI helpers
# -----------------------
def s3_uri(bucket: str, key: str) -> str:
    key = key.lstrip("/")
    return f"s3://{bucket}/{key}"


def parse_s3_uri(uri: str) -> tuple[str, str]:
    """
    s3://bucket/key -> (bucket, key)
    """
    if not uri or not isinstance(uri, str):
        raise ValueError("invalid s3 uri: empty")

    p = urlparse(uri)
    if p.scheme != "s3" or not p.netloc:
        raise ValueError(f"invalid s3 uri: {uri}")

    bucket = p.netloc
    key = p.path.lstrip("/")
    if not key:
        raise ValueError(f"invalid s3 uri (missing key): {uri}")
    return bucket, key


# -----------------------
# Put helpers
# -----------------------
def put_parquet(bucket: str, key: str, df: pd.DataFrame):
    buf = io.BytesIO()
    df.to_parquet(buf, index=False)
    buf.seek(0)
    s3_client().put_object(Bucket=bucket, Key=key, Body=buf.getvalue())


def put_json(bucket: str, key: str, payload: dict):
    body = json.dumps(payload, ensure_ascii=False, indent=2).encode("utf-8")
    s3_client().put_object(
        Bucket=bucket,
        Key=key,
        Body=body,
        ContentType="application/json",
    )

