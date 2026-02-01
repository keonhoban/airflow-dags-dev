# dags/mlops_lib/dp/s3.py
from __future__ import annotations

import io
import json
import boto3
import pandas as pd


def s3_client():
    return boto3.client("s3")


def put_parquet(bucket: str, key: str, df: pd.DataFrame):
    buf = io.BytesIO()
    df.to_parquet(buf, index=False)
    buf.seek(0)
    s3_client().put_object(Bucket=bucket, Key=key, Body=buf.getvalue())


def put_json(bucket: str, key: str, payload: dict):
    body = json.dumps(payload, ensure_ascii=False, indent=2).encode("utf-8")
    s3_client().put_object(Bucket=bucket, Key=key, Body=body, ContentType="application/json")


def s3_uri(bucket: str, key: str) -> str:
    return f"s3://{bucket}/{key}"

