# dags/mlops_lib/dp/s3.py
from __future__ import annotations

from urllib.parse import urlparse
import boto3


def parse_s3_uri(uri: str) -> tuple[str, str]:
    p = urlparse(uri)
    if p.scheme != "s3":
        raise ValueError(f"invalid s3 uri: {uri}")
    bucket = p.netloc
    key = p.path.lstrip("/")
    if not bucket or not key:
        raise ValueError(f"invalid s3 uri: {uri}")
    return bucket, key


def get_s3_client():
    return boto3.client("s3")

