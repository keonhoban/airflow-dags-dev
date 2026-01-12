# dags/mlops_lib/dp/s3.py
from __future__ import annotations

from urllib.parse import urlparse
import boto3


def parse_s3_uri(uri: str) -> tuple[str, str]:
    p = urlparse(uri)
    if p.scheme != "s3":
        raise ValueError(f"지원하지 않는 URI 스키마입니다: {uri}")
    bucket = p.netloc
    key = p.path.lstrip("/")
    if not bucket or not key:
        raise ValueError(f"잘못된 s3 uri 입니다: {uri}")
    return bucket, key


def get_s3_client():
    # boto3 표준 credential chain 사용:
    # 1) env, 2) shared credentials(~/.aws/credentials), 3) IAM role 등
    return boto3.client("s3")
