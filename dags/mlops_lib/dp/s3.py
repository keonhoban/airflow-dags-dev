from __future__ import annotations

import os
from urllib.parse import urlparse

import boto3
from botocore.config import Config


def parse_s3_uri(uri: str) -> tuple[str, str]:
    p = urlparse(uri)
    if p.scheme != "s3":
        raise ValueError(f"Unsupported URI: {uri}")
    bucket = p.netloc
    key = p.path.lstrip("/")
    if not bucket or not key:
        raise ValueError(f"Bad s3 uri: {uri}")
    return bucket, key


def get_s3_client():
    """
    - dev: S3_ENDPOINT_URL이 있으면 MinIO로 붙음 (path-style 강제)
    - prod: 없으면 AWS 기본 엔드포인트 사용
    """
    endpoint = os.getenv("S3_ENDPOINT_URL")
    region = os.getenv("AWS_REGION") or os.getenv("AWS_DEFAULT_REGION") or "ap-northeast-2"

    # MinIO / S3-compatible
    if endpoint:
        return boto3.client(
            "s3",
            endpoint_url=endpoint,
            region_name=region,
            config=Config(s3={"addressing_style": "path"}),
        )

    # AWS S3
    return boto3.client("s3", region_name=region)

