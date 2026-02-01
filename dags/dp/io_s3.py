from __future__ import annotations
import pandas as pd
from datetime import datetime, timezone

def now_version(prefix: str) -> str:
    ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S")
    return f"v_{ts}"

def s3_path(prefix: str, version: str, name: str) -> str:
    # prefix: s3://bucket/path/
    return f"{prefix}{version}/{name}"

def s3_latest(prefix: str, name: str) -> str:
    return f"{prefix}latest/{name}"

def store_parquet_version_and_latest(df: pd.DataFrame, prefix: str, name: str = "features.parquet") -> dict:
    """
    returns:
      fs_version, fs_feature_uri(versioned), fs_latest_uri
    """
    version = now_version(prefix)
    ver_uri = s3_path(prefix, version, name)
    latest_uri = s3_latest(prefix, name)

    df.to_parquet(ver_uri, index=False)
    df.to_parquet(latest_uri, index=False)

    return {
        "fs_version": version,
        "fs_feature_uri": ver_uri,
        "fs_latest_uri": latest_uri,
    }

