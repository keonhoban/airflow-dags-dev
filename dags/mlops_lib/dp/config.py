# dags/mlops_lib/dp/config.py
from __future__ import annotations

import os
from airflow.models import Variable


def cfg(key: str, default=None, *, required: bool = False):
    v = os.getenv(key)
    if v is not None and str(v).strip() != "":
        return v
    try:
        if default is None:
            v = Variable.get(key)
        else:
            v = Variable.get(key, default_var=str(default))
        if v is not None and str(v).strip() != "":
            return v
    except Exception:
        pass
    if required:
        raise RuntimeError(f"[DP Config] missing required key: {key}")
    return default


def dp_bucket() -> str:
    return cfg("DP_BUCKET", required=True)


def dp_prefix() -> str:
    return cfg("DP_PREFIX", "feature-store/user_features")


def dp_latest_key() -> str:
    return f"{dp_prefix()}/latest/features.parquet"


def dp_versioned_key(version: str) -> str:
    return f"{dp_prefix()}/{version}/features.parquet"


def dp_metadata_key(version: str) -> str:
    return f"{dp_prefix()}/{version}/metadata.json"

