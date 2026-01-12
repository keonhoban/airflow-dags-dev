# dags/mlops_lib/dp/feature_schema.py
from __future__ import annotations

import json
import hashlib


def read_local_json(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def json_canonical_bytes(obj: dict) -> bytes:
    return json.dumps(
        obj, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")


def sha256_hex(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()


def load_schema(schema_path: str, expected_feature_set: str) -> tuple[dict, str]:
    schema = read_local_json(schema_path)
    if schema.get("feature_set") != expected_feature_set:
        raise ValueError(
            f"[FS] schema feature_set mismatch: {schema.get('feature_set')} != {expected_feature_set}"
        )

    cols = [c["name"] for c in schema.get("columns", [])]
    if "user_id" not in cols:
        raise ValueError("[FS] schema에 user_id가 없습니다")

    schema_hash = sha256_hex(json_canonical_bytes(schema))
    return schema, schema_hash
