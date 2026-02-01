from __future__ import annotations
import json, hashlib


def read_local_json(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def json_canonical_bytes(obj: dict) -> bytes:
    return json.dumps(obj, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")


def sha256_hex(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()


def load_schema(schema_path: str, expected_feature_set: str) -> tuple[dict, str]:
    schema = read_local_json(schema_path)
    if schema.get("feature_set") != expected_feature_set:
        raise ValueError(f"schema feature_set mismatch: {schema.get('feature_set')} != {expected_feature_set}")

    cols = [c["name"] for c in schema.get("columns", [])]
    if "user_id" not in cols:
        raise ValueError("schema missing user_id")

    # ✅ 권장: 학습을 한다면 label까지 schema로 고정 (면접/재현성에 매우 강함)
    # 다만, 기존 운영 중인 환경에서 즉시 전환이 부담이면 build.py에서 보정하므로 강제는 하지 않음.
    # 여기서는 경고만 남기되, 원하면 raise로 바꿔도 됩니다.
    if "label" not in cols:
        # raise ValueError("schema missing label (recommended for supervised training)")
        pass

    schema_hash = sha256_hex(json_canonical_bytes(schema))
    return schema, schema_hash

