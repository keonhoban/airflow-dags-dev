# dags/mlops_lib/dp/store.py
from __future__ import annotations

import json
from datetime import datetime, timezone, timedelta
from airflow.utils.log.logging_mixin import LoggingMixin
from jinja2 import Template

from .s3 import get_s3_client, parse_s3_uri

logger = LoggingMixin().log
KST = timezone(timedelta(hours=9))


def _kst_now_iso():
    return datetime.now(KST).isoformat()


def _version_id(exec_date=None):
    dt = exec_date.astimezone(KST) if exec_date else datetime.now(KST)
    return "v_" + dt.strftime("%Y%m%dT%H%M%S")


def _read_local_text(path: str) -> str:
    with open(path, "r", encoding="utf-8") as f:
        return f.read()


def store_features(
    feature_base: str,
    pipeline_name: str,
    feature_set: str,
    metadata_tpl_path: str,
    ti,
) -> None:
    """
    XCom(build_features 결과) -> feature_base 아래로 저장
    feature_base 예:
      s3://bucket/features/feature-store
    저장 구조:
      <feature_base>/<feature_set>/<version>/
        - features.csv
        - schema.json
        - metadata.json
    """
    s3 = get_s3_client()

    schema = ti.xcom_pull(key="fs_schema", task_ids="build_features")
    schema_hash = ti.xcom_pull(key="fs_schema_hash", task_ids="build_features")
    features_csv = ti.xcom_pull(key="fs_features_csv", task_ids="build_features")
    rows = ti.xcom_pull(key="fs_feature_rows", task_ids="build_features")
    source_raw = ti.xcom_pull(key="dp_raw_path", task_ids="build_features") or ti.xcom_pull(
        key="dp_raw_path", task_ids="extract_raw_data"
    )

    if not features_csv:
        raise ValueError("[FS] features_csv 누락 (build_features 결과 확인 필요)")

    exec_date = getattr(ti, "execution_date", None)
    ver = _version_id(exec_date)

    bkt, prefix = parse_s3_uri(feature_base)
    prefix = prefix.rstrip("/") + f"/{feature_set}/{ver}/"

    feature_uri = f"s3://{bkt}/{prefix}features.csv"

    tpl = Template(_read_local_text(metadata_tpl_path))
    meta_str = tpl.render(
        version=ver,
        generated_at=_kst_now_iso(),
        source=source_raw,
        pipeline=pipeline_name,
        schema_hash=schema_hash,
        feature_uri=feature_uri,
        feature_set=feature_set,
    )

    s3.put_object(
        Bucket=bkt,
        Key=f"{prefix}features.csv",
        Body=features_csv.encode("utf-8"),
        ContentType="text/csv",
    )
    s3.put_object(
        Bucket=bkt,
        Key=f"{prefix}schema.json",
        Body=json.dumps(schema, ensure_ascii=False, indent=2).encode("utf-8"),
        ContentType="application/json",
    )
    s3.put_object(
        Bucket=bkt,
        Key=f"{prefix}metadata.json",
        Body=meta_str.encode("utf-8"),
        ContentType="application/json",
    )

    ti.xcom_push(key="fs_version", value=ver)
    ti.xcom_push(key="fs_prefix", value=f"s3://{bkt}/{prefix}")
    ti.xcom_push(key="fs_feature_uri", value=feature_uri)

    logger.info("[FS] store_features OK prefix=s3://%s/%s rows=%s", bkt, prefix, rows)
