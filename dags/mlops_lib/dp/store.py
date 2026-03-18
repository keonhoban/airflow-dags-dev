from __future__ import annotations

import io
import json
from datetime import datetime, timezone, timedelta

import pandas as pd
from airflow.utils.log.logging_mixin import LoggingMixin
from jinja2 import Template

from ml_code.triton_xcom import xcom_pull_any
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


def store_features(feature_base: str, pipeline_name: str, feature_set: str, metadata_tpl_path: str, ti) -> None:
    s3 = get_s3_client()

    # TaskGroup 적용 시 task_id prefix가 dp.로 붙음
    BUILD_TASKS = ("dp.build_features", "build_features")
    EXTRACT_TASKS = ("dp.extract_raw_data", "extract_raw_data")

    schema = xcom_pull_any(ti, key="fs_schema", task_ids=BUILD_TASKS)
    schema_hash = xcom_pull_any(ti, key="fs_schema_hash", task_ids=BUILD_TASKS)
    features_csv_uri = xcom_pull_any(ti, key="fs_features_csv_uri", task_ids=BUILD_TASKS)
    rows = xcom_pull_any(ti, key="fs_feature_rows", task_ids=BUILD_TASKS)

    # dp_raw_path는 extract 단계에서 나오는 게 정상이므로 extract에서 우선 pull
    source_raw = xcom_pull_any(ti, key="dp_raw_path", task_ids=EXTRACT_TASKS)

    if not features_csv_uri:
        raise ValueError("fs_features_csv_uri missing — build_features가 staging 경로를 push하지 않았습니다")

    # XCom에는 URI만 전달 — 실제 CSV는 S3 staging에서 읽는다
    stg_bkt, stg_key = parse_s3_uri(features_csv_uri)
    stg_obj = s3.get_object(Bucket=stg_bkt, Key=stg_key)
    features_csv = stg_obj["Body"].read().decode("utf-8")
    logger.info("[FS] staging CSV loaded uri=%s", features_csv_uri)

    exec_date = getattr(ti, "execution_date", None)
    ver = _version_id(exec_date)

    bkt, base_prefix = parse_s3_uri(feature_base)
    base_prefix = base_prefix.rstrip("/") + f"/{feature_set}/"

    ver_prefix = base_prefix + f"{ver}/"
    latest_prefix = base_prefix + "latest/"

    feature_uri = f"s3://{bkt}/{ver_prefix}features.parquet"

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

    schema_bytes = json.dumps(schema, ensure_ascii=False, indent=2).encode("utf-8")
    meta_bytes = meta_str.encode("utf-8")
    feat_csv_bytes = features_csv.encode("utf-8")

    df = pd.read_csv(io.StringIO(features_csv))

    if "event_timestamp" not in df.columns:
        df["event_timestamp"] = pd.Timestamp.now(tz="Asia/Seoul").tz_convert("UTC")
    else:
        df["event_timestamp"] = pd.to_datetime(df["event_timestamp"], utc=True, errors="raise")

    parquet_buf = io.BytesIO()
    df.to_parquet(parquet_buf, index=False)
    parquet_bytes = parquet_buf.getvalue()

    def _put(prefix: str):
        s3.put_object(Bucket=bkt, Key=f"{prefix}features.csv", Body=feat_csv_bytes, ContentType="text/csv")
        s3.put_object(
            Bucket=bkt,
            Key=f"{prefix}features.parquet",
            Body=parquet_bytes,
            ContentType="application/octet-stream",
        )
        s3.put_object(Bucket=bkt, Key=f"{prefix}schema.json", Body=schema_bytes, ContentType="application/json")
        s3.put_object(Bucket=bkt, Key=f"{prefix}metadata.json", Body=meta_bytes, ContentType="application/json")

    _put(ver_prefix)
    _put(latest_prefix)

    ti.xcom_push(key="fs_version", value=ver)
    ti.xcom_push(key="fs_prefix", value=f"s3://{bkt}/{ver_prefix}")
    ti.xcom_push(key="fs_latest_prefix", value=f"s3://{bkt}/{latest_prefix}")
    ti.xcom_push(key="fs_feature_uri", value=feature_uri)

    logger.info("[FS] store_features OK version=%s rows=%s uri=%s", ver, rows, feature_uri)
