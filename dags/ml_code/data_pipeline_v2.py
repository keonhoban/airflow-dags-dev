# ml_code/data_pipeline_v2.py

import io
import csv
from urllib.parse import urlparse

import boto3
from airflow.utils.log.logging_mixin import LoggingMixin
from datetime import datetime, timezone, timedelta
import json

# 한국 시간 정의 (UTC+9)
KST = timezone(timedelta(hours=9))

logger = LoggingMixin().log


# -----------------------
# 공통 유틸
# -----------------------

def _parse_s3_uri(uri: str):
    """
    s3://bucket/key 형태의 URI를 (bucket, key)로 분리합니다.
    """
    parsed = urlparse(uri)
    if parsed.scheme != "s3":
        raise ValueError(f"지원하지 않는 URI 스킴입니다: {uri}")
    bucket = parsed.netloc
    key = parsed.path.lstrip("/")
    return bucket, key


def _get_s3_client():
    """
    boto3 S3 client 생성.
    """
    return boto3.client("s3")


# -----------------------
# Step 1: RAW 데이터 수집
# -----------------------

def extract_raw_data(raw_path: str, pipeline_name: str, ti):
    s3 = _get_s3_client()
    bucket, key = _parse_s3_uri(raw_path)

    logger.info(
        "[DP] pipeline=%s step=extract_raw_data action=head_object bucket=%s key=%s",
        pipeline_name, bucket, key,
    )

    s3.head_object(Bucket=bucket, Key=key)

    logger.info(
        "[DP] pipeline=%s step=extract_raw_data status=success raw_path=%s",
        pipeline_name, raw_path,
    )

    ti.xcom_push(key="dp_raw_path", value=raw_path)


# -----------------------
# Step 2: 데이터 검증
# -----------------------

def validate_data(raw_path: str, pipeline_name: str, ti):
    s3 = _get_s3_client()
    bucket, key = _parse_s3_uri(raw_path)

    logger.info(
        "[DP] pipeline=%s step=validate_data action=get_object bucket=%s key=%s",
        pipeline_name, bucket, key,
    )

    obj = s3.get_object(Bucket=bucket, Key=key)
    text = obj["Body"].read().decode("utf-8")

    reader = csv.reader(io.StringIO(text))
    rows = 0
    null_count = 0
    total_cells = 0
    header = None

    for i, row in enumerate(reader):
        if i == 0:
            header = row
            continue
        rows += 1
        total_cells += len(row)
        null_count += sum(1 for cell in row if cell == "")

    null_rate = (null_count / total_cells) if total_cells > 0 else 0.0
    valid = not (rows == 0 or null_rate > 0.5)

    logger.info(
        "[DP] pipeline=%s step=validate_data status=%s rows=%d null_count=%d null_rate=%.4f",
        pipeline_name, "success" if valid else "failed", rows, null_count, null_rate,
    )

    ti.xcom_push(key="dp_rows", value=rows)
    ti.xcom_push(key="dp_null_rate", value=null_rate)
    ti.xcom_push(key="dp_valid", value=valid)

    if not valid:
        raise ValueError(
            f"[DP] 데이터 검증 실패 (rows={rows}, null_rate={null_rate:.4f})"
        )


# -----------------------
# Step 3: Feature 가공
# -----------------------

def build_features(raw_path: str, feature_path: str, pipeline_name: str, ti):
    s3 = _get_s3_client()
    bucket, key = _parse_s3_uri(raw_path)

    logger.info(
        "[DP] pipeline=%s step=build_features action=get_object bucket=%s key=%s",
        pipeline_name, bucket, key,
    )

    obj = s3.get_object(Bucket=bucket, Key=key)
    text = obj["Body"].read().decode("utf-8")

    reader = csv.reader(io.StringIO(text))
    header = None
    rows = []

    for i, row in enumerate(reader):
        if i == 0:
            header = row
        else:
            rows.append(row)

    if not header:
        raise ValueError("[DP] 헤더가 없는 CSV입니다.")

    numeric_candidates = {
        "user_id", "is_premium", "event_value", "amount", "session_length_sec"
    }

    numeric_indices = [
        idx for idx, col in enumerate(header) if col in numeric_candidates
    ]

    feature_rows = []
    for row in rows:
        numeric_values = []
        for idx in numeric_indices:
            if idx < len(row):
                try:
                    numeric_values.append(float(row[idx] or 0))
                except ValueError:
                    numeric_values.append(0.0)
        feature_rows.append([sum(numeric_values)])

    logger.info(
        "[DP] pipeline=%s step=build_features status=success rows_in=%d rows_out=%d",
        pipeline_name, len(rows), len(feature_rows),
    )

    # --- schema.json ---
    schema = {
        "feature_version": None,
        "columns": {"row_sum": "float"},
        "created_at": datetime.now(KST).isoformat(),
        "pipeline_name": pipeline_name,
    }

    # --- metadata.json ---
    metadata = {
        "source_raw_path": raw_path,
        "rows_raw": ti.xcom_pull(key="dp_rows", task_ids="validate_data"),
        "rows_feature": len(feature_rows),
        "raw_null_rate": ti.xcom_pull(key="dp_null_rate", task_ids="validate_data"),
        "raw_valid": ti.xcom_pull(key="dp_valid", task_ids="validate_data"),
        "created_at": datetime.now(KST).isoformat(),
        "pipeline_name": pipeline_name,
    }

    # CSV 문자열로 직렬화
    buf = io.StringIO()
    writer = csv.writer(buf)
    writer.writerow(["row_sum"])
    writer.writerows(feature_rows)
    feature_csv = buf.getvalue()

    ti.xcom_push(key="dp_feature_csv", value=feature_csv)
    ti.xcom_push(key="dp_feature_rows", value=len(feature_rows))
    ti.xcom_push(key="dp_feature_path", value=feature_path)
    ti.xcom_push(key="dp_schema_dict", value=schema)
    ti.xcom_push(key="dp_metadata_dict", value=metadata)


# -----------------------
# Step 4: Feature 저장
# -----------------------

def store_features(feature_path: str, pipeline_name: str, ti):
    feature_csv = ti.xcom_pull(key="dp_feature_csv", task_ids="build_features")
    feature_rows = ti.xcom_pull(key="dp_feature_rows", task_ids="build_features")
    schema = ti.xcom_pull(key="dp_schema_dict", task_ids="build_features")
    metadata = ti.xcom_pull(key="dp_metadata_dict", task_ids="build_features")

    if feature_csv is None:
        raise ValueError("[DP] feature_csv 누락")

    exec_date = getattr(ti, "execution_date", None)

    if exec_date is None:
        exec_date = datetime.now(KST)
    else:
        if exec_date.tzinfo is None:
            exec_date = exec_date.replace(tzinfo=timezone.utc).astimezone(KST)
        else:
            exec_date = exec_date.astimezone(KST)

    run_ts = exec_date.strftime("%Y%m%dT%H%M%S")
    version_id = f"v_{run_ts}"

    bucket, prefix = _parse_s3_uri(feature_path)
    if not prefix.endswith("/"):
        prefix += "/"

    base_prefix = f"{prefix}{version_id}/"

    s3 = _get_s3_client()

    # feature.csv
    s3.put_object(
        Bucket=bucket,
        Key=f"{base_prefix}feature.csv",
        Body=feature_csv.encode("utf-8"),
        ContentType="text/csv",
    )

    # schema.json
    schema["feature_version"] = version_id
    s3.put_object(
        Bucket=bucket,
        Key=f"{base_prefix}schema.json",
        Body=json.dumps(schema, ensure_ascii=False, indent=2).encode("utf-8"),
        ContentType="application/json",
    )

    # metadata.json
    s3.put_object(
        Bucket=bucket,
        Key=f"{base_prefix}metadata.json",
        Body=json.dumps(metadata, ensure_ascii=False, indent=2).encode("utf-8"),
        ContentType="application/json",
    )

    logger.info(
        "[DP] pipeline=%s step=store_features status=success prefix=%s version=%s",
        pipeline_name, base_prefix, version_id,
    )

    ti.xcom_push(key="dp_stored_rows", value=feature_rows)
    ti.xcom_push(key="dp_feature_version", value=version_id)
    ti.xcom_push(key="dp_feature_prefix", value=base_prefix)


# -----------------------
# Step 5: 실행 요약
# -----------------------

def summarize_run(pipeline_name: str, ti):
    raw_path = ti.xcom_pull(key="dp_raw_path", task_ids="extract_raw_data")
    rows = ti.xcom_pull(key="dp_rows", task_ids="validate_data")
    null_rate = ti.xcom_pull(key="dp_null_rate", task_ids="validate_data")
    valid = ti.xcom_pull(key="dp_valid", task_ids="validate_data")

    feature_prefix = ti.xcom_pull(key="dp_feature_prefix", task_ids="store_features")
    feature_version = ti.xcom_pull(key="dp_feature_version", task_ids="store_features")
    stored_rows = ti.xcom_pull(key="dp_stored_rows", task_ids="store_features")

    logger.info(
        "[DP] pipeline=%s step=summarize_run raw_path=%s feature_prefix=%s "
        "feature_version=%s rows=%s null_rate=%.4f valid=%s stored_rows=%s",
        pipeline_name,
        raw_path,
        feature_prefix,
        feature_version,
        rows,
        null_rate,
        valid,
        stored_rows,
    )
