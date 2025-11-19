# ml_code/data_pipeline.py

import io
from urllib.parse import urlparse

import boto3
import pandas as pd
from airflow.utils.log.logging_mixin import LoggingMixin

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
    - AWS_ACCESS_KEY_ID / AWS_SECRET_ACCESS_KEY / AWS_DEFAULT_REGION
      또는 IAM Role(예: IRSA)로 자격 증명이 잡혀 있어야 합니다.
    """
    return boto3.client("s3")


# -----------------------
# Step 1: RAW 데이터 수집
# -----------------------

def extract_raw_data(raw_path: str, pipeline_name: str, ti):
    """
    ✅ Step 1: RAW 데이터 수집 (S3 기준)
    - 이 단계에서는 'S3에 오늘자 RAW 파일이 존재하는지' 확인만 합니다.
    - 필요하다면 외부 버킷 → 내부 버킷 copy 로직을 넣을 수도 있습니다.
    """
    s3 = _get_s3_client()
    bucket, key = _parse_s3_uri(raw_path)

    logger.info(
        "[DP] pipeline=%s step=extract_raw_data action=head_object bucket=%s key=%s",
        pipeline_name,
        bucket,
        key,
    )

    # 파일 존재 여부 체크 (없으면 에러 발생)
    s3.head_object(Bucket=bucket, Key=key)

    logger.info(
        "[DP] pipeline=%s step=extract_raw_data status=success raw_path=%s",
        pipeline_name,
        raw_path,
    )

    ti.xcom_push(key="dp_raw_path", value=raw_path)


# -----------------------
# Step 2: 데이터 검증
# -----------------------

def validate_data(raw_path: str, pipeline_name: str, ti):
    """
    ✅ Step 2: 데이터 검증
    - S3에서 CSV를 읽어서 row 수, null 비율 등을 계산합니다.
    """
    s3 = _get_s3_client()
    bucket, key = _parse_s3_uri(raw_path)

    logger.info(
        "[DP] pipeline=%s step=validate_data action=get_object bucket=%s key=%s",
        pipeline_name,
        bucket,
        key,
    )

    obj = s3.get_object(Bucket=bucket, Key=key)
    body = obj["Body"].read()

    # CSV 기준 예시
    df = pd.read_csv(io.BytesIO(body))

    rows = len(df)
    total_cells = df.shape[0] * df.shape[1] if rows > 0 else 0
    null_count = int(df.isna().sum().sum())
    null_rate = (null_count / total_cells) if total_cells > 0 else 0.0

    valid = True
    # 간단한 룰 예시: 전체 행이 1 이상이고, null_rate < 0.5
    if rows == 0 or null_rate > 0.5:
        valid = False

    logger.info(
        (
            "[DP] pipeline=%s step=validate_data status=%s "
            "rows=%d null_count=%d null_rate=%.4f raw_path=%s"
        ),
        pipeline_name,
        "success" if valid else "failed",
        rows,
        null_count,
        null_rate,
        raw_path,
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
    """
    ✅ Step 3: Feature 가공
    - 다시 S3에서 raw 데이터를 읽어와 간단한 가공을 수행합니다.
    - 여기서는 예시로 numeric 컬럼만 남기고,
      간단한 파생 컬럼(row_sum)을 추가합니다.
    """
    s3 = _get_s3_client()
    raw_bucket, raw_key = _parse_s3_uri(raw_path)

    logger.info(
        "[DP] pipeline=%s step=build_features action=get_object bucket=%s key=%s",
        pipeline_name,
        raw_bucket,
        raw_key,
    )

    obj = s3.get_object(Bucket=raw_bucket, Key=raw_key)
    body = obj["Body"].read()

    df = pd.read_csv(io.BytesIO(body))

    # 숫자형 컬럼만 남기기
    num_df = df.select_dtypes(include=["number"]).copy()

    # 간단한 파생 컬럼: 각 행의 숫자 합
    if not num_df.empty:
        num_df["row_sum"] = num_df.sum(axis=1)
    else:
        num_df["row_sum"] = 0

    feature_rows = len(num_df)

    logger.info(
        "[DP] pipeline=%s step=build_features status=success rows_in=%d rows_out=%d feature_path=%s",
        pipeline_name,
        len(df),
        feature_rows,
        feature_path,
    )

    # 다음 step에서 저장을 위해 df 전체를 XCom에 직접 싣기보다,
    # CSV 문자열로 변환해서 넣습니다. (소규모 데이터 기준 예시)
    buf = io.StringIO()
    num_df.to_csv(buf, index=False)
    feature_csv = buf.getvalue()

    ti.xcom_push(key="dp_feature_csv", value=feature_csv)
    ti.xcom_push(key="dp_feature_rows", value=feature_rows)
    ti.xcom_push(key="dp_feature_path", value=feature_path)


# -----------------------
# Step 4: Feature 저장 (S3)
# -----------------------

def store_features(feature_path: str, pipeline_name: str, ti):
    """
    ✅ Step 4: Feature 저장
    - 이전 단계에서 만든 CSV 문자열을 S3에 업로드합니다.
    """
    feature_csv = ti.xcom_pull(key="dp_feature_csv", task_ids="build_features")
    feature_rows = ti.xcom_pull(key="dp_feature_rows", task_ids="build_features") or 0

    if feature_csv is None:
        raise ValueError("[DP] feature_csv XCom 누락")

    bucket, key = _parse_s3_uri(feature_path)
    s3 = _get_s3_client()

    logger.info(
        "[DP] pipeline=%s step=store_features action=put_object bucket=%s key=%s rows=%d",
        pipeline_name,
        bucket,
        key,
        feature_rows,
    )

    s3.put_object(
        Bucket=bucket,
        Key=key,
        Body=feature_csv.encode("utf-8"),
        ContentType="text/csv",
    )

    logger.info(
        "[DP] pipeline=%s step=store_features status=success feature_path=%s feature_rows=%d",
        pipeline_name,
        feature_path,
        feature_rows,
    )

    ti.xcom_push(key="dp_stored_rows", value=feature_rows)


# -----------------------
# Step 5: 실행 요약
# -----------------------

def summarize_run(pipeline_name: str, ti):
    """
    ✅ 마지막 Step: 실행 요약 로그
    - Loki / Grafana에서 전체 파이프라인 상태를 한눈에 보기 위한 정보.
    """
    raw_path = ti.xcom_pull(key="dp_raw_path", task_ids="extract_raw_data")
    rows = ti.xcom_pull(key="dp_rows", task_ids="validate_data")
    null_rate = ti.xcom_pull(key="dp_null_rate", task_ids="validate_data")
    valid = ti.xcom_pull(key="dp_valid", task_ids="validate_data")
    feature_path = ti.xcom_pull(key="dp_feature_path", task_ids="build_features")
    stored_rows = ti.xcom_pull(key="dp_stored_rows", task_ids="store_features")

    logger.info(
        (
            "[DP] pipeline=%s step=summarize_run "
            "raw_path=%s feature_path=%s "
            "rows=%s null_rate=%s valid=%s stored_rows=%s"
        ),
        pipeline_name,
        raw_path,
        feature_path,
        rows,
        null_rate,
        valid,
        stored_rows,
    )
