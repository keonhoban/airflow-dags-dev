# dags/mlops_lib/dp/tasks.py
from __future__ import annotations

from airflow.utils.log.logging_mixin import LoggingMixin

from .config import load_pipeline_config
from .s3 import get_s3_client, parse_s3_uri
from .build import build_features as _build_features
from .store import store_features as _store_features

logger = LoggingMixin().log


def task_extract_raw_data(**context):
    cfg = load_pipeline_config()
    ti = context["ti"]

    # 존재 확인(head)
    s3 = get_s3_client()
    bucket, key = parse_s3_uri(cfg.raw_path)
    logger.info(
        "[DP] pipeline=%s step=extract_raw_data action=head_object bucket=%s key=%s",
        cfg.pipeline_name, bucket, key
    )
    s3.head_object(Bucket=bucket, Key=key)

    ti.xcom_push(key="dp_raw_path", value=cfg.raw_path)


def task_validate_data(**context):
    """
    기존 v2 validate 로직을 그대로 쓰고 싶으면, 여기만 갈아끼우면 됩니다.
    지금은 최소 검증(행 수) 형태로 둡니다.
    """
    cfg = load_pipeline_config()
    ti = context["ti"]

    s3 = get_s3_client()
    bucket, key = parse_s3_uri(cfg.raw_path)
    obj = s3.get_object(Bucket=bucket, Key=key)
    text = obj["Body"].read().decode("utf-8")

    # 최소 검증: 데이터가 비어있지 않음
    lines = [ln for ln in text.splitlines() if ln.strip()]
    if len(lines) <= 1:
        raise ValueError("[DP] 데이터 검증 실패: 데이터가 비어있거나 헤더만 존재")

    ti.xcom_push(key="dp_valid", value=True)
    ti.xcom_push(key="dp_lines", value=len(lines))


def task_build_features(**context):
    cfg = load_pipeline_config()
    ti = context["ti"]

    _build_features(
        raw_path=cfg.raw_path,
        pipeline_name=cfg.pipeline_name,
        feature_set=cfg.feature_set,
        schema_path=cfg.schema_path,
        ti=ti,
    )


def task_store_features(**context):
    cfg = load_pipeline_config()
    ti = context["ti"]

    _store_features(
        feature_base=cfg.feature_base,
        pipeline_name=cfg.pipeline_name,
        feature_set=cfg.feature_set,
        metadata_tpl_path=cfg.metadata_tpl_path,
        ti=ti,
    )


def task_summarize_run(**context):
    cfg = load_pipeline_config()
    ti = context["ti"]

    logger.info(
        "[FS] summarize pipeline=%s raw=%s out_prefix=%s version=%s uri=%s",
        cfg.pipeline_name,
        ti.xcom_pull(key="dp_raw_path", task_ids="extract_raw_data"),
        ti.xcom_pull(key="fs_prefix", task_ids="store_features"),
        ti.xcom_pull(key="fs_version", task_ids="store_features"),
        ti.xcom_pull(key="fs_feature_uri", task_ids="store_features"),
    )
