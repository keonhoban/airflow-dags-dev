from __future__ import annotations

from airflow.utils.log.logging_mixin import LoggingMixin

from .config import load_pipeline_config
from .s3 import get_s3_client, parse_s3_uri
from .build import build_features as _build_features
from .store import store_features as _store_features

log = LoggingMixin().log


def task_extract_raw_data(**context):
    cfg = load_pipeline_config()
    ti = context["ti"]

    s3 = get_s3_client()
    bucket, key = parse_s3_uri(cfg.raw_path)
    s3.head_object(Bucket=bucket, Key=key)

    ti.xcom_push(key="dp_raw_path", value=cfg.raw_path)
    log.info("[DP] extract_raw_data OK raw=%s", cfg.raw_path)


def task_validate_data(**context):
    cfg = load_pipeline_config()
    ti = context["ti"]

    s3 = get_s3_client()
    bucket, key = parse_s3_uri(cfg.raw_path)
    obj = s3.get_object(Bucket=bucket, Key=key)
    text = obj["Body"].read().decode("utf-8")

    lines = [ln for ln in text.splitlines() if ln.strip()]
    if len(lines) <= 1:
        raise ValueError("[DP] validate failed: empty body")

    ti.xcom_push(key="dp_valid", value=True)
    ti.xcom_push(key="dp_lines", value=len(lines))
    log.info("[DP] validate OK lines=%d", len(lines))


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
    log.info(
        "[SUMMARY] pipeline=%s raw=%s out_prefix=%s version=%s uri=%s",
        cfg.pipeline_name,
        ti.xcom_pull(key="dp_raw_path", task_ids="extract_raw_data"),
        ti.xcom_pull(key="fs_prefix", task_ids="store_features"),
        ti.xcom_pull(key="fs_version", task_ids="store_features"),
        ti.xcom_pull(key="fs_feature_uri", task_ids="store_features"),
    )

