# dags/mlops_lib/dp/config.py
from __future__ import annotations

from dataclasses import dataclass
from airflow.sdk import Variable


@dataclass(frozen=True)
class PipelineConfig:
    pipeline_name: str
    raw_path: str
    feature_base: str
    feature_set: str

    schema_path: str = "/opt/airflow/feature-store/user_features.schema.json"
    metadata_tpl_path: str = "/opt/airflow/feature-store/metadata.json.j2"


def _get_var(key: str, default: str) -> str:
    try:
        return Variable.get(key)
    except Exception:
        return default


def load_pipeline_config() -> PipelineConfig:
    # ✅ 이미 건호님이 쓰던 “S3 전체 경로” 기반으로만 구성 (DP_BUCKET 같은 중복 설정 제거)
    raw_path = _get_var("dp_raw_path", "s3://datapipeline-raw-data-keonho/daily/user_events_20251119.csv")
    feature_base = _get_var("dp_feature_base", "s3://datapipeline-raw-data-keonho/feature-store")
    pipeline_name = _get_var("dp_pipeline_name", "daily_user_events")
    feature_set = _get_var("dp_feature_set", "user_features")

    return PipelineConfig(
        pipeline_name=pipeline_name,
        raw_path=raw_path,
        feature_base=feature_base,
        feature_set=feature_set,
    )

