# dags/mlops_lib/dp/config.py
from __future__ import annotations

from dataclasses import dataclass
from airflow.models import Variable


@dataclass(frozen=True)
class PipelineConfig:
    pipeline_name: str
    raw_path: str
    feature_base: str
    feature_set: str

    schema_path: str = "/opt/airflow/feature-store/user_features.schema.json"
    metadata_tpl_path: str = "/opt/airflow/feature-store/metadata.json.j2"


def _get_var(key: str, default: str) -> str:
    """
    Airflow 버전/SDK 차이를 피하기 위한 안전 패턴:
    - Variable.get(key) -> 없으면 예외 -> default 반환
    """
    try:
        return Variable.get(key)
    except Exception:
        return default


def load_pipeline_config() -> PipelineConfig:
    raw_path = _get_var(
        "dp_raw_path",
        "s3://datapipeline-raw-data-keonho/daily/user_events_20251119.csv",
    )
    feature_base = _get_var(
        "dp_feature_base",
        "s3://datapipeline-raw-data-keonho/features/feature-store",
    )
    pipeline_name = _get_var(
        "dp_pipeline_name",
        "daily_user_events",
    )
    feature_set = _get_var(
        "dp_feature_set",
        "user_features",
    )

    return PipelineConfig(
        pipeline_name=pipeline_name,
        raw_path=raw_path,
        feature_base=feature_base,
        feature_set=feature_set,
    )
