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

    # feature-store-lite resources (ConfigMap mount)
    schema_path: str = "/opt/airflow/feature-store/user_features.schema.json"
    metadata_tpl_path: str = "/opt/airflow/feature-store/metadata.json.j2"


def load_pipeline_config() -> PipelineConfig:
    """
    Airflow Variable 기반 파이프라인 설정.
    실무에서는 여기만 바꿔서 환경/파이프라인 확장합니다.

    Variables:
      - dp_raw_path:      s3://<bucket>/<key>.csv
      - dp_feature_base:  s3://<bucket>/<prefix>/feature-store   (권장)
      - dp_pipeline_name: e.g., daily_user_events
      - dp_feature_set:   e.g., user_features
    """
    raw_path = Variable.get(
        "dp_raw_path",
        default_var="s3://datapipeline-raw-data-keonho/daily/user_events_20251119.csv",
    )
    feature_base = Variable.get(
        "dp_feature_base",
        default_var="s3://datapipeline-raw-data-keonho/features/feature-store",
    )
    pipeline_name = Variable.get(
        "dp_pipeline_name",
        default_var="daily_user_events",
    )
    feature_set = Variable.get(
        "dp_feature_set",
        default_var="user_features",
    )

    return PipelineConfig(
        pipeline_name=pipeline_name,
        raw_path=raw_path,
        feature_base=feature_base,
        feature_set=feature_set,
    )
