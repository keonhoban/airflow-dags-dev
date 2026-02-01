# dags/mlops_lib/dp/config.py
from __future__ import annotations

from dataclasses import dataclass

# Airflow 2.8+ 권장
try:
    from airflow.sdk import Variable
except Exception:  # fallback (구버전)
    from airflow.models import Variable


@dataclass(frozen=True)
class PipelineConfig:
    pipeline_name: str
    raw_path: str                 # s3://bucket/key.csv
    feature_base: str             # s3://bucket/feature-store
    feature_set: str              # user_features

    # 리소스 (GitOps/ConfigMap으로 주입되는 경로)
    schema_path: str = "/opt/airflow/feature-store/user_features.schema.json"
    metadata_tpl_path: str = "/opt/airflow/feature-store/metadata.json.j2"


def _get_var(key: str, default: str) -> str:
    """
    Airflow Variable 안전 조회:
    - 없으면 default
    """
    try:
        v = Variable.get(key)
        if v is None or str(v).strip() == "":
            return default
        return v
    except Exception:
        return default


def load_pipeline_config() -> PipelineConfig:
    """
    ✅ 제출용 최소 버전:
    - Variable만으로 충분히 동작
    - DP_BUCKET 같은 중복/필수 ENV 제거
    """
    raw_path = _get_var(
        "dp_raw_path",
        "s3://datapipeline-raw-data-keonho/daily/user_events_20251119.csv",
    )
    feature_base = _get_var(
        "dp_feature_base",
        "s3://datapipeline-raw-data-keonho/feature-store",
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

