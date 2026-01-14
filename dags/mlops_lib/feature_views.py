from datetime import timedelta
from feast import FeatureView, Field
from feast.types import Int64, Float64
from feast.infra.offline_stores.file_source import FileSource
from feast.data_format import ParquetFormat

from .entities import user

# Offline feature data location (S3 parquet)
user_features_source = FileSource(
    path="s3://datapipeline-raw-data-keonho/feast/offline/user_features/",
    timestamp_field="event_timestamp",
    created_timestamp_column="created_at",
    file_format=ParquetFormat(),
)

user_features_view = FeatureView(
    name="user_features",
    entities=[user],
    ttl=timedelta(days=30),
    schema=[
        Field(name="f_total_events_7d", dtype=Int64),
        Field(name="f_avg_session_sec_7d", dtype=Float64),
        Field(name="f_last_event_age_sec", dtype=Int64),
    ],
    source=user_features_source,
    online=True,
    tags={"owner": "keonho", "stage": "dev"},
)
