from datetime import timedelta
from feast import Entity, FeatureView, Field
from feast.infra.offline_stores.file_source import FileSource
from feast.types import Int64, Float64

user_features_source = FileSource(
    path="s3://datapipeline-raw-data-keonho/feature-store/user_features/latest/features.parquet",
    timestamp_field=None,
    created_timestamp_column=None,
)

user = Entity(name="user", join_keys=["user_id"])

user_features = FeatureView(
    name="user_features",
    entities=[user],
    ttl=timedelta(days=7),
    schema=[
        Field(name="f_total_events_7d", dtype=Int64),
        Field(name="f_avg_session_sec_7d", dtype=Float64),
        Field(name="f_last_event_age_sec", dtype=Int64),
    ],
    source=user_features_source,
    online=True,
)
