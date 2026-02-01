# dags/mlops_lib/dp/schema.py
from __future__ import annotations
from dataclasses import dataclass
from typing import List


@dataclass(frozen=True)
class FeatureSchema:
    name: str
    columns: List[str]


SCHEMA = FeatureSchema(
    name="user_features_v1",
    columns=[
        "f_total_events_7d",
        "f_avg_session_sec_7d",
        "f_last_event_age_sec",
    ],
)


def required_columns() -> list[str]:
    return SCHEMA.columns

