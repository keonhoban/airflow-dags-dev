# dags/pipelines/p_drift.py
from __future__ import annotations

from typing import Any

from mlops_lib.quality.drift_gate import drift_gate


def drift_gate_task(**context: Any) -> None:
    return drift_gate(**context)
