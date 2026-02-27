# dags/mlops_lib/observability/prometheus_client.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Optional

import requests


@dataclass(frozen=True)
class PrometheusConfig:
    base_url: str
    bearer_token: Optional[str] = None
    verify_tls: bool = True
    timeout_sec: float = 5.0


class PrometheusClient:
    """
    Minimal Prometheus HTTP API client
    - instant query: /api/v1/query
    - range query  : /api/v1/query_range
    """

    def __init__(self, cfg: PrometheusConfig):
        self.cfg = cfg

    def _headers(self) -> dict[str, str]:
        h = {"Accept": "application/json"}
        if self.cfg.bearer_token:
            h["Authorization"] = f"Bearer {self.cfg.bearer_token}"
        return h

    def query_instant(self, promql: str, ts: Optional[float] = None) -> dict[str, Any]:
        url = self.cfg.base_url.rstrip("/") + "/api/v1/query"
        params: dict[str, Any] = {"query": promql}
        if ts is not None:
            params["time"] = ts

        r = requests.get(
            url,
            params=params,
            headers=self._headers(),
            timeout=self.cfg.timeout_sec,
            verify=self.cfg.verify_tls,
        )
        r.raise_for_status()
        data = r.json()
        if data.get("status") != "success":
            raise RuntimeError(f"Prometheus query failed: {data}")
        return data

    def query_range(
        self,
        promql: str,
        start_ts: float,
        end_ts: float,
        step_sec: int = 15,
    ) -> dict[str, Any]:
        url = self.cfg.base_url.rstrip("/") + "/api/v1/query_range"
        params: dict[str, Any] = {
            "query": promql,
            "start": start_ts,
            "end": end_ts,
            "step": step_sec,
        }

        r = requests.get(
            url,
            params=params,
            headers=self._headers(),
            timeout=self.cfg.timeout_sec,
            verify=self.cfg.verify_tls,
        )
        r.raise_for_status()
        data = r.json()
        if data.get("status") != "success":
            raise RuntimeError(f"Prometheus range query failed: {data}")
        return data


def extract_scalar_float(resp: dict[str, Any]) -> Optional[float]:
    """
    Prometheus instant query result -> float
    returns None if empty
    """
    try:
        result = resp["data"]["result"]
        if not result:
            return None

        # result[0]["value"] = [ <ts>, "<value>" ]
        v = result[0]["value"][1]
        return float(v)
    except Exception:
        return None
