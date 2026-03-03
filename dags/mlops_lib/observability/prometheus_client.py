# dags/mlops_lib/observability/prometheus_client.py
from __future__ import annotations

import json
import os
import time
from dataclasses import dataclass
from typing import Any, Dict, Optional

import requests
from airflow.models import Variable

from mlops_lib.core.policy import (
    VAR_PROMETHEUS_BASE_URL,
    VAR_PROMETHEUS_BEARER_TOKEN,
    VAR_PROMETHEUS_VERIFY_TLS,
)

"""
Prometheus query client (SSOT)

목표:
- curl/jq 없이 Python에서 Prometheus HTTP API를 안정적으로 조회
- DAG/오토롤백에서 "관측 쿼리"를 표준화
- 네트워크/일시 오류에 강한 최소한의 재시도 제공

SSOT:
- Prometheus endpoint/token/verify_tls는 policy.py의 Variable key를 사용한다.
- 단, 기존 호환을 위해 legacy key("prometheus_url") / env("PROMETHEUS_URL")도 fallback으로 지원한다.
"""

# legacy (backward compatible)
VAR_PROMETHEUS_URL_LEGACY = "prometheus_url"


def _v(key: str, default: Optional[str] = None) -> Optional[str]:
    try:
        return Variable.get(key)
    except Exception:
        return default


def _to_bool(raw: Optional[str], default: bool) -> bool:
    if raw is None:
        return default
    s = str(raw).strip().lower()
    if s in ("1", "true", "yes", "y", "on"):
        return True
    if s in ("0", "false", "no", "n", "off"):
        return False
    return default


@dataclass(frozen=True)
class PrometheusConfig:
    base_url: str
    timeout_sec: float = 5.0
    retries: int = 2
    retry_backoff_sec: float = 0.6

    # auth / tls
    bearer_token: Optional[str] = None
    verify_tls: bool = True

    @classmethod
    def load(cls) -> "PrometheusConfig":
        # 1) SSOT (policy.py variable keys)
        base = (_v(VAR_PROMETHEUS_BASE_URL, None) or "").strip()
        token = (_v(VAR_PROMETHEUS_BEARER_TOKEN, None) or "").strip() or None
        verify_tls = _to_bool(_v(VAR_PROMETHEUS_VERIFY_TLS, None), True)

        # 2) legacy variable (backward compatible)
        if not base:
            base = (_v(VAR_PROMETHEUS_URL_LEGACY, None) or "").strip()

        # 3) env fallback
        if not base:
            base = (os.environ.get("PROMETHEUS_URL", "") or "").strip()

        # 4) default (port-forward friendly)
        if not base:
            base = "http://localhost:9090"

        return cls(
            base_url=base.rstrip("/"),
            bearer_token=token,
            verify_tls=verify_tls,
        )


class PrometheusError(RuntimeError):
    pass


class PrometheusClient:
    def __init__(self, config: Optional[PrometheusConfig] = None) -> None:
        self.cfg = config or PrometheusConfig.load()

    def _headers(self) -> Dict[str, str]:
        if not self.cfg.bearer_token:
            return {}
        return {"Authorization": f"Bearer {self.cfg.bearer_token}"}

    def _get(self, path: str, params: Dict[str, Any]) -> Dict[str, Any]:
        url = f"{self.cfg.base_url}{path}"
        last_err: Optional[Exception] = None

        for i in range(self.cfg.retries + 1):
            try:
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
                    raise PrometheusError(f"Prometheus API non-success: {json.dumps(data)[:400]}")
                return data
            except Exception as e:
                last_err = e
                if i < self.cfg.retries:
                    time.sleep(self.cfg.retry_backoff_sec * (i + 1))
                    continue
                raise PrometheusError(f"Prometheus request failed: url={url}, err={e}") from e

        raise PrometheusError(str(last_err))

    # ---------------------------
    # Public APIs
    # ---------------------------
    def query(self, promql: str, ts: Optional[float] = None) -> Dict[str, Any]:
        params: Dict[str, Any] = {"query": promql}
        if ts is not None:
            params["time"] = ts
        return self._get("/api/v1/query", params=params)

    def query_range(
        self,
        promql: str,
        *,
        start_ts: float,
        end_ts: float,
        step_sec: int = 15,
    ) -> Dict[str, Any]:
        params = {
            "query": promql,
            "start": start_ts,
            "end": end_ts,
            "step": step_sec,
        }
        return self._get("/api/v1/query_range", params=params)

    # ---------------------------
    # Result helpers
    # ---------------------------
    @staticmethod
    def _extract_vector_first_value(resp: Dict[str, Any]) -> Optional[float]:
        data = resp.get("data") or {}
        if data.get("resultType") != "vector":
            return None
        result = data.get("result") or []
        if not result:
            return None
        value = result[0].get("value")
        if not value or len(value) < 2:
            return None
        try:
            return float(value[1])
        except Exception:
            return None

    def query_scalar(self, promql: str, *, default: Optional[float] = None) -> Optional[float]:
        resp = self.query(promql)
        v = self._extract_vector_first_value(resp)
        return v if v is not None else default

    def is_series_present(self, promql: str) -> bool:
        resp = self.query(promql)
        data = resp.get("data") or {}
        result = data.get("result") or []
        return bool(result)

    def min_up_in_namespace(self, *, namespace: str, job: str) -> Optional[float]:
        q = f'min(up{{namespace="{namespace}",job="{job}"}})'
        return self.query_scalar(q, default=None)


# ---------------------------
# Convenience: build queries
# ---------------------------
def q_fastapi_5xx_ratio(*, job: str, namespace: str, window: str = "1m") -> str:
    return (
        f'sum(rate(http_requests_total{{job="{job}",namespace="{namespace}",status=~"5.."}}[{window}]))'
        f' / clamp_min(sum(rate(http_requests_total{{job="{job}",namespace="{namespace}"}}[{window}])), 1)'
    )


def q_fastapi_5xx_rps(*, job: str, namespace: str, window: str = "1m") -> str:
    return f'sum(rate(http_requests_total{{job="{job}",namespace="{namespace}",status=~"5.."}}[{window}]))'


def q_fastapi_p95_latency_seconds(*, job: str, namespace: str, window: str = "5m") -> str:
    return (
        f'histogram_quantile(0.95, '
        f'sum(rate(http_request_duration_highr_seconds_bucket{{job="{job}",namespace="{namespace}"}}[{window}])) by (le)'
        f')'
    )
