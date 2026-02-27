# dags/mlops_lib/observability/prometheus_client.py
from __future__ import annotations

import json
import os
import time
from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple

import requests
from airflow.models import Variable


"""
Prometheus query client (SSOT)

목표:
- curl/jq 없이 Python에서 Prometheus HTTP API를 안정적으로 조회
- DAG/오토롤백에서 "관측 쿼리"를 표준화
- 네트워크/일시 오류에 강한 최소한의 재시도 제공

권장:
- Airflow Variable "prometheus_url" 로 URL 관리
  예) http://monitoring-dev-kube-promet-prometheus.monitoring-dev.svc:9090
- 로컬 포트포워드 환경이면 http://localhost:9090 도 가능
"""


VAR_PROMETHEUS_URL = "prometheus_url"


def _v(key: str, default: Optional[str] = None) -> Optional[str]:
    try:
        return Variable.get(key)
    except Exception:
        return default


def _now_ts() -> float:
    return time.time()


@dataclass(frozen=True)
class PrometheusConfig:
    base_url: str
    timeout_sec: float = 5.0
    retries: int = 2
    retry_backoff_sec: float = 0.6

    @classmethod
    def load(cls) -> "PrometheusConfig":
        # 1) Airflow Variable
        v = _v(VAR_PROMETHEUS_URL, None)
        # 2) env
        e = os.environ.get("PROMETHEUS_URL")
        # 3) default (port-forward friendly)
        base = (v or e or "http://localhost:9090").strip().rstrip("/")
        return cls(base_url=base)


class PrometheusError(RuntimeError):
    pass


class PrometheusClient:
    def __init__(self, config: Optional[PrometheusConfig] = None) -> None:
        self.cfg = config or PrometheusConfig.load()

    def _get(self, path: str, params: Dict[str, Any]) -> Dict[str, Any]:
        url = f"{self.cfg.base_url}{path}"
        last_err: Optional[Exception] = None

        for i in range(self.cfg.retries + 1):
            try:
                r = requests.get(url, params=params, timeout=self.cfg.timeout_sec)
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

        # unreachable
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
        """
        For instant query:
        resp["data"]["resultType"] == "vector"
        resp["data"]["result"] = [ {"value":[ts,"123.4"]}, ... ]
        """
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
        """
        PromQL 결과가 vector(단일 스칼라 값)로 나오도록 작성한 query용.
        예) sum(rate(...))
        """
        resp = self.query(promql)
        v = self._extract_vector_first_value(resp)
        return v if v is not None else default

    def is_series_present(self, promql: str) -> bool:
        """
        값이 하나라도 나오면 True
        """
        resp = self.query(promql)
        data = resp.get("data") or {}
        result = data.get("result") or []
        return bool(result)

    def min_up_in_namespace(self, *, namespace: str, job: str) -> Optional[float]:
        """
        fastapi-dev 네임스페이스에서 해당 job의 up 최소값.
        - 2 pods면 up 2개가 나오는데, 그 중 하나라도 0이면 min이 0
        """
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
