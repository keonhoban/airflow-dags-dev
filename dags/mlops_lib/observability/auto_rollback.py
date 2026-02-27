# dags/mlops_lib/observability/auto_rollback.py
from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Optional

from airflow.models import Variable
from airflow.utils.log.logging_mixin import LoggingMixin

from mlops_lib.core.policy import (
    VAR_PROMETHEUS_BASE_URL,
    VAR_PROMETHEUS_BEARER_TOKEN,
    VAR_PROMETHEUS_VERIFY_TLS,
    VAR_OBSERVE_WINDOW_SEC,
    VAR_OBSERVE_STEP_SEC,
    VAR_OBSERVE_POKE_INTERVAL_SEC,
    VAR_ERROR_RATE_THRESHOLD,
    VAR_LATENCY_P95_THRESHOLD_SEC,
    VAR_PROMQL_ERROR_RATE,
    VAR_PROMQL_LATENCY_P95,
)
from utils.slack_alerts import notify_info, notify_skip

from mlops_lib.observability.prometheus_client import (
    PrometheusClient,
    PrometheusConfig,
    extract_scalar_float,
)

log = LoggingMixin().log


@dataclass(frozen=True)
class ObservePolicy:
    window_sec: int
    step_sec: int
    poke_interval_sec: int

    error_rate_threshold: float          # e.g. 0.02 (2%)
    latency_p95_threshold_sec: float     # e.g. 0.8 (800ms)

    promql_error_rate: str
    promql_latency_p95: str


def _v(key: str, default: Optional[str] = None) -> Optional[str]:
    try:
        return Variable.get(key)
    except Exception:
        return default


def _to_int(raw: Optional[str], default: int) -> int:
    try:
        return int(str(raw))
    except Exception:
        return default


def _to_float(raw: Optional[str], default: float) -> float:
    try:
        return float(str(raw))
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


def load_observe_policy(env: str) -> ObservePolicy:
    """
    Everything configurable via Airflow Variables (SSOT).
    env placeholder 지원: {env}
    """

    # ---- observation loop ----
    window_sec = _to_int(_v(VAR_OBSERVE_WINDOW_SEC, "180"), 180)              # default 3 min
    step_sec = _to_int(_v(VAR_OBSERVE_STEP_SEC, "15"), 15)                   # prometheus step
    poke_interval_sec = _to_int(_v(VAR_OBSERVE_POKE_INTERVAL_SEC, "20"), 20) # sensor/loop interval

    # ---- thresholds ----
    err_th = _to_float(_v(VAR_ERROR_RATE_THRESHOLD, "0.02"), 0.02)           # 2%
    lat_th = _to_float(_v(VAR_LATENCY_P95_THRESHOLD_SEC, "0.8"), 0.8)        # 800ms

    # ---- promql (override strongly recommended if your metric names differ) ----
    # 기본값은 fastapi-instrumentator 계열에서 자주 보이는 네이밍 기준
    default_err = (
        'sum(rate(http_requests_total{job=~"fastapi.*",status=~"5..",env="{env}"}[1m])) '
        '/ clamp_min(sum(rate(http_requests_total{job=~"fastapi.*",env="{env}"}[1m])), 1)'
    )
    default_lat = (
        'histogram_quantile(0.95, sum(rate(http_request_duration_seconds_bucket{job=~"fastapi.*",env="{env}"}[1m])) by (le))'
    )

    promql_err = (_v(VAR_PROMQL_ERROR_RATE, default_err) or default_err).format(env=env)
    promql_lat = (_v(VAR_PROMQL_LATENCY_P95, default_lat) or default_lat).format(env=env)

    return ObservePolicy(
        window_sec=window_sec,
        step_sec=step_sec,
        poke_interval_sec=poke_interval_sec,
        error_rate_threshold=err_th,
        latency_p95_threshold_sec=lat_th,
        promql_error_rate=promql_err,
        promql_latency_p95=promql_lat,
    )


def _prom_client() -> PrometheusClient:
    base_url = _v(VAR_PROMETHEUS_BASE_URL, "http://prometheus-operated.monitoring.svc.cluster.local:9090") \
        or "http://prometheus-operated.monitoring.svc.cluster.local:9090"

    bearer = _v(VAR_PROMETHEUS_BEARER_TOKEN, None)
    verify_tls = _to_bool(_v(VAR_PROMETHEUS_VERIFY_TLS, "true"), True)

    cfg = PrometheusConfig(
        base_url=base_url,
        bearer_token=bearer,
        verify_tls=verify_tls,
        timeout_sec=5.0,
    )
    return PrometheusClient(cfg)


def observe_and_fail_if_bad(*, env: str, start_ts: Optional[float] = None) -> None:
    """
    관측 window 동안 error_rate / latency_p95를 평가합니다.
    - window 종료 시점까지 "정상"이면 return
    - 초과가 관측되면 Exception 발생 (=> Airflow task 실패 => rollback 트리거)

    start_ts:
    - None이면 "지금"을 관측 시작으로 간주
    """
    pol = load_observe_policy(env=env)
    pc = _prom_client()

    if start_ts is None:
        start_ts = time.time()

    end_ts = start_ts + pol.window_sec

    notify_info(
        "Observe post-deploy metrics (start)",
        env=env,
        window_sec=str(pol.window_sec),
        err_th=str(pol.error_rate_threshold),
        lat_p95_th=str(pol.latency_p95_threshold_sec),
    )

    # loop until window ends
    while True:
        now = time.time()
        if now >= end_ts:
            notify_info("Observe post-deploy metrics (pass)", env=env, result="healthy")
            return

        # instant query (cheap)
        err = extract_scalar_float(pc.query_instant(pol.promql_error_rate))
        lat = extract_scalar_float(pc.query_instant(pol.promql_latency_p95))

        # if metric not found, treat as "cannot judge" but do NOT fail hard
        # (제출/면접 안전장치: 프로메테우스 라벨/메트릭 미스매치로 롤백이 터지지 않게)
        if err is None and lat is None:
            log.warning("[observe] metrics empty (check promql/labels). will keep observing.")
        else:
            log.info(f"[observe] env={env} err={err} lat_p95={lat}")

        err_bad = (err is not None) and (err > pol.error_rate_threshold)
        lat_bad = (lat is not None) and (lat > pol.latency_p95_threshold_sec)

        if err_bad or lat_bad:
            notify_skip(
                "Auto rollback triggered by metrics",
                env=env,
                reason=f"err_bad={err_bad},lat_bad={lat_bad}",
                err=str(err),
                lat_p95=str(lat),
            )
            raise RuntimeError(f"AutoRollback: metrics exceeded (err={err}, lat_p95={lat})")

        time.sleep(pol.poke_interval_sec)
