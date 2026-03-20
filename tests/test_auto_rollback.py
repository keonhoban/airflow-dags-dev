# tests/test_auto_rollback.py
"""
auto_rollback.py 비즈니스 로직 단위 테스트.

AutoRollback.evaluate: 3단계 판정 로직 (UP → Error → Latency)
- 관측 불가 시 롤백 안 함 (오탐 방지)
- 임계값 초과 시만 롤백
"""
from __future__ import annotations

from typing import Optional
from unittest.mock import MagicMock

import pytest

from mlops_lib.observability.auto_rollback import (
    AutoRollback,
    ObservabilityTarget,
    RollbackThresholds,
)


def _make_prom(
    *,
    min_up: Optional[float] = 1.0,
    ratio: Optional[float] = 0.001,
    rps: Optional[float] = 0.1,
    p95: Optional[float] = 0.2,
) -> MagicMock:
    """PrometheusClient mock — query_scalar에 쿼리별 값을 매핑."""
    prom = MagicMock()
    prom.min_up_in_namespace.return_value = min_up

    def scalar_side_effect(query, default=None):
        q = str(query).lower()
        if "5xx" in q and "ratio" in q:
            return ratio
        if "5xx" in q:
            return rps
        if "latency" in q or "histogram" in q:
            return p95
        return default

    prom.query_scalar.side_effect = scalar_side_effect
    return prom


DEFAULT_TH = RollbackThresholds()
DEFAULT_TG = ObservabilityTarget(job="test-job", namespace="test-ns")


class TestAutoRollbackEvaluate:
    """AutoRollback.evaluate 3단계 판정 로직 검증."""

    def test_healthy_service_returns_ok(self):
        prom = _make_prom(min_up=1.0, ratio=0.001, rps=0.05, p95=0.1)
        ar = AutoRollback(prom=prom, thresholds=DEFAULT_TH, target=DEFAULT_TG)
        d = ar.evaluate()
        assert d.should_rollback is False
        assert d.reason == "OK"

    def test_up_metric_missing_returns_observability_unavailable(self):
        """관측 불가 시 롤백하지 않음 — 오탐 방지 철학."""
        prom = _make_prom(min_up=None)
        ar = AutoRollback(prom=prom, thresholds=DEFAULT_TH, target=DEFAULT_TG)
        d = ar.evaluate()
        assert d.should_rollback is False
        assert "OBSERVABILITY_UNAVAILABLE" in d.reason

    def test_pod_down_triggers_rollback(self):
        prom = _make_prom(min_up=0.0)
        ar = AutoRollback(prom=prom, thresholds=DEFAULT_TH, target=DEFAULT_TG)
        d = ar.evaluate()
        assert d.should_rollback is True
        assert "FASTAPI_UNHEALTHY" in d.reason

    def test_high_error_rate_and_rps_triggers_rollback(self):
        """에러율 AND RPS 모두 초과해야 롤백."""
        prom = _make_prom(ratio=0.05, rps=2.0)  # 둘 다 초과
        ar = AutoRollback(prom=prom, thresholds=DEFAULT_TH, target=DEFAULT_TG)
        d = ar.evaluate()
        assert d.should_rollback is True
        assert "ERROR_BUDGET_EXCEEDED" in d.reason

    def test_high_error_rate_but_low_rps_no_rollback(self):
        """에러율 높지만 RPS 낮으면 롤백 안 함 (트래픽 부족 = 노이즈)."""
        prom = _make_prom(ratio=0.05, rps=0.1)  # ratio만 초과
        ar = AutoRollback(prom=prom, thresholds=DEFAULT_TH, target=DEFAULT_TG)
        d = ar.evaluate()
        assert d.should_rollback is False

    def test_error_metrics_missing_returns_partial(self):
        prom = _make_prom(ratio=None, rps=None)
        ar = AutoRollback(prom=prom, thresholds=DEFAULT_TH, target=DEFAULT_TG)
        d = ar.evaluate()
        assert d.should_rollback is False
        assert "OBSERVABILITY_PARTIAL" in d.reason

    def test_high_latency_triggers_rollback(self):
        prom = _make_prom(p95=1.5)  # 0.8s 초과
        ar = AutoRollback(prom=prom, thresholds=DEFAULT_TH, target=DEFAULT_TG)
        d = ar.evaluate()
        assert d.should_rollback is True
        assert "LATENCY_TOO_HIGH" in d.reason

    def test_latency_metric_missing_returns_partial(self):
        prom = _make_prom(p95=None)
        ar = AutoRollback(prom=prom, thresholds=DEFAULT_TH, target=DEFAULT_TG)
        d = ar.evaluate()
        assert d.should_rollback is False
        assert "OBSERVABILITY_PARTIAL" in d.reason

    def test_custom_thresholds_respected(self):
        """커스텀 임계값이 적용되는지 검증."""
        strict_th = RollbackThresholds(max_p95_latency_sec=0.1)
        prom = _make_prom(p95=0.15)  # 기본(0.8)에선 OK, strict(0.1)에선 롤백
        ar = AutoRollback(prom=prom, thresholds=strict_th, target=DEFAULT_TG)
        d = ar.evaluate()
        assert d.should_rollback is True

    def test_signals_dict_contains_all_metrics(self):
        prom = _make_prom()
        ar = AutoRollback(prom=prom, thresholds=DEFAULT_TH, target=DEFAULT_TG)
        d = ar.evaluate()
        assert "min_up" in d.signals
        assert "5xx_ratio" in d.signals
        assert "5xx_rps" in d.signals
        assert "p95_latency_sec" in d.signals
        assert "thresholds" in d.signals
