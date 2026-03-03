# dags/mlops_lib/observability/auto_rollback.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional

from airflow.models import Variable
from airflow.utils.log.logging_mixin import LoggingMixin

from utils.slack_alerts import notify_info, notify_skip, notify_success

from mlops_lib.core.policy import (
    VAR_ERROR_RATE_THRESHOLD,
    VAR_LATENCY_P95_THRESHOLD_SEC,
)

from mlops_lib.observability.prometheus_client import (
    PrometheusClient,
    q_fastapi_5xx_ratio,
    q_fastapi_5xx_rps,
    q_fastapi_p95_latency_seconds,
)

log = LoggingMixin().log


"""
Auto rollback policy (SSOT)

목표:
- "배포 후 관측 기반 판정"을 DAG에서 재사용 가능한 모듈로 고정
- metrics 없거나(스크랩 실패) 라벨 불일치 같은 경우:
  - 잘못된 롤백(오탐) 대신 "판정 보류"를 기본값으로 둠 (운영 안전)

철학:
- 롤백은 비용이 큰 행위 → '확실한 신호'에서만 실행
- 관측 실패는 롤백 트리거가 아니라 "관측 인프라 문제"로 분류

✅ 이번 수정의 핵심:
- AutoRollback이 Airflow Variable(SSOT: mlops_lib.core.policy) 값을 실제로 읽어서
  observe_error_rate_threshold / observe_latency_p95_threshold_sec를
  롤백 임계값으로 반영한다.
"""


def _v(key: str, default: Optional[str] = None) -> Optional[str]:
    try:
        return Variable.get(key)
    except Exception:
        return default


def _to_float(raw: Optional[str], default: float) -> float:
    try:
        return float(str(raw))
    except Exception:
        return default


@dataclass(frozen=True)
class RollbackThresholds:
    """
    관측 기준(팀 정책값) - 제출/운영용으로 깔끔한 숫자 예시를 기본 제공
    필요 시 Airflow Variable로 외부화해도 됨.
    """
    # up (pod health)
    min_up_required: float = 1.0  # min(up) must be 1.0

    # errors
    max_5xx_ratio: float = 0.02   # 2%
    max_5xx_rps: float = 1.0      # 1 req/sec

    # latency
    max_p95_latency_sec: float = 0.8  # 800ms

    # query windows
    win_err: str = "1m"
    win_latency: str = "5m"


def _thresholds_from_airflow_variables(defaults: RollbackThresholds) -> RollbackThresholds:
    """
    ✅ SSOT 연결:
    - observe_error_rate_threshold -> max_5xx_ratio
    - observe_latency_p95_threshold_sec -> max_p95_latency_sec

    나머지(max_5xx_rps, window 등)는 "제출용 기본값"을 유지.
    (원하면 동일 패턴으로 Variable 외부화 확장 가능)
    """
    max_ratio = _to_float(_v(VAR_ERROR_RATE_THRESHOLD, str(defaults.max_5xx_ratio)), defaults.max_5xx_ratio)
    max_p95 = _to_float(_v(VAR_LATENCY_P95_THRESHOLD_SEC, str(defaults.max_p95_latency_sec)), defaults.max_p95_latency_sec)

    return RollbackThresholds(
        min_up_required=defaults.min_up_required,
        max_5xx_ratio=max_ratio,
        max_5xx_rps=defaults.max_5xx_rps,
        max_p95_latency_sec=max_p95,
        win_err=defaults.win_err,
        win_latency=defaults.win_latency,
    )


@dataclass(frozen=True)
class ObservabilityTarget:
    """
    Prometheus label selectors (SSOT)
    - 지금 건호님 환경에서 확정된 값 기준
    """
    job: str
    namespace: str


@dataclass(frozen=True)
class RollbackDecision:
    should_rollback: bool
    reason: str
    signals: Dict[str, Any]


class AutoRollback:
    def __init__(
        self,
        *,
        prom: Optional[PrometheusClient] = None,
        thresholds: Optional[RollbackThresholds] = None,
        target: Optional[ObservabilityTarget] = None,
    ) -> None:
        self.prom = prom or PrometheusClient()

        defaults = RollbackThresholds()
        # ✅ 외부에서 thresholds 주입하면 그걸 우선, 아니면 Variable 기반으로 로드
        self.th = thresholds or _thresholds_from_airflow_variables(defaults)

        if target is None:
            # ✅ 건호님 실측 기반 "정답 라벨" 기본값
            target = ObservabilityTarget(job="fastapi-dev-service", namespace="fastapi-dev")
        self.tg = target

    # ---------------------------
    # Core evaluation
    # ---------------------------
    def evaluate(self) -> RollbackDecision:
        """
        관측 기반 판단:
        1) up(min) < 1 -> rollback
        2) 5xx_ratio > threshold AND (5xx_rps > threshold) -> rollback (오탐 방지)
        3) p95 latency > threshold -> rollback (지속 느림)
        관측값이 없으면 "판정 보류" (rollback False) + reason에 남김
        """
        signals: Dict[str, Any] = {
            "job": self.tg.job,
            "namespace": self.tg.namespace,
            "thresholds": {
                "min_up_required": self.th.min_up_required,
                "max_5xx_ratio": self.th.max_5xx_ratio,
                "max_5xx_rps": self.th.max_5xx_rps,
                "max_p95_latency_sec": self.th.max_p95_latency_sec,
                "win_err": self.th.win_err,
                "win_latency": self.th.win_latency,
            },
        }

        # ---- 1) UP check
        min_up = self.prom.min_up_in_namespace(namespace=self.tg.namespace, job=self.tg.job)
        signals["min_up"] = min_up

        if min_up is None:
            # 관측 불가 = 롤백 근거로 쓰지 않음(오탐 방지)
            return RollbackDecision(
                should_rollback=False,
                reason="OBSERVABILITY_UNAVAILABLE: up metric missing",
                signals=signals,
            )

        if float(min_up) < self.th.min_up_required:
            return RollbackDecision(
                should_rollback=True,
                reason=f"FASTAPI_UNHEALTHY: min(up)={min_up} < {self.th.min_up_required}",
                signals=signals,
            )

        # ---- 2) Error ratio & RPS (둘 다 넘어야 롤백)
        q_ratio = q_fastapi_5xx_ratio(job=self.tg.job, namespace=self.tg.namespace, window=self.th.win_err)
        q_rps = q_fastapi_5xx_rps(job=self.tg.job, namespace=self.tg.namespace, window=self.th.win_err)

        ratio = self.prom.query_scalar(q_ratio, default=None)
        rps = self.prom.query_scalar(q_rps, default=None)
        signals["5xx_ratio"] = ratio
        signals["5xx_rps"] = rps

        if ratio is None or rps is None:
            # 일부 신호만 없을 때도 롤백 금지(오탐 방지)
            return RollbackDecision(
                should_rollback=False,
                reason="OBSERVABILITY_PARTIAL: 5xx metrics missing",
                signals=signals,
            )

        if float(ratio) > self.th.max_5xx_ratio and float(rps) > self.th.max_5xx_rps:
            return RollbackDecision(
                should_rollback=True,
                reason=(
                    f"ERROR_BUDGET_EXCEEDED: 5xx_ratio={float(ratio):.6f} (> {self.th.max_5xx_ratio}), "
                    f"5xx_rps={float(rps):.3f} (> {self.th.max_5xx_rps})"
                ),
                signals=signals,
            )

        # ---- 3) Latency p95
        q_p95 = q_fastapi_p95_latency_seconds(job=self.tg.job, namespace=self.tg.namespace, window=self.th.win_latency)
        p95 = self.prom.query_scalar(q_p95, default=None)
        signals["p95_latency_sec"] = p95

        if p95 is None:
            return RollbackDecision(
                should_rollback=False,
                reason="OBSERVABILITY_PARTIAL: latency metric missing",
                signals=signals,
            )

        if float(p95) > self.th.max_p95_latency_sec:
            return RollbackDecision(
                should_rollback=True,
                reason=f"LATENCY_TOO_HIGH: p95={float(p95):.6f}s (> {self.th.max_p95_latency_sec}s)",
                signals=signals,
            )

        return RollbackDecision(
            should_rollback=False,
            reason="OK",
            signals=signals,
        )

    # ---------------------------
    # Airflow-friendly callables
    # ---------------------------
    def task_decide(self, **context: Any) -> bool:
        """
        Airflow PythonOperator에서 호출:
        - True면 'rollback 경로'로 branch/trigger 할 수 있음
        - False면 정상 진행
        """
        d = self.evaluate()

        # Slack 메시지 표준화
        if d.should_rollback:
            notify_info(
                "AutoRollback: TRIGGERED",
                reason=d.reason,
                job=self.tg.job,
                namespace=self.tg.namespace,
                min_up=str(d.signals.get("min_up")),
                ratio=str(d.signals.get("5xx_ratio")),
                rps=str(d.signals.get("5xx_rps")),
                p95=str(d.signals.get("p95_latency_sec")),
                thresholds=str(d.signals.get("thresholds")),
            )
            return True

        if d.reason.startswith("OBSERVABILITY_"):
            notify_skip(
                "AutoRollback: SKIPPED",
                reason=d.reason,
                next_action="Prometheus scrape/ServiceMonitor 라벨/네트워크 확인",
                job=self.tg.job,
                namespace=self.tg.namespace,
            )
            return False

        notify_success(
            "AutoRollback: OK",
            job=self.tg.job,
            namespace=self.tg.namespace,
            min_up=str(d.signals.get("min_up")),
            ratio=str(d.signals.get("5xx_ratio")),
            rps=str(d.signals.get("5xx_rps")),
            p95=str(d.signals.get("p95_latency_sec")),
            thresholds=str(d.signals.get("thresholds")),
        )
        return False
