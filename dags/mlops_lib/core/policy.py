# dags/mlops_lib/core/policy.py
from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Any, Optional

# NOTE:
# Airflow 2.10+ / 3.x 계열에서 `airflow.models.Variable` 경고가 뜰 수 있음.
# 현재는 동작에 문제 없고, 추후 `from airflow.sdk import Variable`로 이관하면 됨.
from airflow.models import Variable

"""
Policy SSOT (Core)

- DAG 정책값(재시도/timeout/센서 모드 등)
- Triton timeouts
- Runtime Settings (Airflow Variable 기반)
- Drift Gate 정책(Pre-deploy quality gate)
- Observability 정책(자동 롤백용 Prometheus settings / thresholds / selectors)

목표:
- DAG/파이프라인 코드에서 숫자/문자열 하드코딩 제거
- 운영/면접에서 "정책은 한 곳에서 관리"를 증명

원칙:
- core(policy)는 외부 시스템(utils/slack, observability client 등)에 의존하지 않는다.
"""

# ============================================================
# Airflow DAG policies (SSOT)
# ============================================================

E2E_START_DATE_YMD = (2025, 1, 1)
E2E_RETRIES = 1
E2E_RETRY_DELAY_MIN = 2
E2E_MAX_ACTIVE_RUNS = 1
E2E_DAGRUN_TIMEOUT_MIN = 30
# SLA: 트리거 후 이 시간 안에 DAG가 완료되지 않으면 sla_miss_callback 호출.
# dagrun_timeout(30분)보다 크게 설정해야 의미가 있음.
E2E_SLA_HOUR = 1

MODEL_READY_POKE_INTERVAL_SEC = 10
MODEL_READY_TIMEOUT_SEC = 180
MODEL_READY_MODE = "reschedule"

# ✅ 정책: FastAPI reload 실패는 자동 롤백하지 않음
# (model repo SSOT(current.json / config.pbtxt)를 되돌리는 건 위험)
ROLLBACK_ON_FASTAPI_RELOAD_FAILURE = False

# ============================================================
# Triton timeouts (SSOT)
# ============================================================

T_TRITON_UNLOAD = 10
T_TRITON_LOAD = 10
T_TRITON_READY = 5
T_TRITON_INFER = 10

# ============================================================
# Triton 최적화 설정 (SSOT)
#
# dynamic_batching:
#   Triton이 여러 요청을 묶어 한 번에 inference하는 서버사이드 배칭.
#   max_batch_size > 0 이어야 활성화된다.
#   preferred_batch_size: 묶을 요청 수 후보 (ex. [8, 16])
#   max_queue_delay_microseconds: batch 완성을 기다릴 최대 시간 (µs)
#
# instance_group:
#   모델 인스턴스를 GPU 또는 CPU에 몇 개 배치할지 설정.
#   기본값은 KIND_GPU 1개. CPU 전용 환경은 KIND_CPU로 변경.
# ============================================================

VAR_TRITON_DYNAMIC_BATCHING_ENABLED = "triton_dynamic_batching_enabled"   # default "false"
VAR_TRITON_PREFERRED_BATCH_SIZES    = "triton_preferred_batch_sizes"       # default "8,16"
VAR_TRITON_MAX_QUEUE_DELAY_US       = "triton_max_queue_delay_us"          # default "5000"
VAR_TRITON_MAX_BATCH_SIZE           = "triton_max_batch_size"              # default "32"
VAR_TRITON_INSTANCE_GROUP_ENABLED   = "triton_instance_group_enabled"      # default "false"
VAR_TRITON_INSTANCE_GROUP_KIND      = "triton_instance_group_kind"         # default "KIND_GPU"
VAR_TRITON_INSTANCE_GROUP_COUNT     = "triton_instance_group_count"        # default "1"

# ============================================================
# Airflow Variable keys (SSOT)  - 오타 방지용
# ============================================================

VAR_TRITON_ENV = "triton_env"
VAR_ACCURACY_THRESHOLD = "accuracy_threshold"
VAR_LOGREG_C = "logreg_C"
VAR_LOGREG_MAX_ITER = "logreg_max_iter"
VAR_TRITON_MODEL_NAME = "triton_model_name"
VAR_MODEL_NAME = "model_name"
VAR_MLFLOW_ALIAS = "mlflow_alias"
VAR_CODE_VERSION = "code_version"

# ============================================================
# Drift Gate (SSOT)
# ============================================================
# 최소 스택: KS-stat(D) 기반 gate (p-value 없이도 재현 가능)
VAR_DRIFT_KS_STAT_THRESHOLD = "drift_ks_stat_threshold"  # default 0.20
VAR_DRIFT_SAMPLE_N = "drift_sample_n"  # default 2000
VAR_DRIFT_MAX_COLS = "drift_max_columns"  # default 20

# ============================================================
# Data Pipeline (SSOT)
# ============================================================
VAR_DP_MIN_ROWS = "dp_min_rows"  # default 50 — 학습 의미가 있는 최소 피처 행 수

# ============================================================
# Observability / Auto Rollback (SSOT)
# ============================================================

# Prometheus endpoint
VAR_PROMETHEUS_BASE_URL = "prometheus_base_url"  # e.g. http://prometheus-operated.monitoring.svc.cluster.local:9090
VAR_PROMETHEUS_BEARER_TOKEN = "prometheus_bearer_token"  # optional
VAR_PROMETHEUS_VERIFY_TLS = "prometheus_verify_tls"  # true/false

# Observe window params
VAR_OBSERVE_WINDOW_SEC = "observe_window_sec"  # default 180
VAR_OBSERVE_STEP_SEC = "observe_step_sec"  # default 15
VAR_OBSERVE_POKE_INTERVAL_SEC = "observe_poke_interval_sec"  # default 20

# Thresholds
VAR_ERROR_RATE_THRESHOLD = "observe_error_rate_threshold"  # default 0.02 (2%)
VAR_LATENCY_P95_THRESHOLD_SEC = "observe_latency_p95_threshold_sec"  # default 0.8 (sec)

# Target label selectors (job/namespace) — ✅ dev/prod 하드코딩 제거용
VAR_OBSERVE_JOB = "observe_job"  # e.g. fastapi-dev-service / fastapi-prod-service
VAR_OBSERVE_NAMESPACE = "observe_namespace"  # e.g. fastapi-dev / fastapi-prod

# PromQL overrides (metric/label 다르면 이걸로 바꾸면 됨)
VAR_PROMQL_ERROR_RATE = "promql_error_rate"
VAR_PROMQL_LATENCY_P95 = "promql_latency_p95"


def _non_empty(v: Any) -> Optional[str]:
    """빈 문자열과 None을 동일하게 처리. str.strip() 후 비어있으면 None 반환."""
    if v is None:
        return None
    s = str(v).strip()
    return s if s else None


def get_var(key: str, default: Optional[str] = None, *, required: bool = False) -> Optional[str]:
    """
    SSOT 설정 조회 — policy.py와 ml_code/config.py가 공유하는 단일 구현.

    우선순위:
      1) 환경변수 (os.getenv)
      2) Airflow Variable
      3) default

    Args:
        key      : 환경변수명 / Airflow Variable key (동일 값 사용)
        default  : 위 1-2가 모두 없을 때 반환할 값
        required : True면 최종적으로 값이 없을 때 RuntimeError 발생
    """
    v = _non_empty(os.getenv(key))
    if v is not None:
        return v

    try:
        raw = Variable.get(key, default_var=str(default) if default is not None else None)
        v = _non_empty(raw)
        if v is not None:
            return v
    except Exception:
        pass

    if required:
        raise RuntimeError(f"[Config] missing required key: {key!r}")
    return default


# 파일 내부 호출용 alias — 기존 _v() 호출부를 일괄 교체 없이 유지
_v = get_var


def _to_float(raw: Optional[str], default: float) -> float:
    try:
        return float(str(raw))
    except Exception:
        return default


def _to_int(raw: Optional[str], default: int) -> int:
    try:
        return int(str(raw))
    except Exception:
        return default


def _to_bool(raw: Optional[str], default: bool) -> bool:
    """
    Airflow Variable의 문자열을 bool로 변환 (SSOT)
    """
    if raw is None:
        return default
    s = str(raw).strip().lower()
    if s in ("1", "true", "yes", "y", "on"):
        return True
    if s in ("0", "false", "no", "n", "off"):
        return False
    return default


@dataclass(frozen=True)
class Settings:
    """
    Runtime Settings (Variable 기반)

    - 이 값들은 "DAG 실행 중"에도 Variable로 조정 가능
    - 하지만 파이프라인 안정성을 위해, train 시작 시점에 XCom으로 고정해두는 전략을 권장
    """

    env: str
    accuracy_threshold: float
    logreg_c: float
    logreg_max_iter: int
    model_name: str
    alias: str
    code_version: Optional[str]

    @classmethod
    def load(cls) -> "Settings":
        env = (_v(VAR_TRITON_ENV, "dev") or "dev").strip()
        th = _to_float(_v(VAR_ACCURACY_THRESHOLD, "0.60"), 0.60)

        c = _to_float(_v(VAR_LOGREG_C, "1.0"), 1.0)
        it = _to_int(_v(VAR_LOGREG_MAX_ITER, "200"), 200)

        model_name = (_v(VAR_TRITON_MODEL_NAME, _v(VAR_MODEL_NAME, "best_model")) or "best_model").strip()
        alias = (_v(VAR_MLFLOW_ALIAS, "A") or "A").strip()

        # 제출/운영용: git sha 같은 버전값을 Variable로 주입 가능(없으면 None)
        code_version = (_v(VAR_CODE_VERSION, None) or None)
        if code_version is not None:
            code_version = str(code_version).strip() or None

        return cls(
            env=env,
            accuracy_threshold=th,
            logreg_c=c,
            logreg_max_iter=it,
            model_name=model_name,
            alias=alias,
            code_version=code_version,
        )


@dataclass(frozen=True)
class DriftSettings:
    ks_stat_threshold: float
    sample_n: int
    max_columns: int

    @classmethod
    def load(cls) -> "DriftSettings":
        ks_th = _to_float(_v(VAR_DRIFT_KS_STAT_THRESHOLD, "0.20"), 0.20)
        n = _to_int(_v(VAR_DRIFT_SAMPLE_N, "2000"), 2000)
        mc = _to_int(_v(VAR_DRIFT_MAX_COLS, "20"), 20)
        return cls(ks_stat_threshold=ks_th, sample_n=n, max_columns=mc)


def drift_settings() -> DriftSettings:
    return DriftSettings.load()


@dataclass(frozen=True)
class TritonOptConfig:
    """
    Triton 최적화 설정 (Variable 기반 런타임 제어).

    dynamic_batching_enabled:
        True면 config.pbtxt에 dynamic_batching 블록이 추가된다.
        False(기본)면 블록 없이 단순 요청 처리.

    instance_group_enabled:
        True면 config.pbtxt에 instance_group 블록이 추가된다.
        GPU 환경에서는 True + KIND_GPU 권장.
        CPU 전용 환경에서는 KIND_CPU로 변경.

    max_batch_size:
        dynamic_batching 활성화 시 max_batch_size에 이 값이 들어간다.
        비활성화 시에는 0 (Triton의 unbatched 모드).
    """

    dynamic_batching_enabled: bool
    preferred_batch_sizes: list
    max_queue_delay_us: int
    max_batch_size: int

    instance_group_enabled: bool
    instance_group_kind: str
    instance_group_count: int

    @classmethod
    def load(cls) -> "TritonOptConfig":
        dyn_enabled = _to_bool(_v(VAR_TRITON_DYNAMIC_BATCHING_ENABLED, "false"), False)

        raw_sizes = (_v(VAR_TRITON_PREFERRED_BATCH_SIZES, "8,16") or "8,16").strip()
        preferred = [int(s.strip()) for s in raw_sizes.split(",") if s.strip().isdigit()]
        if not preferred:
            preferred = [8, 16]

        delay_us = _to_int(_v(VAR_TRITON_MAX_QUEUE_DELAY_US, "5000"), 5000)
        max_bs   = _to_int(_v(VAR_TRITON_MAX_BATCH_SIZE, "32"), 32)

        ig_enabled = _to_bool(_v(VAR_TRITON_INSTANCE_GROUP_ENABLED, "false"), False)
        ig_kind    = (_v(VAR_TRITON_INSTANCE_GROUP_KIND, "KIND_GPU") or "KIND_GPU").strip()
        ig_count   = _to_int(_v(VAR_TRITON_INSTANCE_GROUP_COUNT, "1"), 1)

        return cls(
            dynamic_batching_enabled=dyn_enabled,
            preferred_batch_sizes=preferred,
            max_queue_delay_us=delay_us,
            max_batch_size=max_bs,
            instance_group_enabled=ig_enabled,
            instance_group_kind=ig_kind,
            instance_group_count=ig_count,
        )


def triton_opt_config() -> TritonOptConfig:
    return TritonOptConfig.load()
