# dags/mlops_lib/core/policy.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

# NOTE:
# Airflow 2.10+ / 3.x 계열에서 `airflow.models.Variable` 경고가 뜰 수 있음.
# 현재는 동작에 문제 없고, 추후 `from airflow.sdk import Variable`로 이관하면 됨.
from airflow.models import Variable

from utils.slack_alerts import notify_info, notify_skip, notify_success
from mlops_lib.core.ids import (
    SHADOW_REASON_TRAIN_SKIPPED,
    SHADOW_REASON_ACCURACY_INVALID,
    SHADOW_REASON_BELOW_THRESHOLD,
    SHADOW_REASON_DRIFT_DETECTED,
)

"""
Policy SSOT

- DAG 정책값(재시도/timeout/센서 모드 등)
- Triton timeouts
- Runtime Settings (Airflow Variable 기반)
- Drift Gate 정책(Pre-deploy quality gate)
- Slack notify wrapper (표준 메시지 포맷)
- Observability 정책(자동 롤백용 Prometheus settings / thresholds / promql)

목표:
- DAG/파이프라인 코드에서 숫자/문자열 하드코딩 제거
- 운영/면접에서 "정책은 한 곳에서 관리"를 증명
"""

# ============================================================
# Airflow DAG policies (SSOT)
# ============================================================

E2E_START_DATE_YMD = (2025, 1, 1)
E2E_RETRIES = 1
E2E_RETRY_DELAY_MIN = 2
E2E_MAX_ACTIVE_RUNS = 1
E2E_DAGRUN_TIMEOUT_MIN = 30

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

# PromQL overrides (metric/label 다르면 이걸로 바꾸면 됨)
VAR_PROMQL_ERROR_RATE = "promql_error_rate"
VAR_PROMQL_LATENCY_P95 = "promql_latency_p95"


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
      (현재 pipelines/full_e2e.py에서 alias/model_name을 XCom에 push하고 있음)
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


# ============================================================
# Slack notify helpers (SSOT)
# ============================================================

def notify_train_completed(
    *,
    env: str,
    accuracy: float,
    alias: str,
    run_id: str,
    fs_version: str,
    schema_hash: str,
    code_version: str = "",
) -> None:
    notify_info(
        "Train completed",
        env=env,
        accuracy=f"{float(accuracy):.4f}",
        alias=alias,
        run_id=run_id,
        fs_version=fs_version,
        schema_hash=schema_hash,
        code_version=code_version,
    )


def notify_branch_promotion(*, env: str, accuracy: float, threshold: float) -> None:
    notify_info(
        "Branch: promotion",
        env=env,
        accuracy=f"{float(accuracy):.4f}",
        threshold=str(threshold),
    )


def notify_branch_shadow(
    *,
    env: str,
    reason: str,
    threshold: float,
    accuracy: Optional[float] = None,
) -> None:
    fields = {"env": env, "threshold": str(threshold), "reason": str(reason)}
    if accuracy is not None:
        fields["accuracy"] = f"{float(accuracy):.4f}"
    notify_info("Branch: shadow", **fields)


def notify_register_completed(*, env: str, model: str, alias: str, version: int) -> None:
    notify_success(
        "MLflow register+alias completed",
        env=env,
        model=model,
        alias=alias,
        version=str(int(version)),
    )


def notify_shadow_reason(*, env: str, reason: Optional[str]) -> None:
    """
    Shadow로 빠진 이유(SSOT) 기준으로 Slack 메시지를 표준화합니다.
    """
    title = "Shadow path selected"

    if reason == SHADOW_REASON_TRAIN_SKIPPED:
        notify_skip(title, env=env, reason="train skipped", next_action="데이터/피처/라벨 조건 확인")
        return

    if reason == SHADOW_REASON_ACCURACY_INVALID:
        notify_skip(title, env=env, reason="accuracy invalid", next_action="train task의 accuracy 산출/형 변환 확인")
        return

    if reason == SHADOW_REASON_DRIFT_DETECTED:
        notify_skip(
            title,
            env=env,
            reason="drift detected (pre-deploy gate)",
            next_action="feature contract / schema / data shift 확인",
        )
        return

    # default: below threshold
    if reason is None:
        reason = SHADOW_REASON_BELOW_THRESHOLD

    notify_skip(
        title,
        env=env,
        reason="accuracy below threshold",
        next_action="feature/label/model 개선 후 재시도",
    )
