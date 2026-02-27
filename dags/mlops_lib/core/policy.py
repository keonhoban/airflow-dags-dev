# dags/mlops_lib/core/policy.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

from airflow.models import Variable

from utils.slack_alerts import notify_info, notify_skip, notify_success

# -----------------------
# Airflow DAG policies (SSOT)
# -----------------------
E2E_START_DATE_YMD = (2025, 1, 1)
E2E_RETRIES = 1
E2E_RETRY_DELAY_MIN = 2
E2E_MAX_ACTIVE_RUNS = 1
E2E_DAGRUN_TIMEOUT_MIN = 30

MODEL_READY_POKE_INTERVAL_SEC = 10
MODEL_READY_TIMEOUT_SEC = 180
MODEL_READY_MODE = "reschedule"

# ✅ 정책: FastAPI reload 실패는 자동 롤백하지 않음 (model repo SSOT를 되돌리는 건 위험)
ROLLBACK_ON_FASTAPI_RELOAD_FAILURE = False

# -----------------------
# Triton timeouts (SSOT)
# -----------------------
T_TRITON_UNLOAD = 10
T_TRITON_LOAD = 10
T_TRITON_READY = 5
T_TRITON_INFER = 10


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


@dataclass(frozen=True)
class Settings:
    env: str
    accuracy_threshold: float
    logreg_c: float
    logreg_max_iter: int
    model_name: str
    alias: str
    code_version: Optional[str]

    @classmethod
    def load(cls) -> "Settings":
        env = (_v("triton_env", "dev") or "dev").strip()
        th = _to_float(_v("accuracy_threshold", "0.60"), 0.60)

        c = _to_float(_v("logreg_C", "1.0"), 1.0)
        it = _to_int(_v("logreg_max_iter", "200"), 200)

        model_name = (_v("triton_model_name", _v("model_name", "best_model")) or "best_model").strip()
        alias = (_v("mlflow_alias", "A") or "A").strip()

        code_version = (_v("code_version", None) or None)
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


# -----------------------
# Slack notify helpers (SSOT)
# -----------------------
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
    notify_info("Branch: promotion", env=env, accuracy=f"{float(accuracy):.4f}", threshold=str(threshold))


def notify_branch_shadow(*, env: str, reason: str, threshold: float, accuracy: Optional[float] = None) -> None:
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
        version=str(version),
    )


def notify_shadow_reason(*, env: str, reason: Optional[str]) -> None:
    title = "Shadow path selected"
    if reason == "train_skipped":
        notify_skip(title, env=env, reason="train skipped", next_action="데이터/피처/라벨 조건 확인")
        return
    if reason == "accuracy_invalid":
        notify_skip(title, env=env, reason="accuracy invalid", next_action="train task의 accuracy 산출/형 변환 확인")
        return
    notify_skip(title, env=env, reason="accuracy below threshold", next_action="feature/label/model 개선 후 재시도")
