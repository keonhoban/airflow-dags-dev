# dags/mlops_lib/observability/notify.py
from __future__ import annotations

from typing import Optional

from utils.slack_alerts import notify_info, notify_skip, notify_success
from mlops_lib.core.ids import (
    SHADOW_REASON_TRAIN_SKIPPED,
    SHADOW_REASON_ACCURACY_INVALID,
    SHADOW_REASON_BELOW_THRESHOLD,
    SHADOW_REASON_DRIFT_DETECTED,
)


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


def notify_branch_canary(
    *,
    env: str,
    accuracy: float,
    threshold: float,
    promote_threshold: float,
) -> None:
    notify_info(
        "Branch: canary",
        env=env,
        accuracy=f"{float(accuracy):.4f}",
        canary_threshold=str(threshold),
        promote_threshold=str(promote_threshold),
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
