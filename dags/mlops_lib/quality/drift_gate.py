# dags/mlops_lib/quality/drift_gate.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from airflow.utils.log.logging_mixin import LoggingMixin

from mlops_lib.core.ids import (
    DP_STORE_TASK_ID,
    XCOM_FS_FEATURE_URI,
    XCOM_FS_LATEST_PREFIX,
    XCOM_DRIFT_BLOCK_PROMOTION,
    XCOM_DRIFT_REASON,
    XCOM_SHADOW_REASON,
    SHADOW_REASON_DRIFT_DETECTED,
)
from mlops_lib.core.policy import drift_settings

log = LoggingMixin().log


@dataclass(frozen=True)
class DriftDecision:
    block_promotion: bool
    reason: str
    signals: Dict[str, Any]


def _ks_stat(x: np.ndarray, y: np.ndarray) -> float:
    """
    Two-sample KS statistic (D).
    - p-value는 외부 의존성/환경차가 생길 수 있어, 제출/운영 최소 스택에서는 D만 사용
    """
    x = np.sort(x)
    y = np.sort(y)
    n = x.size
    m = y.size
    if n == 0 or m == 0:
        return 0.0

    data_all = np.sort(np.concatenate([x, y]))
    cdf_x = np.searchsorted(x, data_all, side="right") / n
    cdf_y = np.searchsorted(y, data_all, side="right") / m
    return float(np.max(np.abs(cdf_x - cdf_y)))


def _pick_numeric_columns(df: pd.DataFrame, max_cols: int) -> List[str]:
    cols: List[str] = []
    for c in df.columns:
        if pd.api.types.is_numeric_dtype(df[c]):
            cols.append(c)
    return cols[:max_cols]


def _read_feature_df(uri: str, sample_n: int) -> pd.DataFrame:
    df = pd.read_parquet(uri)
    if len(df) > sample_n:
        df = df.sample(n=sample_n, random_state=42)
    return df


def _latest_feature_uri(latest_prefix: str) -> str:
    # store.py에서 fs_latest_prefix가 "s3://bucket/prefix" 형태라고 가정
    p = latest_prefix.rstrip("/")
    return f"{p}/features.parquet"


def drift_gate(**context: Any) -> None:
    """
    Pre-deploy drift gate:
    - new fs_feature_uri vs fs_latest_prefix/features.parquet 비교
    - KS stat worst가 threshold 초과 -> promotion 차단(Shadow-only)
    """
    ti = context["ti"]
    ds = drift_settings()

    ks_th = float(ds.ks_stat_threshold)
    sample_n = int(ds.sample_n)
    max_cols = int(ds.max_columns)

    feature_uri = ti.xcom_pull(key=XCOM_FS_FEATURE_URI, task_ids=DP_STORE_TASK_ID)
    latest_prefix = ti.xcom_pull(key=XCOM_FS_LATEST_PREFIX, task_ids=DP_STORE_TASK_ID)

    if not feature_uri or not latest_prefix:
        d = DriftDecision(
            block_promotion=False,
            reason="DRIFT_SKIPPED: missing feature_uri or latest_prefix",
            signals={"feature_uri": feature_uri, "latest_prefix": latest_prefix},
        )
        ti.xcom_push(key=XCOM_DRIFT_BLOCK_PROMOTION, value=d.block_promotion)
        ti.xcom_push(key=XCOM_DRIFT_REASON, value=d.reason)
        log.info("[drift_gate] %s signals=%s", d.reason, d.signals)
        return

    ref_uri = _latest_feature_uri(str(latest_prefix))

    new_df = _read_feature_df(str(feature_uri), sample_n=sample_n)
    ref_df = _read_feature_df(str(ref_uri), sample_n=sample_n)

    cols = _pick_numeric_columns(new_df, max_cols=max_cols)
    cols = [c for c in cols if c in ref_df.columns]

    if not cols:
        d = DriftDecision(
            block_promotion=False,
            reason="DRIFT_SKIPPED: no common numeric columns",
            signals={"new_rows": len(new_df), "ref_rows": len(ref_df)},
        )
        ti.xcom_push(key=XCOM_DRIFT_BLOCK_PROMOTION, value=d.block_promotion)
        ti.xcom_push(key=XCOM_DRIFT_REASON, value=d.reason)
        log.info("[drift_gate] %s signals=%s", d.reason, d.signals)
        return

    worst_col = "-"
    worst_stat = 0.0
    stats: Dict[str, float] = {}

    for c in cols:
        x = new_df[c].dropna().to_numpy()
        y = ref_df[c].dropna().to_numpy()
        if x.size < 50 or y.size < 50:
            continue

        d_stat = _ks_stat(x.astype(float), y.astype(float))
        stats[c] = d_stat
        if d_stat > worst_stat:
            worst_stat = d_stat
            worst_col = c

    block = bool(worst_stat > ks_th)
    reason = (
        f"DRIFT_BLOCK: worst_col={worst_col} ks={worst_stat:.4f} (> {ks_th})"
        if block
        else f"DRIFT_OK: worst_col={worst_col} ks={worst_stat:.4f} (<= {ks_th})"
    )

    ti.xcom_push(key=XCOM_DRIFT_BLOCK_PROMOTION, value=block)
    ti.xcom_push(key=XCOM_DRIFT_REASON, value=reason)

    if block:
        # branch에서 별도 판단 없이 SSOT reason으로 shadow 알림까지 연결되게
        ti.xcom_push(key=XCOM_SHADOW_REASON, value=SHADOW_REASON_DRIFT_DETECTED)

    log.info(
        "[drift_gate] %s feature_uri=%s ref_uri=%s sample_n=%s max_cols=%s stats_top=%s",
        reason,
        feature_uri,
        ref_uri,
        sample_n,
        max_cols,
        dict(list(sorted(stats.items(), key=lambda kv: kv[1], reverse=True))[:5]),
    )
