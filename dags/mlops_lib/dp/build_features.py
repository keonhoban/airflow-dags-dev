# dags/mlops_lib/dp/build_features.py
from __future__ import annotations

import pandas as pd


def build_features(df_raw: pd.DataFrame) -> pd.DataFrame:
    """
    제출용 최소 Feature Engineering:
    - 이벤트 로그(raw)에서 유저 단위 집계
    - 필수 3개 feature 생성
    """
    # 요구 칼럼이 없으면 안전하게 실패
    required_raw = {"user_id", "event_ts", "session_sec"}
    missing = sorted(list(required_raw - set(df_raw.columns)))
    if missing:
        raise ValueError(f"raw columns missing: {missing}")

    df = df_raw.copy()

    # 기본 정리
    df["session_sec"] = pd.to_numeric(df["session_sec"], errors="coerce").fillna(0.0).clip(lower=0.0)
    df["event_ts"] = pd.to_datetime(df["event_ts"], errors="coerce")
    df = df.dropna(subset=["user_id", "event_ts"])

    # 최근 7일 기준(제출용: 현재 df 범위 기반)
    max_ts = df["event_ts"].max()
    start = max_ts - pd.Timedelta(days=7)
    df7 = df[df["event_ts"] >= start]

    g = df7.groupby("user_id")

    out = pd.DataFrame(
        {
            "user_id": g.size().index,
            "f_total_events_7d": g.size().values.astype(float),
            "f_avg_session_sec_7d": g["session_sec"].mean().values.astype(float),
            "f_last_event_age_sec": (max_ts - g["event_ts"].max()).dt.total_seconds().values.astype(float),
        }
    )

    # user_id는 학습에는 쓰지 않고(설명용) 남겨두기
    return out

