# dags/mlops_lib/dp/tasks.py
from __future__ import annotations

import hashlib
from datetime import datetime
from zoneinfo import ZoneInfo

import pandas as pd
from airflow.utils.log.logging_mixin import LoggingMixin

from mlops_lib.dp.config import dp_bucket, dp_latest_key, dp_versioned_key, dp_metadata_key
from mlops_lib.dp.schema import required_columns, SCHEMA
from mlops_lib.dp.s3 import put_parquet, put_json, s3_uri
from mlops_lib.dp.build_features import build_features

log = LoggingMixin().log


def _kst_version() -> str:
    return datetime.now(ZoneInfo("Asia/Seoul")).strftime("%Y%m%d%H%M%S")


def _schema_hash(cols: list[str]) -> str:
    x = ",".join(cols).encode("utf-8")
    return hashlib.sha256(x).hexdigest()[:12]


def task_extract_raw_data(ti, **_):
    """
    제출용 최소:
    - raw 데이터 소스 연결 대신 '모의 raw' 생성
    - 실무에선 여기만 교체(S3/DB/Kafka 등)
    """
    now = pd.Timestamp.utcnow()
    df_raw = pd.DataFrame(
        {
            "user_id": ["u1", "u1", "u2", "u3", "u3", "u3"],
            "event_ts": [now, now - pd.Timedelta(hours=1), now - pd.Timedelta(days=1),
                         now - pd.Timedelta(days=2), now - pd.Timedelta(days=3), now - pd.Timedelta(days=6)],
            "session_sec": [120, 30, 60, 10, 200, 15],
        }
    )
    # XCom으로 raw를 직접 넘기면 크기 문제 생길 수 있어서 'pickle' 방식을 피하고,
    # 제출용 최소에서는 바로 다음 task가 생성하도록 "raw_ready" 신호만 남깁니다.
    ti.xcom_push(key="raw_ready", value=True)
    log.info("[DP] raw prepared rows=%s", len(df_raw))

    # 여기서는 간단히 features 생성 쪽에서 raw를 다시 만들도록(제출용 단순화)
    return True


def task_validate_data(ti, **_):
    # 제출용 최소: raw_ready만 확인
    ok = bool(ti.xcom_pull(task_ids="extract_raw_data", key="raw_ready"))
    if not ok:
        raise RuntimeError("raw not ready")
    return True


def task_build_features(ti, **_):
    """
    raw → features 생성
    제출용에서는 extract에서 만든 raw 대신 동일 mock 생성(단순화)
    """
    now = pd.Timestamp.utcnow()
    df_raw = pd.DataFrame(
        {
            "user_id": ["u1", "u1", "u2", "u3", "u3", "u3"],
            "event_ts": [now, now - pd.Timedelta(hours=1), now - pd.Timedelta(days=1),
                         now - pd.Timedelta(days=2), now - pd.Timedelta(days=3), now - pd.Timedelta(days=6)],
            "session_sec": [120, 30, 60, 10, 200, 15],
        }
    )

    df_feat = build_features(df_raw)

    # 스키마(필수 칼럼) 검증
    cols = required_columns()
    missing = [c for c in cols if c not in df_feat.columns]
    if missing:
        raise RuntimeError(f"feature columns missing: {missing}")

    ti.xcom_push(key="feature_rows", value=int(len(df_feat)))
    ti.xcom_push(key="schema_name", value=SCHEMA.name)
    ti.xcom_push(key="schema_hash", value=_schema_hash(cols))

    # Airflow XCom에 DataFrame을 올리지 않음(실무 기준)
    # store_features에서 다시 생성하는 대신, 제출용에서는 parquet 저장을 여기서 수행하지 않고
    # 다음 task에서 한번에 저장(정리).
    return True


def task_store_features(ti, **_):
    """
    S3에 version + latest 저장, metadata 남김, XCom으로 feature_uri 전달
    """
    version = _kst_version()
    bucket = dp_bucket()

    # 제출용: build_features와 동일 raw/feature 재생성(단순화)
    now = pd.Timestamp.utcnow()
    df_raw = pd.DataFrame(
        {
            "user_id": ["u1", "u1", "u2", "u3", "u3", "u3"],
            "event_ts": [now, now - pd.Timedelta(hours=1), now - pd.Timedelta(days=1),
                         now - pd.Timedelta(days=2), now - pd.Timedelta(days=3), now - pd.Timedelta(days=6)],
            "session_sec": [120, 30, 60, 10, 200, 15],
        }
    )
    df_feat = build_features(df_raw)
    df_feat = df_feat.drop(columns=["user_id"], errors="ignore")  # 학습용에서는 제거

    vkey = dp_versioned_key(version)
    lkey = dp_latest_key()
    mkey = dp_metadata_key(version)

    put_parquet(bucket, vkey, df_feat)
    put_parquet(bucket, lkey, df_feat)

    meta = {
        "schema_name": ti.xcom_pull(task_ids="build_features", key="schema_name"),
        "schema_hash": ti.xcom_pull(task_ids="build_features", key="schema_hash"),
        "version": version,
        "rows": int(ti.xcom_pull(task_ids="build_features", key="feature_rows") or 0),
        "saved_keys": {"versioned": vkey, "latest": lkey},
        "created_at_utc": datetime.utcnow().isoformat() + "Z",
    }
    put_json(bucket, mkey, meta)

    feature_uri = s3_uri(bucket, vkey)
    ti.xcom_push(key="fs_feature_uri", value=feature_uri)

    log.info("[DP] stored version=%s uri=%s", version, feature_uri)
    return feature_uri


def task_summarize_run(ti, **_):
    # 제출용 요약: run에서 핵심 xcom만 모아 로그로 남김
    feature_uri = ti.xcom_pull(task_ids="store_features", key="fs_feature_uri")
    acc = ti.xcom_pull(task_ids="train_and_eval", key="accuracy")
    run_id = ti.xcom_pull(task_ids="train_and_eval", key="run_id")
    deploy_ver = ti.xcom_pull(task_ids="triton_materialize_shadow", key="deploy_version")

    log.info("[SUMMARY] feature_uri=%s acc=%s run_id=%s deploy_ver=%s", feature_uri, acc, run_id, deploy_ver)
    return True

