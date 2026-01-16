# dags/mlops_lib/dp/build.py
from __future__ import annotations

import io
import csv
from datetime import datetime, timezone, timedelta

from airflow.utils.log.logging_mixin import LoggingMixin

from .s3 import get_s3_client, parse_s3_uri
from .feature_schema import load_schema

logger = LoggingMixin().log
KST = timezone(timedelta(hours=9))


def build_features(
    raw_path: str,
    pipeline_name: str,
    feature_set: str,
    schema_path: str,
    ti,
) -> None:
    """
    RAW S3 -> schema 기반 Feature 계산 -> XCom 저장
    - XCom에는 "features_csv"를 담습니다 (실무에서는 크기 커지면 S3 임시 저장으로 바꿉니다)

    ✅ Feast를 위해 event_timestamp 컬럼을 추가합니다.
    - schema에 없으면 자동으로 cols에 append
    - 값은 KST now를 ISO-8601 형태로 넣습니다.
    """
    schema, schema_hash = load_schema(schema_path, expected_feature_set=feature_set)
    cols = [c["name"] for c in schema["columns"]]

    # ✅ Feast apply에서 timestamp_field inference 실패 방지용
    # schema에 event_timestamp가 없더라도 강제로 컬럼을 추가합니다.
    if "event_timestamp" not in cols:
        cols.append("event_timestamp")

    s3 = get_s3_client()
    bucket, key = parse_s3_uri(raw_path)

    logger.info(
        "[DP] pipeline=%s step=build_features action=get_object bucket=%s key=%s",
        pipeline_name, bucket, key
    )
    obj = s3.get_object(Bucket=bucket, Key=key)
    text = obj["Body"].read().decode("utf-8")

    reader = csv.DictReader(io.StringIO(text))
    rows = list(reader)

    # 최소 실무형 피처(샘플):
    # - f_total_events_7d: user별 row count
    # - f_avg_session_sec_7d: session_length_sec 평균 (없으면 0)
    # - f_last_event_age_sec: event_ts(now - max(ts)), 없으면 0
    by_user: dict[int, dict] = {}
    now = datetime.now(KST)

    def _to_float(x):
        try:
            return float(x)
        except Exception:
            return 0.0

    def _to_int(x):
        try:
            return int(float(x))
        except Exception:
            return 0

    def _parse_ts(s: str | None):
        if not s:
            return None
        s = s.strip()
        for fmt in ("%Y-%m-%dT%H:%M:%S", "%Y-%m-%d %H:%M:%S"):
            try:
                return datetime.strptime(s, fmt).replace(tzinfo=KST)
            except Exception:
                pass
        return None

    for r in rows:
        uid = _to_int(r.get("user_id"))
        if uid == 0:
            continue

        rec = by_user.setdefault(uid, {"cnt": 0, "sess_sum": 0.0, "sess_cnt": 0, "max_ts": None})
        rec["cnt"] += 1

        sess = r.get("session_length_sec")
        if sess is not None:
            rec["sess_sum"] += _to_float(sess)
            rec["sess_cnt"] += 1

        ts = _parse_ts(r.get("event_ts") or r.get("timestamp") or r.get("ts"))
        if ts:
            if (rec["max_ts"] is None) or (ts > rec["max_ts"]):
                rec["max_ts"] = ts

    # ✅ Feast timestamp_field로 쓸 이벤트 타임스탬프 (KST, ISO-8601)
    event_ts_iso = now.isoformat()

    feature_rows = []
    for uid, rec in by_user.items():
        row_map = {
            "user_id": uid,
            "f_total_events_7d": rec["cnt"],
            "f_avg_session_sec_7d": (rec["sess_sum"] / rec["sess_cnt"]) if rec["sess_cnt"] > 0 else 0.0,
            "f_last_event_age_sec": int((now - rec["max_ts"]).total_seconds()) if rec["max_ts"] else 0,
            # ✅ Feast용 timestamp 컬럼
            "event_timestamp": event_ts_iso,
        }
        feature_rows.append([row_map.get(c, "") for c in cols])

    buf = io.StringIO()
    w = csv.writer(buf)
    w.writerow(cols)
    w.writerows(feature_rows)
    features_csv = buf.getvalue()

    ti.xcom_push(key="fs_schema", value=schema)
    ti.xcom_push(key="fs_schema_hash", value=schema_hash)
    ti.xcom_push(key="fs_features_csv", value=features_csv)
    ti.xcom_push(key="fs_feature_rows", value=len(feature_rows))
    ti.xcom_push(key="dp_raw_path", value=raw_path)  # downstream에서 source로 사용

    logger.info(
        "[FS] build_features OK set=%s rows=%d schema_hash=%s event_timestamp=%s",
        feature_set, len(feature_rows), schema_hash, event_ts_iso
    )
