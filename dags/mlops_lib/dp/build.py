from __future__ import annotations

import io, csv
from datetime import datetime, timezone, timedelta
from airflow.models import Variable
from airflow.utils.log.logging_mixin import LoggingMixin

from mlops_lib.core.policy import VAR_DP_MIN_ROWS
from .s3 import get_s3_client, parse_s3_uri
from .feature_schema import load_schema

logger = LoggingMixin().log
KST = timezone(timedelta(hours=9))


def build_features(raw_path: str, pipeline_name: str, feature_set: str, schema_path: str, feature_base: str, ti) -> None:
    schema, schema_hash = load_schema(schema_path, expected_feature_set=feature_set)
    cols = [c["name"] for c in schema["columns"]]

    # schema columns 기준으로 CSV를 만들되, event_timestamp는 항상 보장
    if "event_timestamp" not in cols:
        cols.append("event_timestamp")

    s3 = get_s3_client()
    bucket, key = parse_s3_uri(raw_path)
    obj = s3.get_object(Bucket=bucket, Key=key)
    text = obj["Body"].read().decode("utf-8")

    reader = csv.DictReader(io.StringIO(text))
    rows = list(reader)

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
        if ts and ((rec["max_ts"] is None) or (ts > rec["max_ts"])):
            rec["max_ts"] = ts

    # -----------------------
    # feature rows 만들기
    # -----------------------
    feature_dicts: list[dict] = []
    for uid, rec in by_user.items():
        row_map = {
            "user_id": uid,
            "f_total_events_7d": int(rec["cnt"]),
            "f_avg_session_sec_7d": (rec["sess_sum"] / rec["sess_cnt"]) if rec["sess_cnt"] > 0 else 0.0,
            "f_last_event_age_sec": int((now - rec["max_ts"]).total_seconds()) if rec["max_ts"] else 0,
            "event_timestamp": now.isoformat(),
        }
        feature_dicts.append(row_map)

    # -----------------------
    # ✅ label 생성 (정답 정의 고정)
    # - 아주 설명 가능한 규칙 기반
    # - label: 0(저활성) / 1(중간) / 2(고활성)
    # -----------------------
    def _make_label(row: dict) -> int:
        cnt = int(row.get("f_total_events_7d", 0) or 0)
        avg_sess = float(row.get("f_avg_session_sec_7d", 0.0) or 0.0)

        # 고활성: 이벤트 많고 세션 길이도 충분
        if cnt >= 4 and avg_sess >= 300:
            return 2

        # 저활성: 이벤트가 거의 없거나 세션이 너무 짧음
        if cnt <= 1 or avg_sess <= 30:
            return 0

        # 나머지
        return 1

    for row in feature_dicts:
        row["label"] = _make_label(row)

    # -----------------------
    # ✅ 품질 게이트 (너무 작은 데이터면 "모델 학습이 의미 없음")
    # - rows 너무 적으면 downstream에서 스킵하게 만드는 게 맞음
    #   (TrainSkippableError로 넘기는 대신, 여기서 상황을 명확히 남김)
    # -----------------------
    n_rows = len(feature_dicts)
    label_counts = {}
    for row in feature_dicts:
        label_counts[row["label"]] = label_counts.get(row["label"], 0) + 1

    # 최소 조건(데모/운영 기준으로 권장)
    min_rows = int(Variable.get(VAR_DP_MIN_ROWS, default_var=50))
    min_classes = 2

    uniq_labels = sorted(label_counts.keys())
    if n_rows < 2:
        logger.warning("[FS] too few rows=%d; training will be skipped downstream", n_rows)
    elif len(uniq_labels) < min_classes:
        logger.warning("[FS] insufficient label diversity labels=%s; training will be skipped downstream", uniq_labels)
    elif n_rows < min_rows:
        logger.warning("[FS] rows=%d < min_rows=%d (recommend). training may be weak", n_rows, min_rows)

    # -----------------------
    # schema columns 순서대로 CSV 구성
    # - schema에 label이 포함되면 자동 포함
    # - schema에 label이 없으면 cols에 label이 없어서 빠질 수 있으니 아래에서 보정
    # -----------------------
    if "label" not in cols:
        # schema 업데이트 전/후 모두 안전하게 동작하도록
        cols.insert(len(cols) - 1 if "event_timestamp" in cols else len(cols), "label")

    feature_rows = []
    for row in feature_dicts:
        feature_rows.append([row.get(c, "") for c in cols])

    buf = io.StringIO()
    w = csv.writer(buf)
    w.writerow(cols)
    w.writerows(feature_rows)
    csv_bytes = buf.getvalue().encode("utf-8")

    # CSV를 S3 staging 경로에 직접 쓰고 URI만 XCom에 전달.
    # XCom은 메타데이터(URI, 수치) 전용 — 대용량 바이트를 MetadataDB에 저장하지 않는다.
    bkt, base_prefix = parse_s3_uri(feature_base)
    staging_key = f"{base_prefix.rstrip('/')}/{pipeline_name}/staging/features.csv"
    staging_uri = f"s3://{bkt}/{staging_key}"

    s3.put_object(Bucket=bkt, Key=staging_key, Body=csv_bytes, ContentType="text/csv")
    logger.info("[FS] staging CSV written uri=%s bytes=%d", staging_uri, len(csv_bytes))

    ti.xcom_push(key="fs_schema", value=schema)
    ti.xcom_push(key="fs_schema_hash", value=schema_hash)
    ti.xcom_push(key="fs_features_csv_uri", value=staging_uri)
    ti.xcom_push(key="fs_feature_rows", value=n_rows)
    ti.xcom_push(key="dp_raw_path", value=raw_path)

    # label 분포를 XCom에 남겨서 run summary/MLflow tag로 연결 가능
    ti.xcom_push(key="fs_label_counts", value=label_counts)

    logger.info(
        "[FS] build_features OK rows=%d schema_hash=%s labels=%s",
        n_rows, schema_hash, label_counts
    )

