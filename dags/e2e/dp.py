from __future__ import annotations

import io, csv, json
from datetime import datetime, timedelta, timezone as dt_tz
from zoneinfo import ZoneInfo

import boto3
import pandas as pd
from jinja2 import Template
from airflow.utils.log.logging_mixin import LoggingMixin

from e2e.config import cfg
from e2e.utils import parse_s3_uri, kst_now_iso

log = LoggingMixin().log
KST = ZoneInfo("Asia/Seoul")

def _s3():
    return boto3.client("s3")

def extract_raw(**context):
    raw_path = cfg("dp_raw_path", required=True)
    b, k = parse_s3_uri(raw_path)
    _s3().head_object(Bucket=b, Key=k)
    context["ti"].xcom_push(key="raw_path", value=raw_path)
    log.info("[DP] extract_raw OK raw=%s", raw_path)

def validate_raw(**context):
    raw_path = context["ti"].xcom_pull(task_ids="extract_raw", key="raw_path")
    b, k = parse_s3_uri(raw_path)
    text = _s3().get_object(Bucket=b, Key=k)["Body"].read().decode("utf-8")
    lines = [ln for ln in text.splitlines() if ln.strip()]
    if len(lines) <= 1:
        raise ValueError("[DP] validate failed: empty or header-only")
    context["ti"].xcom_push(key="raw_lines", value=len(lines))
    log.info("[DP] validate_raw OK lines=%d", len(lines))

def build_features(**context):
    """
    RAW -> user-level features (small, deterministic)
    + event_timestamp (Feast timestamp_field)
    """
    raw_path = context["ti"].xcom_pull(task_ids="extract_raw", key="raw_path")
    b, k = parse_s3_uri(raw_path)
    text = _s3().get_object(Bucket=b, Key=k)["Body"].read().decode("utf-8")

    reader = csv.DictReader(io.StringIO(text))
    rows = list(reader)

    now = datetime.now(KST)
    def _to_int(x):
        try: return int(float(x))
        except Exception: return 0
    def _to_float(x):
        try: return float(x)
        except Exception: return 0.0
    def _parse_ts(s: str | None):
        if not s: return None
        s = s.strip()
        for fmt in ("%Y-%m-%dT%H:%M:%S", "%Y-%m-%d %H:%M:%S"):
            try:
                return datetime.strptime(s, fmt).replace(tzinfo=KST)
            except Exception:
                pass
        return None

    by_user: dict[int, dict] = {}
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
        if ts and (rec["max_ts"] is None or ts > rec["max_ts"]):
            rec["max_ts"] = ts

    cols = ["user_id","f_total_events_7d","f_avg_session_sec_7d","f_last_event_age_sec","event_timestamp"]
    out = io.StringIO()
    w = csv.writer(out)
    w.writerow(cols)

    for uid, rec in by_user.items():
        w.writerow([
            uid,
            rec["cnt"],
            (rec["sess_sum"]/rec["sess_cnt"]) if rec["sess_cnt"] else 0.0,
            int((now - rec["max_ts"]).total_seconds()) if rec["max_ts"] else 0,
            now.isoformat(),
        ])

    features_csv = out.getvalue()
    ti = context["ti"]
    ti.xcom_push(key="features_csv", value=features_csv)
    ti.xcom_push(key="feature_rows", value=len(by_user))
    log.info("[DP] build_features OK rows=%d", len(by_user))

def store_features(**context):
    """
    versioned + latest (csv + parquet + metadata)
    """
    ti = context["ti"]
    feature_base = cfg("dp_feature_base", required=True)  # s3://.../feature-store
    feature_set = cfg("dp_feature_set", "user_features")
    pipeline_name = cfg("dp_pipeline_name", "daily_user_events")
    metadata_tpl_path = cfg("dp_metadata_tpl_path", "/opt/airflow/feature-store/metadata.json.j2")

    features_csv = ti.xcom_pull(task_ids="build_features", key="features_csv")
    if not features_csv:
        raise ValueError("[DP] store_features: features_csv missing")

    exec_dt = getattr(ti, "execution_date", None)
    kst = ZoneInfo("Asia/Seoul")
    ver_dt = exec_dt.astimezone(kst) if exec_dt else datetime.now(kst)
    version = "v_" + ver_dt.strftime("%Y%m%dT%H%M%S")

    bkt, base_prefix = parse_s3_uri(feature_base)
    base_prefix = base_prefix.rstrip("/") + f"/{feature_set}/"
    ver_prefix = base_prefix + f"{version}/"
    latest_prefix = base_prefix + "latest/"

    # csv -> parquet (Feast용)
    df = pd.read_csv(io.StringIO(features_csv))
    df["event_timestamp"] = pd.to_datetime(df["event_timestamp"], utc=True, errors="raise")
    buf = io.BytesIO()
    df.to_parquet(buf, index=False)
    parquet_bytes = buf.getvalue()

    # metadata template
    with open(metadata_tpl_path, "r", encoding="utf-8") as f:
        tpl = Template(f.read())
    feature_uri = f"s3://{bkt}/{ver_prefix}features.parquet"
    meta = tpl.render(
        version=version,
        generated_at=kst_now_iso(),
        pipeline=pipeline_name,
        feature_set=feature_set,
        feature_uri=feature_uri,
        source=ti.xcom_pull(task_ids="extract_raw", key="raw_path"),
    ).encode("utf-8")

    s3 = _s3()
    def _put(prefix: str):
        s3.put_object(Bucket=bkt, Key=f"{prefix}features.csv", Body=features_csv.encode("utf-8"), ContentType="text/csv")
        s3.put_object(Bucket=bkt, Key=f"{prefix}features.parquet", Body=parquet_bytes, ContentType="application/octet-stream")
        s3.put_object(Bucket=bkt, Key=f"{prefix}metadata.json", Body=meta, ContentType="application/json")

    _put(ver_prefix)
    _put(latest_prefix)

    ti.xcom_push(key="fs_version", value=version)
    ti.xcom_push(key="feature_uri", value=feature_uri)
    log.info("[DP] store_features OK version=%s uri=%s", version, feature_uri)

def summarize_run(**context):
    ti = context["ti"]
    raw = ti.xcom_pull(task_ids="extract_raw", key="raw_path")
    uri = ti.xcom_pull(task_ids="store_features", key="feature_uri")
    acc = ti.xcom_pull(task_ids="train", key="accuracy")
    run_id = ti.xcom_pull(task_ids="train", key="run_id")
    log.info("[SUMMARY] raw=%s uri=%s acc=%s run_id=%s", raw, uri, acc, run_id)

