# ml_code/data_pipeline_v2.py 

import io, csv, json, hashlib
from urllib.parse import urlparse
from datetime import datetime, timezone, timedelta

import boto3
from airflow.utils.log.logging_mixin import LoggingMixin

from jinja2 import Template

KST = timezone(timedelta(hours=9))
logger = LoggingMixin().log

FS_SCHEMA_PATH_DEFAULT = "/opt/airflow/feature-store/user_features.schema.json"
FS_META_TPL_PATH_DEFAULT = "/opt/airflow/feature-store/metadata.json.j2"

def _parse_s3_uri(uri: str):
    p = urlparse(uri)
    if p.scheme != "s3":
        raise ValueError(f"지원하지 않는 URI 스킴입니다: {uri}")
    return p.netloc, p.path.lstrip("/")

def _get_s3_client():
    return boto3.client("s3")

def _read_local_json(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

def _read_local_text(path: str) -> str:
    with open(path, "r", encoding="utf-8") as f:
        return f.read()

def _json_canonical_bytes(obj: dict) -> bytes:
    return json.dumps(obj, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")

def _sha256_hex(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()

def _kst_now_iso():
    return datetime.now(KST).isoformat()

def _version_id(exec_date=None):
    dt = exec_date.astimezone(KST) if exec_date else datetime.now(KST)
    return "v_" + dt.strftime("%Y%m%dT%H%M%S")

# -----------------------
# Step 1: RAW 데이터 수집
# -----------------------
def extract_raw_data(raw_path: str, pipeline_name: str, ti):
    s3 = _get_s3_client()
    bucket, key = _parse_s3_uri(raw_path)

    logger.info("[DP] pipeline=%s step=extract_raw_data action=head_object bucket=%s key=%s",
                pipeline_name, bucket, key)
    s3.head_object(Bucket=bucket, Key=key)

    ti.xcom_push(key="dp_raw_path", value=raw_path)

# -----------------------
# Step 2: 데이터 검증 (기존 유지)
# -----------------------
def validate_data(raw_path: str, pipeline_name: str, ti):
    s3 = _get_s3_client()
    bucket, key = _parse_s3_uri(raw_path)

    obj = s3.get_object(Bucket=bucket, Key=key)
    text = obj["Body"].read().decode("utf-8")

    reader = csv.reader(io.StringIO(text))
    rows, null_count, total_cells = 0, 0, 0
    header = None

    for i, row in enumerate(reader):
        if i == 0:
            header = row
            continue
        rows += 1
        total_cells += len(row)
        null_count += sum(1 for cell in row if cell == "")

    null_rate = (null_count / total_cells) if total_cells > 0 else 0.0
    valid = not (rows == 0 or null_rate > 0.5)

    logger.info("[DP] pipeline=%s step=validate_data status=%s rows=%d null_rate=%.4f",
                pipeline_name, "success" if valid else "failed", rows, null_rate)

    ti.xcom_push(key="dp_rows", value=rows)
    ti.xcom_push(key="dp_null_rate", value=null_rate)
    ti.xcom_push(key="dp_valid", value=valid)

    if not valid:
        raise ValueError(f"[DP] 데이터 검증 실패 (rows={rows}, null_rate={null_rate:.4f})")

# -----------------------
# Step 3: Feature 가공 (스키마 기반)
# -----------------------
def build_features(raw_path: str, feature_base: str, pipeline_name: str, feature_set: str, ti,
                   schema_path: str = FS_SCHEMA_PATH_DEFAULT):
    """
    - schema 파일을 읽고
    - feature_set 컬럼 순서대로 CSV를 생성
    """
    schema = _read_local_json(schema_path)
    if schema.get("feature_set") != feature_set:
        raise ValueError(f"[FS] schema feature_set mismatch: {schema.get('feature_set')} != {feature_set}")

    cols = [c["name"] for c in schema["columns"]]
    if "user_id" not in cols:
        raise ValueError("[FS] schema에 user_id가 없습니다")

    s3 = _get_s3_client()
    bucket, key = _parse_s3_uri(raw_path)
    obj = s3.get_object(Bucket=bucket, Key=key)
    text = obj["Body"].read().decode("utf-8")

    reader = csv.DictReader(io.StringIO(text))
    rows = list(reader)

    # ✅ 실무형 “최소” 피처 계산 (raw에 따라 유연하게)
    # - f_total_events_7d: user별 row count
    # - f_avg_session_sec_7d: session_length_sec 평균 (없으면 0)
    # - f_last_event_age_sec: event_ts가 있으면 now - max(ts), 없으면 0
    by_user = {}
    now = datetime.now(KST)

    def _to_float(x):
        try: return float(x)
        except: return 0.0

    def _to_int(x):
        try: return int(float(x))
        except: return 0

    def _parse_ts(s):
        # 흔한 케이스만 처리 (없으면 None)
        # 예: 2025-11-19T12:34:56, 2025-11-19 12:34:56
        if not s: return None
        s = s.strip()
        for fmt in ("%Y-%m-%dT%H:%M:%S", "%Y-%m-%d %H:%M:%S"):
            try:
                return datetime.strptime(s, fmt).replace(tzinfo=KST)
            except:
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

    feature_rows = []
    for uid, rec in by_user.items():
        f_total_events_7d = rec["cnt"]
        f_avg_session_sec_7d = (rec["sess_sum"] / rec["sess_cnt"]) if rec["sess_cnt"] > 0 else 0.0
        f_last_event_age_sec = int((now - rec["max_ts"]).total_seconds()) if rec["max_ts"] else 0

        row_map = {
            "user_id": uid,
            "f_total_events_7d": f_total_events_7d,
            "f_avg_session_sec_7d": f_avg_session_sec_7d,
            "f_last_event_age_sec": f_last_event_age_sec,
        }

        # schema 컬럼 순서대로
        feature_rows.append([row_map.get(c, "") for c in cols])

    # CSV 직렬화
    buf = io.StringIO()
    w = csv.writer(buf)
    w.writerow(cols)
    w.writerows(feature_rows)
    features_csv = buf.getvalue()

    # schema_hash
    schema_hash = _sha256_hex(_json_canonical_bytes(schema))

    ti.xcom_push(key="fs_schema", value=schema)
    ti.xcom_push(key="fs_schema_hash", value=schema_hash)
    ti.xcom_push(key="fs_features_csv", value=features_csv)
    ti.xcom_push(key="fs_feature_rows", value=len(feature_rows))
    ti.xcom_push(key="fs_feature_base", value=feature_base)
    ti.xcom_push(key="fs_feature_set", value=feature_set)

    logger.info("[FS] build_features OK set=%s rows=%d schema_hash=%s", feature_set, len(feature_rows), schema_hash)

# -----------------------
# Step 4: Feature 저장 + metadata 렌더
# -----------------------
def store_features(feature_base: str, pipeline_name: str, feature_set: str, ti,
                   metadata_tpl_path: str = FS_META_TPL_PATH_DEFAULT):
    s3 = _get_s3_client()

    schema = ti.xcom_pull(key="fs_schema", task_ids="build_features")
    schema_hash = ti.xcom_pull(key="fs_schema_hash", task_ids="build_features")
    features_csv = ti.xcom_pull(key="fs_features_csv", task_ids="build_features")
    rows = ti.xcom_pull(key="fs_feature_rows", task_ids="build_features")

    if not features_csv:
        raise ValueError("[FS] features_csv 누락")

    exec_date = getattr(ti, "execution_date", None)
    ver = _version_id(exec_date)

    # base: s3://.../feature-store
    bkt, prefix = _parse_s3_uri(feature_base)
    prefix = prefix.rstrip("/") + f"/{feature_set}/{ver}/"

    feature_uri = f"s3://{bkt}/{prefix}features.csv"

    # metadata 렌더
    tpl = Template(_read_local_text(metadata_tpl_path))
    meta_str = tpl.render(
        version=ver,
        generated_at=_kst_now_iso(),
        source=ti.xcom_pull(key="dp_raw_path", task_ids="extract_raw_data"),
        pipeline=pipeline_name,
        schema_hash=schema_hash,
        feature_uri=feature_uri,
    )

    # 저장
    s3.put_object(Bucket=bkt, Key=f"{prefix}features.csv",
                  Body=features_csv.encode("utf-8"), ContentType="text/csv")

    s3.put_object(Bucket=bkt, Key=f"{prefix}schema.json",
                  Body=json.dumps(schema, ensure_ascii=False, indent=2).encode("utf-8"),
                  ContentType="application/json")

    s3.put_object(Bucket=bkt, Key=f"{prefix}metadata.json",
                  Body=meta_str.encode("utf-8"), ContentType="application/json")

    ti.xcom_push(key="fs_version", value=ver)
    ti.xcom_push(key="fs_prefix", value=f"s3://{bkt}/{prefix}")
    ti.xcom_push(key="fs_feature_uri", value=feature_uri)

    logger.info("[FS] store_features OK prefix=s3://%s/%s rows=%s", bkt, prefix, rows)

def summarize_run(pipeline_name: str, ti):
    logger.info(
        "[FS] summarize pipeline=%s raw=%s out_prefix=%s version=%s uri=%s",
        pipeline_name,
        ti.xcom_pull(key="dp_raw_path", task_ids="extract_raw_data"),
        ti.xcom_pull(key="fs_prefix", task_ids="store_features"),
        ti.xcom_pull(key="fs_version", task_ids="store_features"),
        ti.xcom_pull(key="fs_feature_uri", task_ids="store_features"),
    )
