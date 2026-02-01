from __future__ import annotations
import os
from airflow.sdk import Variable

def v(key: str, default=None):
    try:
        return Variable.get(key)
    except Exception:
        return default

def env() -> str:
    return v("env", os.getenv("ENV", "dev")) or "dev"

def accuracy_threshold() -> float:
    x = v("accuracy_threshold", os.getenv("ACCURACY_THRESHOLD", "0.60"))
    try:
        return float(x)
    except Exception:
        return 0.60

def model_name() -> str:
    return v("model_name", os.getenv("MODEL_NAME", "best_model")) or "best_model"

def alias() -> str:
    return v("mlflow_alias", os.getenv("MLFLOW_ALIAS", "A")) or "A"

def mlflow_tracking_uri() -> str:
    uri = os.getenv("MLFLOW_TRACKING_URI")
    if not uri:
        raise RuntimeError("MLFLOW_TRACKING_URI is required")
    return uri

def artifacts_s3_bucket() -> str:
    # MLflow artifact store bucket (for integrity checks / policy)
    return v("mlflow_artifacts_bucket", os.getenv("MLFLOW_ARTIFACTS_BUCKET", "")) or ""

def feature_s3_prefix() -> str:
    # e.g. s3://datapipeline-raw-data-keonho/feature-store/user_features/
    p = os.getenv("FEATURE_S3_PREFIX") or v("feature_s3_prefix", "")
    if not p:
        raise RuntimeError("FEATURE_S3_PREFIX is required")
    return p.rstrip("/") + "/"

def triton_http_url() -> str:
    return os.getenv("TRITON_HTTP_URL") or v("triton_http_url", "http://triton.triton-dev.svc.cluster.local:8000")

def triton_model_repo() -> str:
    # mounted path inside pods (shared PV)
    return os.getenv("TRITON_MODEL_REPO") or v("triton_model_repo", "/models")

def fastapi_base_url() -> str:
    # inside cluster or via ingress - choose one consistent policy
    return os.getenv("FASTAPI_BASE_URL") or v("fastapi_base_url", "https://fastapi.local")

def fastapi_token() -> str:
    t = os.getenv("FASTAPI_TOKEN") or v("fastapi_token", "")
    if not t:
        raise RuntimeError("FASTAPI_TOKEN is required")
    return t

