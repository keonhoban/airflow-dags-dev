import os, json, shutil
from datetime import datetime, timezone
from zoneinfo import ZoneInfo
import requests
import mlflow
from mlflow.tracking import MlflowClient
from airflow.sdk import Variable
from airflow.utils.log.logging_mixin import LoggingMixin

log = LoggingMixin().log

def cfg(key: str, default=None, *, required: bool = False):
    v = os.getenv(key)
    if v is not None and str(v).strip() != "":
        return v

    try:
        if default is None:
            v = Variable.get(key)
        else:
            v = Variable.get(key, default_var=str(default))
        if v is not None and str(v).strip() != "":
            return v
    except Exception:
        pass

    if required:
        raise RuntimeError(f"[Config] missing required key: {key} (ENV or Airflow Variable)")
    return default

def utc_ts():
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S")

def kst_ts():
    return datetime.now(ZoneInfo("Asia/Seoul")).strftime("%Y%m%dT%H%M%S")

def select_prod(model_name: str, tag_key="stage", tag_value="Production"):
    uri = cfg("MLFLOW_TRACKING_URI", required=True)
    mlflow.set_tracking_uri(uri)
    c = MlflowClient(tracking_uri=uri)

    prod = []
    for mv in c.search_model_versions(f"name='{model_name}'"):
        if (mv.tags or {}).get(tag_key) == tag_value:
            prod.append(mv)

    if not prod:
        raise RuntimeError(f"no prod version: {model_name} tags.{tag_key}={tag_value}")

    prod.sort(key=lambda x: int(x.version), reverse=True)
    mv = prod[0]
    return mv.version, mv.run_id

def materialize(ti, **_):
    model = cfg("triton_model_name", required=True)
    repo  = cfg("triton_repo_base", "/models")
    onnx_rel = cfg("triton_onnx_artifact_path", "onnx/model.onnx")

    v, run_id = select_prod(model)

    deploy = f"v{v}_{kst_ts()}"
    model_dir = os.path.join(repo, model)
    ver_dir = os.path.join(model_dir, deploy)
    os.makedirs(ver_dir, exist_ok=True)

    local = mlflow.artifacts.download_artifacts(artifact_uri=f"runs:/{run_id}/{onnx_rel}")
    dst = os.path.join(ver_dir, "model.onnx")
    shutil.copyfile(local, dst)

    ti.xcom_push(key="model", value=model)
    ti.xcom_push(key="model_dir", value=model_dir)
    ti.xcom_push(key="deploy", value=deploy)
    ti.xcom_push(key="mlflow_version", value=int(v))
    ti.xcom_push(key="run_id", value=run_id)

    log.info("[W6] materialize OK model=%s deploy=%s dst=%s", model, deploy, dst)

def triton_load(ti, **_):
    model = ti.xcom_pull(task_ids="materialize_repo", key="model")
    triton = cfg("triton_http_url", required=True)

    r = requests.post(f"{triton}/v2/repository/models/{model}/load", timeout=5)
    if r.status_code != 200:
        raise RuntimeError(f"load failed: {r.status_code} {r.text}")

def commit_current(ti, **_):
    model_dir = ti.xcom_pull(task_ids="materialize_repo", key="model_dir")
    payload = {
        "active_version": ti.xcom_pull(task_ids="materialize_repo", key="deploy"),
        "mlflow_version": ti.xcom_pull(task_ids="materialize_repo", key="mlflow_version"),
        "run_id": ti.xcom_pull(task_ids="materialize_repo", key="run_id"),
        "updated_at_utc": utc_ts(),
    }
    path = os.path.join(model_dir, "current.json")
    with open(path + ".tmp", "w") as f:
        json.dump(payload, f, indent=2)
    os.replace(path + ".tmp", path)
