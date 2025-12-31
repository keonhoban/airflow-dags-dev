import os, json, shutil
from datetime import datetime, timezone
from zoneinfo import ZoneInfo

import requests
import mlflow
from mlflow.tracking import MlflowClient

from airflow.sdk import Variable
from airflow.utils.log.logging_mixin import LoggingMixin

log = LoggingMixin().log


# -----------------------------------------------------------------------------
# Config helper: ENV -> Airflow Variable -> default
# -----------------------------------------------------------------------------
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


# -----------------------------------------------------------------------------
# Time helpers
# -----------------------------------------------------------------------------
def utc_ts():
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S")


def kst_ts():
    return datetime.now(ZoneInfo("Asia/Seoul")).strftime("%Y%m%dT%H%M%S")


# -----------------------------------------------------------------------------
# MLflow: select model version by alias (ex: @A / @B)
# -----------------------------------------------------------------------------
def select_by_alias(model_name: str, alias: str):
    uri = cfg("MLFLOW_TRACKING_URI", required=True)
    mlflow.set_tracking_uri(uri)
    c = MlflowClient(tracking_uri=uri)

    try:
        mv = c.get_model_version_by_alias(model_name, alias)
    except Exception as e:
        raise RuntimeError(f"no alias version: {model_name} alias=@{alias} ({e})")

    return int(mv.version), mv.run_id


# -----------------------------------------------------------------------------
# Task 1) materialize_repo
# - Select prod target by alias
# - Download ONNX from MLflow run artifacts
# - Copy to NFS-backed Triton model repo (/models)
# -----------------------------------------------------------------------------
def materialize(ti, alias=None, **_):
    model = cfg("triton_model_name", required=True)               # ex) best_model
    alias = (alias or cfg("triton_model_alias", "A")).strip()     # ex) A or B
    repo  = cfg("triton_repo_base", "/models")                    # ex) /models
    onnx_rel = cfg("triton_onnx_artifact_path", "onnx/model.onnx")# fixed rel path

    v, run_id = select_by_alias(model, alias=alias)

    deploy = f"v{v}_{kst_ts()}"
    model_dir = os.path.join(repo, model)
    ver_dir = os.path.join(model_dir, deploy)
    os.makedirs(ver_dir, exist_ok=True)

    # MLflow artifact -> local -> NFS copy (standardize name to model.onnx)
    local = mlflow.artifacts.download_artifacts(artifact_uri=f"runs:/{run_id}/{onnx_rel}")
    dst = os.path.join(ver_dir, "model.onnx")
    shutil.copyfile(local, dst)

    # XCom for downstream tasks
    ti.xcom_push(key="model", value=model)
    ti.xcom_push(key="alias", value=alias)
    ti.xcom_push(key="model_dir", value=model_dir)
    ti.xcom_push(key="deploy", value=deploy)
    ti.xcom_push(key="mlflow_version", value=int(v))
    ti.xcom_push(key="run_id", value=run_id)
    ti.xcom_push(key="onnx_dst", value=dst)

    log.info(
        "[W6] materialize OK model=%s alias=@%s deploy=%s mlflow_v=%s run_id=%s dst=%s",
        model, alias, deploy, v, run_id, dst
    )


# -----------------------------------------------------------------------------
# Task 2) triton_load
# - Ask Triton to load model (repository API)
# -----------------------------------------------------------------------------
def triton_load(ti, **_):
    model = ti.xcom_pull(task_ids="materialize_repo", key="model")
    triton = cfg("triton_http_url", required=True)  # ex) http://triton...:8000

    r = requests.post(f"{triton}/v2/repository/models/{model}/load", timeout=5)
    if r.status_code != 200:
        raise RuntimeError(f"load failed: {r.status_code} {r.text}")

    log.info("[W6] triton_load OK model=%s status=%s", model, r.status_code)


# -----------------------------------------------------------------------------
# Task 3) commit_current
# - Write current.json so "what is active" is traceable
# -----------------------------------------------------------------------------
def commit_current(ti, **_):
    model_dir = ti.xcom_pull(task_ids="materialize_repo", key="model_dir")
    payload = {
        "active_version": ti.xcom_pull(task_ids="materialize_repo", key="deploy"),
        "mlflow_version": ti.xcom_pull(task_ids="materialize_repo", key="mlflow_version"),
        "run_id": ti.xcom_pull(task_ids="materialize_repo", key="run_id"),
        "alias": ti.xcom_pull(task_ids="materialize_repo", key="alias"),
        "updated_at_utc": utc_ts(),
    }

    path = os.path.join(model_dir, "current.json")
    with open(path + ".tmp", "w") as f:
        json.dump(payload, f, indent=2)
    os.replace(path + ".tmp", path)

    log.info("[W6] commit_current OK path=%s payload=%s", path, payload)
