# dags/ml_code/triton_deploy.py

import os, json, shutil
from datetime import datetime, timezone
from zoneinfo import ZoneInfo

import requests
import mlflow
from mlflow.tracking import MlflowClient

from airflow.sdk import Variable
from airflow.utils.log.logging_mixin import LoggingMixin

log = LoggingMixin().log


# -----------------------
# Triton config template
# -----------------------
CONFIG_TEMPLATE = """\
name: "{model}"
platform: "onnxruntime_onnx"

max_batch_size: 0

input [
  {{
    name: "input"
    data_type: TYPE_FP32
    dims: [ 4 ]
  }}
]

output [
  {{
    name: "probabilities"
    data_type: TYPE_FP32
    dims: [ 3 ]
  }},
  {{
    name: "label"
    data_type: TYPE_INT64
    dims: [ 1 ]
  }}
]
"""


# -----------------------
# Config helpers
# -----------------------
def cfg(key: str, default=None, *, required: bool = False):
    v = os.getenv(key)
    if v and str(v).strip():
        return v

    try:
        v = Variable.get(key, default_var=str(default) if default is not None else None)
        if v and str(v).strip():
            return v
    except Exception:
        pass

    if required:
        raise RuntimeError(f"[Config] missing required key: {key}")
    return default


def utc_ts():
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S")


def build_triton_http_url() -> str:
    svc = cfg("TRITON_SERVICE", "triton")
    ns = cfg("TRITON_NAMESPACE", "triton-dev")
    port = cfg("TRITON_HTTP_PORT", "8000")

    full = cfg("TRITON_HTTP_URL")
    if full:
        return full

    return f"http://{svc}.{ns}.svc.cluster.local:{port}"


# -----------------------
# MLflow selection
# -----------------------
def select_by_alias(model_name: str, alias: str):
    uri = cfg("MLFLOW_TRACKING_URI", required=True)
    mlflow.set_tracking_uri(uri)
    client = MlflowClient(tracking_uri=uri)

    mv = client.get_model_version_by_alias(model_name, alias)
    return int(mv.version), mv.run_id


# -----------------------
# Tasks
# -----------------------
def materialize(ti, alias: str = "A", **_):
    model = cfg("triton_model_name", required=True)
    repo = cfg("triton_repo_base", "/models")
    onnx_rel = cfg("triton_onnx_artifact_path", "onnx/model.onnx")

    alias = (alias or "").strip() or cfg("mlflow_alias", "A")

    version, run_id = select_by_alias(model, alias)

    model_dir = os.path.join(repo, model)
    ver_dir = os.path.join(model_dir, str(version))
    os.makedirs(ver_dir, exist_ok=True)

    # ONNX 복사
    local = mlflow.artifacts.download_artifacts(
        artifact_uri=f"runs:/{run_id}/{onnx_rel}"
    )
    shutil.copyfile(local, os.path.join(ver_dir, "model.onnx"))

    # config.pbtxt (model root)
    config_path = os.path.join(model_dir, "config.pbtxt")
    if not os.path.exists(config_path):
        with open(config_path, "w") as f:
            f.write(CONFIG_TEMPLATE.format(model=model))

    ti.xcom_push(key="model", value=model)
    ti.xcom_push(key="model_dir", value=model_dir)
    ti.xcom_push(key="version", value=version)
    ti.xcom_push(key="run_id", value=run_id)
    ti.xcom_push(key="alias", value=alias)

    log.info("[W6] materialize OK model=%s alias=@%s version=%s", model, alias, version)


def triton_load(ti, **_):
    model = ti.xcom_pull(task_ids="materialize_repo", key="model")
    triton = build_triton_http_url()

    r = requests.post(
        f"{triton}/v2/repository/models/{model}/load", timeout=10
    )
    if r.status_code != 200:
        raise RuntimeError(f"load failed: {r.status_code} {r.text}")

    log.info("[W6] triton_load OK model=%s", model)


def triton_ready(ti, **_):
    model = ti.xcom_pull(task_ids="materialize_repo", key="model")
    triton = build_triton_http_url()

    r = requests.get(f"{triton}/v2/models/{model}/ready", timeout=5)
    if r.status_code != 200:
        raise RuntimeError(f"ready failed: {r.status_code} {r.text}")

    log.info("[W6] triton_ready OK model=%s", model)


def triton_infer_smoke(ti, **_):
    model = ti.xcom_pull(task_ids="materialize_repo", key="model")
    triton = build_triton_http_url()

    payload = {
        "inputs": [
            {
                "name": "input",
                "shape": [1, 4],
                "datatype": "FP32",
                "data": [[5.1, 3.5, 1.4, 0.2]],
            }
        ]
    }

    r = requests.post(
        f"{triton}/v2/models/{model}/infer",
        json=payload,
        headers={"Content-Type": "application/json"},
        timeout=10,
    )

    if r.status_code != 200:
        raise RuntimeError(f"infer failed: {r.status_code} {r.text}")

    log.info("[W6] triton_infer_smoke OK resp=%s", r.json())


def commit_current(ti, **_):
    model_dir = ti.xcom_pull(task_ids="materialize_repo", key="model_dir")

    payload = {
        "active_version": ti.xcom_pull(task_ids="materialize_repo", key="version"),
        "run_id": ti.xcom_pull(task_ids="materialize_repo", key="run_id"),
        "alias": ti.xcom_pull(task_ids="materialize_repo", key="alias"),
        "updated_at_utc": utc_ts(),
    }

    path = os.path.join(model_dir, "current.json")
    with open(path + ".tmp", "w") as f:
        json.dump(payload, f, indent=2)
    os.replace(path + ".tmp", path)

    log.info("[W6] commit_current OK path=%s", path)
