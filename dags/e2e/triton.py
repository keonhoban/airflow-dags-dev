from __future__ import annotations

import os, json, shutil
import requests
import mlflow
from mlflow.tracking import MlflowClient
from airflow.utils.log.logging_mixin import LoggingMixin

from e2e.config import cfg
from e2e.utils import utc_ts, atomic_json_write

log = LoggingMixin().log

def _triton_url() -> str:
    full = cfg("TRITON_HTTP_URL", None)
    if full:
        return full
    svc = cfg("TRITON_SERVICE", "triton")
    ns = cfg("TRITON_NAMESPACE", "triton-dev")
    port = cfg("TRITON_HTTP_PORT", "8000")
    return f"http://{svc}.{ns}.svc.cluster.local:{port}"

def _mlflow_client():
    uri = cfg("MLFLOW_TRACKING_URI", required=True)
    mlflow.set_tracking_uri(uri)
    return MlflowClient(tracking_uri=uri)

def _parse_out_names(raw: str):
    return [x.strip() for x in (raw or "").split(",") if x.strip()]

def _build_config_pbtxt(model: str, in_name: str, out_prob: str, out_label: str, n_features: int, n_classes: int) -> str:
    return f'''name: "{model}"
platform: "onnxruntime_onnx"
max_batch_size: 0
input [ {{
  name: "{in_name}"
  data_type: TYPE_FP32
  dims: [ -1, {n_features} ]
}} ]
output [
  {{
    name: "{out_label}"
    data_type: TYPE_INT64
    dims: [ -1 ]
  }},
  {{
    name: "{out_prob}"
    data_type: TYPE_FP32
    dims: [ -1, {n_classes} ]
  }}
]
'''

def snapshot_current(**context):
    ti = context["ti"]
    model = cfg("model_name", "best_model")
    repo = cfg("triton_repo_base", "/models")
    model_dir = os.path.join(repo, model)
    path = os.path.join(model_dir, "current.json")

    prev = None
    if os.path.exists(path):
        try:
            with open(path, "r") as f:
                prev = json.load(f)
        except Exception as e:
            log.warning("[TRITON] snapshot read failed: %s", e)

    ti.xcom_push(key="prev_current", value=prev)
    log.info("[TRITON] snapshot_current OK prev=%s", prev)

def _materialize_common(ti, *, model: str, run_id: str, deploy_version: int, mode: str, alias: str | None):
    repo = cfg("triton_repo_base", "/models")
    onnx_rel = cfg("triton_onnx_artifact_path", "onnx/model.onnx")

    uri = cfg("MLFLOW_TRACKING_URI", required=True)
    mlflow.set_tracking_uri(uri)
    local = mlflow.artifacts.download_artifacts(artifact_uri=f"runs:/{run_id}/{onnx_rel}")

    # read params for config automation
    c = _mlflow_client()
    run = c.get_run(run_id)
    params = dict(run.data.params or {})

    n_features = int(params.get("n_features", "0") or "0")
    n_classes = int(params.get("n_classes", "0") or "0")
    in_name = params.get("onnx_input_name", "input")
    outs = _parse_out_names(params.get("onnx_output_names", ""))

    out_label = "label"
    out_prob = "probabilities"
    if outs:
        for cand in outs:
            if "label" in cand.lower():
                out_label = cand; break
        for cand in outs:
            if "prob" in cand.lower():
                out_prob = cand; break
        if len(outs) >= 2 and (out_label not in outs or out_prob not in outs):
            out_label, out_prob = outs[0], outs[1]

    if n_features <= 0 or n_classes <= 0:
        raise RuntimeError(f"[TRITON] invalid params: n_features={n_features}, n_classes={n_classes}")

    model_dir = os.path.join(repo, model)
    ver_dir = os.path.join(model_dir, str(deploy_version))
    os.makedirs(ver_dir, exist_ok=True)

    dst = os.path.join(ver_dir, "model.onnx")
    tmp = dst + ".tmp"
    shutil.copyfile(local, tmp)
    os.replace(tmp, dst)

    config_text = _build_config_pbtxt(model, in_name, out_prob, out_label, n_features, n_classes)
    os.makedirs(model_dir, exist_ok=True)
    with open(os.path.join(model_dir, "config.pbtxt.tmp"), "w") as f:
        f.write(config_text)
    os.replace(os.path.join(model_dir, "config.pbtxt.tmp"), os.path.join(model_dir, "config.pbtxt"))

    ti.xcom_push(key="model", value=model)
    ti.xcom_push(key="model_dir", value=model_dir)
    ti.xcom_push(key="deploy_version", value=int(deploy_version))
    ti.xcom_push(key="run_id", value=run_id)
    ti.xcom_push(key="deploy_mode", value=mode)
    ti.xcom_push(key="alias", value=alias or "")
    ti.xcom_push(key="n_features", value=n_features)
    ti.xcom_push(key="onnx_input_name", value=in_name)

    log.info("[TRITON] materialize OK mode=%s model=%s version=%s run_id=%s", mode, model, deploy_version, run_id)

def materialize_promote(**context):
    ti = context["ti"]
    model = cfg("model_name", "best_model")
    alias = ti.xcom_pull(task_ids="train", key="alias") or cfg("mlflow_alias", "A")

    c = _mlflow_client()
    mv = c.get_model_version_by_alias(model, alias)
    version = int(mv.version)
    run_id = mv.run_id

    _materialize_common(ti, model=model, run_id=run_id, deploy_version=version, mode="promote", alias=alias)

def materialize_shadow(**context):
    ti = context["ti"]
    model = cfg("model_name", "best_model")
    run_id = ti.xcom_pull(task_ids="train", key="run_id")
    if not run_id:
        raise ValueError("[TRITON] shadow needs run_id")
    deploy_version = int(__import__("datetime").datetime.now().strftime("%Y%m%d%H%M%S"))
    _materialize_common(ti, model=model, run_id=run_id, deploy_version=deploy_version, mode="shadow", alias=None)

def triton_load(**context):
    ti = context["ti"]
    model = ti.xcom_pull(key="model", task_ids=["materialize_promote","materialize_shadow"])
    url = _triton_url()
    r = requests.post(f"{url}/v2/repository/models/{model}/load", timeout=10)
    if r.status_code != 200:
        raise RuntimeError(f"[TRITON] load failed: {r.status_code} {r.text[:300]}")
    log.info("[TRITON] load OK model=%s", model)

def triton_ready(**context):
    ti = context["ti"]
    model = ti.xcom_pull(key="model", task_ids=["materialize_promote","materialize_shadow"])
    url = _triton_url()
    r = requests.get(f"{url}/v2/models/{model}/ready", timeout=5)
    if r.status_code != 200:
        raise RuntimeError(f"[TRITON] ready failed: {r.status_code} {r.text[:300]}")
    log.info("[TRITON] ready OK model=%s", model)

def triton_infer_smoke(**context):
    ti = context["ti"]
    model = ti.xcom_pull(key="model", task_ids=["materialize_promote","materialize_shadow"])
    n_features = int(ti.xcom_pull(key="n_features", task_ids=["materialize_promote","materialize_shadow"]) or 0)
    in_name = ti.xcom_pull(key="onnx_input_name", task_ids=["materialize_promote","materialize_shadow"]) or "input"
    if n_features <= 0:
        raise RuntimeError("[TRITON] smoke missing n_features")

    url = _triton_url()
    payload = {"inputs":[{"name":in_name,"shape":[1,n_features],"datatype":"FP32","data":[[0.0]*n_features]}]}
    r = requests.post(f"{url}/v2/models/{model}/infer", json=payload, timeout=10)
    if r.status_code != 200:
        raise RuntimeError(f"[TRITON] infer failed: {r.status_code} {r.text[:300]}")
    log.info("[TRITON] smoke OK model=%s resp=%s", model, r.text[:300])

def commit_current(**context):
    ti = context["ti"]
    model_dir = ti.xcom_pull(key="model_dir", task_ids=["materialize_promote","materialize_shadow"])
    payload = {
        "active_version": ti.xcom_pull(key="deploy_version", task_ids=["materialize_promote","materialize_shadow"]),
        "run_id": ti.xcom_pull(key="run_id", task_ids=["materialize_promote","materialize_shadow"]),
        "alias": ti.xcom_pull(key="alias", task_ids=["materialize_promote","materialize_shadow"]),
        "deploy_mode": ti.xcom_pull(key="deploy_mode", task_ids=["materialize_promote","materialize_shadow"]),
        "updated_at_utc": utc_ts(),
    }
    atomic_json_write(os.path.join(model_dir, "current.json"), payload)
    log.info("[TRITON] commit_current OK payload=%s", payload)

def rollback_minimal(**context):
    ti = context["ti"]
    model = cfg("model_name", "best_model")
    repo = cfg("triton_repo_base", "/models")
    model_dir = os.path.join(repo, model)

    prev = ti.xcom_pull(task_ids="snapshot_current", key="prev_current")
    deploy_v = ti.xcom_pull(key="deploy_version", task_ids=["materialize_promote","materialize_shadow"])

    # restore current.json if possible
    if prev is not None:
        atomic_json_write(os.path.join(model_dir, "current.json"), prev)
        log.warning("[TRITON][RB] restored current.json")
    else:
        log.warning("[TRITON][RB] prev_current is None; skip restore")

    # quarantine failed dir
    if deploy_v is not None:
        ver_dir = os.path.join(model_dir, str(deploy_v))
        if os.path.isdir(ver_dir):
            os.rename(ver_dir, ver_dir + f".failed_{utc_ts()}")
            log.warning("[TRITON][RB] quarantined version dir=%s", ver_dir)

    # best-effort reload
    try:
        url = _triton_url()
        requests.post(f"{url}/v2/repository/models/{model}/load", timeout=10)
    except Exception as e:
        log.warning("[TRITON][RB] reload failed: %s", e)

