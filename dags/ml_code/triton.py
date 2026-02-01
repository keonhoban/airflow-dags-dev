# dags/ml_code/triton.py
from __future__ import annotations

import os, json, shutil
from datetime import datetime, timezone
from zoneinfo import ZoneInfo

import requests
import mlflow
from mlflow.tracking import MlflowClient
from airflow.utils.log.logging_mixin import LoggingMixin

from ml_code.config import cfg

log = LoggingMixin().log


def _atomic_write(path: str, content: str):
    tmp = path + ".tmp"
    with open(tmp, "w") as f:
        f.write(content)
    os.replace(tmp, path)


def utc_ts():
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S")


def kst_ts():
    return datetime.now(ZoneInfo("Asia/Seoul")).strftime("%Y%m%dT%H%M%S")


def build_triton_http_url() -> str:
    full = cfg("TRITON_HTTP_URL", None)
    if full:
        return full
    svc = cfg("TRITON_SERVICE", "triton")
    ns = cfg("TRITON_NAMESPACE", "triton-dev")
    port = cfg("TRITON_HTTP_PORT", "8000")
    return f"http://{svc}.{ns}.svc.cluster.local:{port}"


def build_config_pbtxt(model: str, in_name: str, out_prob: str, out_label: str, n_features: int, n_classes: int) -> str:
    return f'''name: "{model}"
platform: "onnxruntime_onnx"
max_batch_size: 0
input [
  {{
    name: "{in_name}"
    data_type: TYPE_FP32
    dims: [ -1, {n_features} ]
  }}
]
output [
  {{
    name: "{out_prob}"
    data_type: TYPE_FP32
    dims: [ -1, {n_classes} ]
  }},
  {{
    name: "{out_label}"
    data_type: TYPE_INT64
    dims: [ -1 ]
  }}
]
'''


def _parse_output_names(raw: str):
    return [x.strip() for x in (raw or "").split(",") if x.strip()]


def _get_run_params(run_id: str) -> dict:
    uri = cfg("MLFLOW_TRACKING_URI", required=True)
    c = MlflowClient(tracking_uri=uri)
    run = c.get_run(run_id)
    return dict(run.data.params or {})


def snapshot_current(ti, **_):
    model = cfg("triton_model_name", required=True)
    repo = cfg("triton_repo_base", "/models")
    model_dir = os.path.join(repo, model)
    path = os.path.join(model_dir, "current.json")

    prev = None
    if os.path.exists(path):
        try:
            with open(path, "r") as f:
                prev = json.load(f)
        except Exception as e:
            log.warning("snapshot_current read failed: %s", e)

    ti.xcom_push(key="prev_current", value=prev)
    log.info("snapshot_current OK model=%s prev=%s", model, prev)


def materialize_shadow_from_run(ti, *, run_id: str, alias: str = "A"):
    """
    ✅ run_id 기반 shadow deploy:
      repo/<model>/<timestamp>/model.onnx + config.pbtxt
    """
    model = cfg("triton_model_name", required=True)
    repo = cfg("triton_repo_base", "/models")
    onnx_rel = cfg("triton_onnx_artifact_path", "onnx/model.onnx")

    deploy_version = int(datetime.now().strftime("%Y%m%d%H%M%S"))

    uri = cfg("MLFLOW_TRACKING_URI", required=True)
    mlflow.set_tracking_uri(uri)
    local = mlflow.artifacts.download_artifacts(artifact_uri=f"runs:/{run_id}/{onnx_rel}")

    params = _get_run_params(run_id)
    n_features = int(params.get("n_features", "0") or "0")
    n_classes = int(params.get("n_classes", "0") or "0")
    in_name = params.get("onnx_input_name", "input")

    outs = _parse_output_names(params.get("onnx_output_names", ""))
    out_label, out_prob = "label", "probabilities"

    if outs:
        for cand in outs:
            if "label" in cand.lower():
                out_label = cand
                break
        for cand in outs:
            if "prob" in cand.lower():
                out_prob = cand
                break
        if len(outs) >= 2 and (out_label not in outs and out_prob not in outs):
            out_label, out_prob = outs[0], outs[1]

    if n_features <= 0 or n_classes <= 0:
        raise RuntimeError(f"invalid n_features/n_classes: {n_features}/{n_classes}")

    model_dir = os.path.join(repo, model)
    ver_dir = os.path.join(model_dir, str(deploy_version))
    os.makedirs(ver_dir, exist_ok=True)

    dst = os.path.join(ver_dir, "model.onnx")
    tmp = dst + ".tmp"
    shutil.copyfile(local, tmp)
    os.replace(tmp, dst)

    os.makedirs(model_dir, exist_ok=True)
    config_path = os.path.join(model_dir, "config.pbtxt")
    _atomic_write(config_path, build_config_pbtxt(model, in_name, out_prob, out_label, n_features, n_classes))

    # XCom
    ti.xcom_push(key="model", value=model)
    ti.xcom_push(key="model_dir", value=model_dir)
    ti.xcom_push(key="deploy_version", value=int(deploy_version))
    ti.xcom_push(key="run_id", value=run_id)
    ti.xcom_push(key="alias", value=alias)
    ti.xcom_push(key="n_features", value=n_features)
    ti.xcom_push(key="onnx_input_name", value=in_name)

    log.info("materialize_shadow OK model=%s version=%s run_id=%s dst=%s", model, deploy_version, run_id, dst)


def triton_load(ti, **_):
    model = ti.xcom_pull(task_ids="triton_materialize_shadow", key="model")
    triton = build_triton_http_url()

    r = requests.post(f"{triton}/v2/repository/models/{model}/load", timeout=10)
    if r.status_code != 200:
        raise RuntimeError(f"load failed: {r.status_code} {r.text}")
    log.info("load OK model=%s", model)


def triton_ready(ti, **_):
    model = ti.xcom_pull(task_ids="triton_materialize_shadow", key="model")
    triton = build_triton_http_url()

    r = requests.get(f"{triton}/v2/models/{model}/ready", timeout=5)
    if r.status_code != 200:
        raise RuntimeError(f"ready failed: {r.status_code} {r.text}")
    log.info("ready OK model=%s", model)


def triton_infer_smoke(ti, **_):
    model = ti.xcom_pull(task_ids="triton_materialize_shadow", key="model")
    triton = build_triton_http_url()

    n_features = int(ti.xcom_pull(task_ids="triton_materialize_shadow", key="n_features") or 0)
    in_name = ti.xcom_pull(task_ids="triton_materialize_shadow", key="onnx_input_name") or "input"
    if n_features <= 0:
        raise RuntimeError("smoke missing n_features")

    payload = {
        "inputs": [{"name": in_name, "shape": [1, n_features], "datatype": "FP32", "data": [[0.0] * n_features]}]
    }
    r = requests.post(f"{triton}/v2/models/{model}/infer", json=payload, timeout=10)
    if r.status_code != 200:
        raise RuntimeError(f"infer failed: {r.status_code} {r.text}")
    log.info("infer OK model=%s resp=%s", model, r.text[:300])


def commit_current(ti, **_):
    model_dir = ti.xcom_pull(task_ids="triton_materialize_shadow", key="model_dir")
    payload = {
        "active_version": ti.xcom_pull(task_ids="triton_materialize_shadow", key="deploy_version"),
        "run_id": ti.xcom_pull(task_ids="triton_materialize_shadow", key="run_id"),
        "alias": ti.xcom_pull(task_ids="triton_materialize_shadow", key="alias"),
        "updated_at_utc": utc_ts(),
    }
    path = os.path.join(model_dir, "current.json")
    tmp = path + ".tmp"
    with open(tmp, "w") as f:
        json.dump(payload, f, indent=2)
    os.replace(tmp, path)
    log.info("commit_current OK path=%s", path)


def rollback_minimal(ti, **_):
    model = ti.xcom_pull(task_ids="triton_materialize_shadow", key="model") or cfg("triton_model_name", required=True)
    repo = cfg("triton_repo_base", "/models")
    model_dir = ti.xcom_pull(task_ids="triton_materialize_shadow", key="model_dir") or os.path.join(repo, model)

    deploy_v = ti.xcom_pull(task_ids="triton_materialize_shadow", key="deploy_version")
    prev = ti.xcom_pull(task_ids="snapshot_current", key="prev_current")  # may be None
    triton = build_triton_http_url()

    log.warning("[ROLLBACK] start model=%s deploy=%s prev=%s", model, deploy_v, prev)

    # 1) current.json 복구
    if prev is not None:
        path = os.path.join(model_dir, "current.json")
        tmp = path + ".tmp"
        with open(tmp, "w") as f:
            json.dump(prev, f, indent=2)
        os.replace(tmp, path)

    # 2) 실패 버전 폴더 격리
    if deploy_v is not None:
        ver_dir = os.path.join(model_dir, str(deploy_v))
        if os.path.isdir(ver_dir):
            os.rename(ver_dir, ver_dir + f".failed_{utc_ts()}")

    # 3) Triton reload best-effort
    try:
        r = requests.post(f"{triton}/v2/repository/models/{model}/load", timeout=10)
        log.warning("[ROLLBACK] reload status=%s body=%s", r.status_code, r.text[:200])
    except Exception as e:
        log.warning("[ROLLBACK] reload failed: %s", e)

