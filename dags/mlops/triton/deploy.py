from __future__ import annotations

import os
import json
import shutil
from datetime import datetime, timezone
from zoneinfo import ZoneInfo

import requests
import mlflow
from mlflow.tracking import MlflowClient

from airflow.utils.log.logging_mixin import LoggingMixin
from mlops.config import cfg
from mlops.slack import notify

log = LoggingMixin().log


def utc_ts():
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S")


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


def _parse_output_names(raw: str):
    return [x.strip() for x in (raw or "").split(",") if x.strip()]


def get_run_params(run_id: str) -> dict:
    uri = cfg("MLFLOW_TRACKING_URI", required=True)
    c = MlflowClient(tracking_uri=uri)
    run = c.get_run(run_id)
    return dict(run.data.params or {})


def select_by_alias(model_name: str, alias: str):
    uri = cfg("MLFLOW_TRACKING_URI", required=True)
    c = MlflowClient(tracking_uri=uri)
    mv = c.get_model_version_by_alias(model_name, alias)
    return int(mv.version), mv.run_id


def _atomic_write(path: str, content: str):
    tmp = path + ".tmp"
    with open(tmp, "w") as f:
        f.write(content)
    os.replace(tmp, path)


def task_snapshot_current(**context):
    ti = context["ti"]
    model = cfg("triton_model_name", cfg("model_name", "best_model"), required=True)
    repo = cfg("triton_repo_base", "/models")
    model_dir = os.path.join(repo, model)
    path = os.path.join(model_dir, "current.json")

    prev = None
    if os.path.exists(path):
        try:
            with open(path, "r") as f:
                prev = json.load(f)
        except Exception as e:
            log.warning("[SNAP] failed to read current.json: %s", e)

    ti.xcom_push(key="prev_current", value=prev)
    log.info("[SNAP] model=%s prev=%s", model, prev)


def task_materialize_repo(**context):
    ti = context["ti"]

    # 어떤 체인(promotion/shadow)인지 task_id로 판단
    caller = context["task"].task_id
    shadow = "shadow" in caller

    base_model = cfg("triton_model_name", cfg("model_name", "best_model"), required=True)
    model = cfg("triton_model_name_shadow", base_model) if shadow else base_model

    repo = cfg("triton_repo_base", "/models")
    onnx_rel = cfg("triton_onnx_artifact_path", "onnx/model.onnx")

    alias = ti.xcom_pull(task_ids="train_and_evaluate", key="alias") or cfg("mlflow_alias", "A")
    model_name = ti.xcom_pull(task_ids="train_and_evaluate", key="model_name") or cfg("model_name", base_model)
    run_id = ti.xcom_pull(task_ids="train_and_evaluate", key="run_id")

    if shadow:
        if not run_id:
            raise ValueError("shadow deploy requires run_id")
        chosen_run_id = run_id
        deploy_version = int(datetime.now(ZoneInfo("Asia/Seoul")).strftime("%Y%m%d%H%M%S"))
        mode = "shadow"
    else:
        deploy_version, chosen_run_id = select_by_alias(model_name, alias)
        mode = "promote"

    uri = cfg("MLFLOW_TRACKING_URI", required=True)
    mlflow.set_tracking_uri(uri)
    local = mlflow.artifacts.download_artifacts(artifact_uri=f"runs:/{chosen_run_id}/{onnx_rel}")

    params = get_run_params(chosen_run_id)
    n_features = int(params.get("n_features", "0") or "0")
    n_classes = int(params.get("n_classes", "0") or "0")
    in_name = params.get("onnx_input_name", "input")

    outs = _parse_output_names(params.get("onnx_output_names", ""))
    out_label = "label"
    out_prob = "probabilities"
    if outs:
        for cand in outs:
            if "label" in cand.lower():
                out_label = cand
        for cand in outs:
            if "prob" in cand.lower():
                out_prob = cand
        if len(outs) >= 2 and (out_label not in outs or out_prob not in outs):
            out_label, out_prob = outs[0], outs[1]

    if n_features <= 0 or n_classes <= 0:
        raise RuntimeError(f"[TRITON] invalid n_features/n_classes: {n_features}/{n_classes}")

    model_dir = os.path.join(repo, model)
    ver_dir = os.path.join(model_dir, str(deploy_version))
    os.makedirs(ver_dir, exist_ok=True)

    dst = os.path.join(ver_dir, "model.onnx")
    tmp = dst + ".tmp"
    shutil.copyfile(local, tmp)
    os.replace(tmp, dst)

    os.makedirs(model_dir, exist_ok=True)
    config_path = os.path.join(model_dir, "config.pbtxt")
    config_text = build_config_pbtxt(model, in_name, out_prob, out_label, n_features, n_classes)
    _atomic_write(config_path, config_text)

    # xcom
    ti.xcom_push(key="model", value=model)
    ti.xcom_push(key="model_dir", value=model_dir)
    ti.xcom_push(key="deploy_version", value=int(deploy_version))
    ti.xcom_push(key="run_id", value=chosen_run_id)
    ti.xcom_push(key="alias", value=alias)
    ti.xcom_push(key="deploy_mode", value=mode)
    ti.xcom_push(key="n_features", value=n_features)
    ti.xcom_push(key="onnx_input_name", value=in_name)

    notify("Triton materialize OK", mode=mode, model=model, version=deploy_version, run_id=chosen_run_id)
    log.info("[MAT] mode=%s model=%s version=%s run_id=%s", mode, model, deploy_version, chosen_run_id)


def task_triton_load(**context):
    ti = context["ti"]
    model = ti.xcom_pull(key="model") or cfg("triton_model_name", "best_model")
    triton = build_triton_http_url()
    r = requests.post(f"{triton}/v2/repository/models/{model}/load", timeout=10)
    if r.status_code != 200:
        raise RuntimeError(f"[TRITON] load failed: {r.status_code} {r.text[:300]}")
    log.info("[LOAD] model=%s ok", model)


def task_triton_ready(**context):
    ti = context["ti"]
    model = ti.xcom_pull(key="model") or cfg("triton_model_name", "best_model")
    triton = build_triton_http_url()
    r = requests.get(f"{triton}/v2/models/{model}/ready", timeout=5)
    if r.status_code != 200:
        raise RuntimeError(f"[TRITON] ready failed: {r.status_code} {r.text[:300]}")
    log.info("[READY] model=%s ok", model)


def task_triton_infer_smoke(**context):
    ti = context["ti"]
    model = ti.xcom_pull(key="model") or cfg("triton_model_name", "best_model")
    triton = build_triton_http_url()

    n_features = int(ti.xcom_pull(key="n_features") or 0)
    in_name = ti.xcom_pull(key="onnx_input_name") or "input"
    if n_features <= 0:
        raise RuntimeError("smoke: missing n_features")

    payload = {
        "inputs": [
            {
                "name": in_name,
                "shape": [1, n_features],
                "datatype": "FP32",
                "data": [[0.0 for _ in range(n_features)]],
            }
        ]
    }

    r = requests.post(
        f"{triton}/v2/models/{model}/infer",
        headers={"Content-Type": "application/json"},
        json=payload,
        timeout=10,
    )
    if r.status_code != 200:
        raise RuntimeError(f"[TRITON] infer failed: {r.status_code} {r.text[:300]}")

    log.info("[SMOKE] ok model=%s resp=%s", model, r.text[:200])


def task_commit_current(**context):
    ti = context["ti"]
    model_dir = ti.xcom_pull(key="model_dir")
    if not model_dir:
        raise ValueError("commit_current: model_dir missing")

    payload = {
        "active_version": ti.xcom_pull(key="deploy_version"),
        "run_id": ti.xcom_pull(key="run_id"),
        "alias": ti.xcom_pull(key="alias"),
        "deploy_mode": ti.xcom_pull(key="deploy_mode"),
        "updated_at_utc": utc_ts(),
    }

    path = os.path.join(model_dir, "current.json")
    tmp = path + ".tmp"
    with open(tmp, "w") as f:
        json.dump(payload, f, indent=2)
    os.replace(tmp, path)

    notify("commit_current OK", path=path, payload=str(payload)[:200])
    log.info("[COMMIT] %s", payload)


def task_rollback_minimal(**context):
    ti = context["ti"]
    model = ti.xcom_pull(key="model") or cfg("triton_model_name", "best_model")
    model_dir = ti.xcom_pull(key="model_dir") or os.path.join(cfg("triton_repo_base", "/models"), model)
    deploy_v = ti.xcom_pull(key="deploy_version")

    prev = ti.xcom_pull(task_ids=context["task"].upstream_task_ids.pop(), key="prev_current") \
        if context.get("task") else ti.xcom_pull(key="prev_current")

    triton = build_triton_http_url()
    log.warning("[ROLLBACK] start model=%s deploy_v=%s prev=%s", model, deploy_v, prev)

    # restore current.json if possible
    if prev is not None:
        path = os.path.join(model_dir, "current.json")
        tmp = path + ".tmp"
        with open(tmp, "w") as f:
            json.dump(prev, f, indent=2)
        os.replace(tmp, path)
        log.warning("[ROLLBACK] restored current.json")

    # isolate failed version directory
    if deploy_v is not None:
        ver_dir = os.path.join(model_dir, str(deploy_v))
        if os.path.isdir(ver_dir):
            failed_dir = ver_dir + f".failed_{utc_ts()}"
            os.rename(ver_dir, failed_dir)
            log.warning("[ROLLBACK] moved %s -> %s", ver_dir, failed_dir)

    # best effort reload
    try:
        r = requests.post(f"{triton}/v2/repository/models/{model}/load", timeout=10)
        log.warning("[ROLLBACK] reload status=%s body=%s", r.status_code, r.text[:200])
    except Exception as e:
        log.warning("[ROLLBACK] reload exception: %s", e)

    notify("ROLLBACK executed", model=model, deploy_v=deploy_v)

