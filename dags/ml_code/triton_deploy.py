# dags/ml_code/triton_deploy.py
import os, json, shutil
from datetime import datetime, timezone
from zoneinfo import ZoneInfo

import requests
import mlflow
from mlflow.tracking import MlflowClient

from airflow.models import Variable
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
        raise RuntimeError(f"[Config] missing required key: {key}")
    return default


def utc_ts():
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S")


def build_triton_http_url() -> str:
    # 기본값은 건호님 환경에 맞춤
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


def get_run_params(run_id: str) -> dict:
    uri = cfg("MLFLOW_TRACKING_URI", required=True)
    mlflow.set_tracking_uri(uri)
    c = MlflowClient(tracking_uri=uri)
    run = c.get_run(run_id)
    return dict(run.data.params or {})


def select_by_alias(model_name: str, alias: str):
    uri = cfg("MLFLOW_TRACKING_URI", required=True)
    mlflow.set_tracking_uri(uri)
    c = MlflowClient(tracking_uri=uri)
    mv = c.get_model_version_by_alias(model_name, alias)
    return int(mv.version), mv.run_id


def _atomic_write(path: str, content: str):
    tmp = path + ".tmp"
    with open(tmp, "w") as f:
        f.write(content)
    os.replace(tmp, path)


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
            log.warning("[snapshot] read current.json failed: %s", e)

    ti.xcom_push(key="prev_current", value=prev)
    log.info("[snapshot] model=%s prev=%s", model, prev)


def materialize(
    ti,
    alias: str = "A",
    *,
    run_id: str | None = None,
    shadow: bool = False,
    **_,
):
    """
    promotion: alias -> model_version -> repo/<model>/<version>/
    shadow: run_id -> repo/<model>/<timestamp>/  (registry 없이도 검증 가능)
    """
    base_model = cfg("triton_model_name", required=True)
    model = cfg("triton_model_name_shadow", base_model) if shadow else base_model

    repo = cfg("triton_repo_base", "/models")
    onnx_rel = cfg("triton_onnx_artifact_path", "onnx/model.onnx")

    if run_id:
        chosen_run_id = run_id
        deploy_version = int(datetime.now().strftime("%Y%m%d%H%M%S"))
        mode = "shadow"
    else:
        alias = (alias or "").strip() or cfg("mlflow_alias", "A")
        deploy_version, chosen_run_id = select_by_alias(model, alias)
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
                break
        for cand in outs:
            if "prob" in cand.lower():
                out_prob = cand
                break
        if len(outs) >= 2 and out_label not in outs and out_prob not in outs:
            out_label, out_prob = outs[0], outs[1]

    if n_features <= 0 or n_classes <= 0:
        raise RuntimeError(f"[Triton] invalid n_features/n_classes: {n_features}/{n_classes}")

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

    ti.xcom_push(key="model", value=model)
    ti.xcom_push(key="model_dir", value=model_dir)
    ti.xcom_push(key="deploy_version", value=int(deploy_version))
    ti.xcom_push(key="run_id", value=chosen_run_id)
    ti.xcom_push(key="alias", value=alias)
    ti.xcom_push(key="deploy_mode", value=mode)
    ti.xcom_push(key="n_features", value=n_features)
    ti.xcom_push(key="n_classes", value=n_classes)
    ti.xcom_push(key="onnx_input_name", value=in_name)

    log.info("[materialize] mode=%s model=%s alias=@%s version=%s run_id=%s", mode, model, alias, deploy_version, chosen_run_id)


def triton_load(ti, **_):
    model = ti.xcom_pull(task_ids="materialize_repo", key="model")
    triton = build_triton_http_url()
    r = requests.post(f"{triton}/v2/repository/models/{model}/load", timeout=10)
    if r.status_code != 200:
        raise RuntimeError(f"[load] failed: {r.status_code} {r.text}")
    log.info("[load] OK model=%s", model)


def triton_ready(ti, **_):
    model = ti.xcom_pull(task_ids="materialize_repo", key="model")
    triton = build_triton_http_url()
    r = requests.get(f"{triton}/v2/models/{model}/ready", timeout=5)
    if r.status_code != 200:
        raise RuntimeError(f"[ready] failed: {r.status_code} {r.text}")
    log.info("[ready] OK model=%s", model)


def triton_infer_smoke(ti, **_):
    model = ti.xcom_pull(task_ids="materialize_repo", key="model")
    triton = build_triton_http_url()

    n_features = int(ti.xcom_pull(task_ids="materialize_repo", key="n_features") or 0)
    in_name = ti.xcom_pull(task_ids="materialize_repo", key="onnx_input_name") or "input"
    if n_features <= 0:
        raise RuntimeError("[smoke] missing n_features")

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
        raise RuntimeError(f"[infer] failed: {r.status_code} {r.text}")

    log.info("[smoke] OK model=%s resp=%s", model, r.text[:300])


def commit_current(ti, **_):
    model_dir = ti.xcom_pull(task_ids="materialize_repo", key="model_dir")
    payload = {
        "active_version": ti.xcom_pull(task_ids="materialize_repo", key="deploy_version"),
        "run_id": ti.xcom_pull(task_ids="materialize_repo", key="run_id"),
        "alias": ti.xcom_pull(task_ids="materialize_repo", key="alias"),
        "deploy_mode": ti.xcom_pull(task_ids="materialize_repo", key="deploy_mode"),
        "updated_at_utc": utc_ts(),
    }
    path = os.path.join(model_dir, "current.json")
    tmp = path + ".tmp"
    with open(tmp, "w") as f:
        json.dump(payload, f, indent=2)
    os.replace(tmp, path)
    log.info("[commit] OK path=%s", path)


def rollback_minimal(ti, **_):
    model = ti.xcom_pull(task_ids="materialize_repo", key="model") or cfg("triton_model_name", required=True)
    model_dir = ti.xcom_pull(task_ids="materialize_repo", key="model_dir") or os.path.join(cfg("triton_repo_base", "/models"), model)
    deploy_v = ti.xcom_pull(task_ids="materialize_repo", key="deploy_version")

    prev = ti.xcom_pull(task_ids="snapshot_current", key="prev_current")

    triton = build_triton_http_url()
    log.warning("[ROLLBACK] start model=%s deploy_version=%s prev=%s", model, deploy_v, prev)

    if prev is not None:
        path = os.path.join(model_dir, "current.json")
        tmp = path + ".tmp"
        with open(tmp, "w") as f:
            json.dump(prev, f, indent=2)
        os.replace(tmp, path)
        log.warning("[ROLLBACK] restored current.json")

    if deploy_v is not None:
        ver_dir = os.path.join(model_dir, str(deploy_v))
        if os.path.isdir(ver_dir):
            failed_dir = ver_dir + f".failed_{utc_ts()}"
            os.rename(ver_dir, failed_dir)
            log.warning("[ROLLBACK] moved failed dir: %s -> %s", ver_dir, failed_dir)

    try:
        r = requests.post(f"{triton}/v2/repository/models/{model}/load", timeout=10)
        log.warning("[ROLLBACK] reload status=%s body=%s", r.status_code, r.text[:200])
    except Exception as e:
        log.warning("[ROLLBACK] reload failed: %s", e)

