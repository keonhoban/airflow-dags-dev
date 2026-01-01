# dags/ml_code/triton_deploy.py

import os, json, shutil, time
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
    dims: [ -1, 4 ]
  }}
]

output [
  {{
    name: "probabilities"
    data_type: TYPE_FP32
    dims: [ -1, 3 ]
  }},
  {{
    name: "label"
    data_type: TYPE_INT64
    dims: [ -1 ]
  }}
]
"""


# -----------------------
# Config helpers
# -----------------------
def cfg(key: str, default=None, *, required: bool = False):
    """
    우선순위: ENV > Airflow Variable > default
    """
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


def build_triton_http_url() -> str:
    svc = cfg("TRITON_SERVICE", "triton")
    ns = cfg("TRITON_NAMESPACE", "triton-dev")
    port = cfg("TRITON_HTTP_PORT", "8000")

    full = cfg("TRITON_HTTP_URL", None)
    if full:
        return full

    return f"http://{svc}.{ns}.svc.cluster.local:{port}"


# -----------------------
# MLflow selection
# -----------------------
def select_by_alias(model_name: str, alias: str):
    uri = cfg("MLFLOW_TRACKING_URI", required=True)
    mlflow.set_tracking_uri(uri)
    c = MlflowClient(tracking_uri=uri)

    mv = c.get_model_version_by_alias(model_name, alias)
    return int(mv.version), mv.run_id


# -----------------------
# Tasks
# -----------------------
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
            log.warning("[W6] snapshot_current: failed to read current.json (%s). treat as None", e)

    ti.xcom_push(key="prev_current", value=prev)
    log.info("[W6] snapshot_current OK model=%s path=%s prev=%s", model, path, prev)


def materialize(ti, alias: str = "A", **_):
    model = cfg("triton_model_name", required=True)
    repo = cfg("triton_repo_base", "/models")
    onnx_rel = cfg("triton_onnx_artifact_path", "onnx/model.onnx")

    alias = (alias or "").strip() or cfg("mlflow_alias", "A")
    v, run_id = select_by_alias(model, alias)

    model_dir = os.path.join(repo, model)
    ver_dir = os.path.join(model_dir, str(v))
    os.makedirs(ver_dir, exist_ok=True)

    local = mlflow.artifacts.download_artifacts(artifact_uri=f"runs:/{run_id}/{onnx_rel}")
    dst = os.path.join(ver_dir, "model.onnx")
    shutil.copyfile(local, dst)

    os.makedirs(model_dir, exist_ok=True)
    config_path = os.path.join(model_dir, "config.pbtxt")
    with open(config_path, "w") as f:
        f.write(CONFIG_TEMPLATE.format(model=model))

    ti.xcom_push(key="model", value=model)
    ti.xcom_push(key="model_dir", value=model_dir)
    ti.xcom_push(key="deploy_version", value=v)
    ti.xcom_push(key="run_id", value=run_id)
    ti.xcom_push(key="alias", value=alias)

    log.info("[W6] materialize OK model=%s alias=@%s version=%s dst=%s", model, alias, v, dst)
    log.info("[W6] config.pbtxt created/updated path=%s", config_path)


def triton_load(ti, **_):
    model = ti.xcom_pull(task_ids="materialize_repo", key="model")
    alias = ti.xcom_pull(task_ids="materialize_repo", key="alias")
    triton = build_triton_http_url()

    log.info("[W6] triton_load model=%s alias=@%s triton=%s", model, alias, triton)

    r = requests.post(f"{triton}/v2/repository/models/{model}/load", timeout=10)
    if r.status_code != 200:
        raise RuntimeError(f"[W6] load failed: {r.status_code} {r.text}")

    log.info("[W6] load OK model=%s", model)


def triton_ready(ti, **_):
    model = ti.xcom_pull(task_ids="materialize_repo", key="model")
    triton = build_triton_http_url()

    r = requests.get(f"{triton}/v2/models/{model}/ready", timeout=5)
    if r.status_code != 200:
        raise RuntimeError(f"[W6] ready failed: {r.status_code} {r.text}")

    log.info("[W6] ready OK model=%s", model)


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
        headers={"Content-Type": "application/json"},
        json=payload,
        timeout=10,
    )
    if r.status_code != 200:
        raise RuntimeError(f"[W6] infer failed: {r.status_code} {r.text}")

    log.info("[W6] infer smoke OK model=%s resp=%s", model, r.json())


def commit_current(ti, **_):
    model_dir = ti.xcom_pull(task_ids="materialize_repo", key="model_dir")
    payload = {
        "active_version": ti.xcom_pull(task_ids="materialize_repo", key="deploy_version"),
        "run_id": ti.xcom_pull(task_ids="materialize_repo", key="run_id"),
        "alias": ti.xcom_pull(task_ids="materialize_repo", key="alias"),
        "updated_at_utc": utc_ts(),
    }

    path = os.path.join(model_dir, "current.json")
    with open(path + ".tmp", "w") as f:
        json.dump(payload, f, indent=2)
    os.replace(path + ".tmp", path)

    log.info("[W6] commit_current OK path=%s payload=%s", path, payload)


def rollback_minimal(ti, **_):
    model = ti.xcom_pull(task_ids="materialize_repo", key="model") or cfg("triton_model_name", required=True)
    model_dir = ti.xcom_pull(task_ids="materialize_repo", key="model_dir") or os.path.join(cfg("triton_repo_base", "/models"), model)
    deploy_v = ti.xcom_pull(task_ids="materialize_repo", key="deploy_version")

    # ✅ 방어: snapshot task가 실패/스킵이면 prev가 None일 수 있음
    prev = ti.xcom_pull(task_ids="snapshot_current", key="prev_current")  # may be None

    triton = build_triton_http_url()
    log.warning("[W6][ROLLBACK] start model=%s deploy_version=%s prev_current=%s", model, deploy_v, prev)

    # (1) current.json 복구 (가능한 경우만)
    if prev is not None:
        path = os.path.join(model_dir, "current.json")
        with open(path + ".tmp", "w") as f:
            json.dump(prev, f, indent=2)
        os.replace(path + ".tmp", path)
        log.warning("[W6][ROLLBACK] restored current.json path=%s", path)
    else:
        log.warning("[W6][ROLLBACK] prev_current is None (no previous). skip current.json restore.")

    # (2) 실패한 새 버전 폴더 격리
    if deploy_v is not None:
        ver_dir = os.path.join(model_dir, str(deploy_v))
        if os.path.isdir(ver_dir):
            failed_dir = ver_dir + f".failed_{utc_ts()}"
            try:
                os.replace(ver_dir, failed_dir)
                log.warning("[W6][ROLLBACK] moved failed version dir %s -> %s", ver_dir, failed_dir)
            except Exception as e:
                log.warning("[W6][ROLLBACK] failed to move version dir (%s). keep as-is. err=%s", ver_dir, e)

    # (3) Triton unload -> load (best-effort)
    try:
        r1 = requests.post(f"{triton}/v2/repository/models/{model}/unload", timeout=10)
        log.warning("[W6][ROLLBACK] unload status=%s body=%s", r1.status_code, (r1.text or "")[:200])
    except Exception as e:
        log.warning("[W6][ROLLBACK] unload request failed: %s", e)

    time.sleep(1)

    try:
        r2 = requests.post(f"{triton}/v2/repository/models/{model}/load", timeout=10)
        log.warning("[W6][ROLLBACK] load status=%s body=%s", r2.status_code, (r2.text or "")[:200])
    except Exception as e:
        log.warning("[W6][ROLLBACK] load request failed: %s", e)

    # (4) ready best-effort
    try:
        r3 = requests.get(f"{triton}/v2/models/{model}/ready", timeout=5)
        log.warning("[W6][ROLLBACK] ready status=%s", r3.status_code)
    except Exception as e:
        log.warning("[W6][ROLLBACK] ready check failed: %s", e)

    log.warning("[W6][ROLLBACK] done model=%s", model)
