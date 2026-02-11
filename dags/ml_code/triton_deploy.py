# dags/ml_code/triton_deploy.py
from __future__ import annotations

import os
import json
import shutil
import re
from datetime import datetime, timezone

import requests
import mlflow
from mlflow.tracking import MlflowClient

from airflow.models import Variable
from airflow.utils.log.logging_mixin import LoggingMixin

log = LoggingMixin().log


# -----------------------
# Config
# -----------------------
def cfg(key: str, default=None, *, required: bool = False):
    """
    Priority:
    1) Env var
    2) Airflow Variable
    3) default
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
        raise RuntimeError(f"[Config] missing required key: {key}")
    return default


def utc_ts() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S")


def build_triton_http_url() -> str:
    """
    권장: Helm values로 TRITON_HTTP_URL 또는 TRITON_NAMESPACE를 명시하세요.
    """
    full = cfg("TRITON_HTTP_URL", None)
    if full:
        return full

    svc = cfg("TRITON_SERVICE", "triton")

    # env/variable로 제어 가능
    ns = cfg("TRITON_NAMESPACE", None) or cfg("triton_namespace", None)
    if not ns:
        env = (cfg("triton_env", "dev") or "dev").strip()
        ns = f"triton-{env}"

    port = cfg("TRITON_HTTP_PORT", "8000")
    return f"http://{svc}.{ns}.svc.cluster.local:{port}"


# -----------------------
# MLflow helpers
# -----------------------
def get_run_params(run_id: str) -> dict:
    uri = cfg("MLFLOW_TRACKING_URI", required=True)
    mlflow.set_tracking_uri(uri)
    c = MlflowClient(tracking_uri=uri)
    run = c.get_run(run_id)
    return dict(run.data.params or {})


def select_by_alias(model_name: str, alias: str):
    """
    model_name(등록 모델) + alias -> (version, run_id)
    """
    uri = cfg("MLFLOW_TRACKING_URI", required=True)
    mlflow.set_tracking_uri(uri)
    c = MlflowClient(tracking_uri=uri)
    mv = c.get_model_version_by_alias(model_name, alias)
    return int(mv.version), mv.run_id


# -----------------------
# config.pbtxt builder
# -----------------------
def build_config_pbtxt(
    model: str,
    in_name: str,
    out_prob: str,
    out_label: str,
    n_features: int,
    n_classes: int,
) -> str:
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


def _atomic_write(path: str, content: str):
    tmp = path + ".tmp"
    with open(tmp, "w") as f:
        f.write(content)
    os.replace(tmp, path)


def _read_json(path: str):
    if not os.path.exists(path):
        return None
    try:
        with open(path, "r") as f:
            return json.load(f)
    except Exception as e:
        log.warning("[json] read failed: %s path=%s", e, path)
        return None


def _write_json(path: str, payload: dict):
    tmp = path + ".tmp"
    with open(tmp, "w") as f:
        json.dump(payload, f, indent=2)
    os.replace(tmp, path)


def _parse_output_names(raw: str):
    return [x.strip() for x in (raw or "").split(",") if x.strip()]


# -----------------------
# ✅ version_policy specific (핵심 스위치)
# -----------------------
_VERSION_POLICY_RE = re.compile(r"\nversion_policy\s*\{.*?\}\s*\n", re.DOTALL)


def _set_version_policy_specific(config_text: str, version: int) -> str:
    """
    config.pbtxt에 version_policy specific을 강제 주입/치환
    """
    vp = f"""
version_policy {{
  specific {{
    versions: [ {int(version)} ]
  }}
}}
"""
    if _VERSION_POLICY_RE.search(config_text):
        return _VERSION_POLICY_RE.sub(vp + "\n", config_text)
    return config_text.rstrip() + vp + "\n"


def write_or_update_config_policy(model_dir: str, *, version: int):
    """
    기존 config.pbtxt 유지 + version_policy만 specific으로 갱신
    """
    config_path = os.path.join(model_dir, "config.pbtxt")
    if not os.path.exists(config_path):
        raise RuntimeError(f"[config] missing config.pbtxt at {config_path}")

    with open(config_path, "r") as f:
        cur = f.read()

    updated = _set_version_policy_specific(cur, version)
    _atomic_write(config_path, updated)
    log.warning("[config] version_policy specific set to %s (%s)", version, config_path)


# -----------------------
# Airflow tasks
# -----------------------
def snapshot_current(ti, **_):
    model = cfg("triton_model_name", required=True)
    repo = cfg("triton_repo_base", "/models")
    model_dir = os.path.join(repo, model)
    path = os.path.join(model_dir, "current.json")

    prev = _read_json(path)
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
    promotion: alias -> (version, run_id) -> repo/<model>/<version>/
    shadow   : run_id only -> repo/<shadow_model>/<timestamp>/
      ✅ shadow는 main model 디렉토리를 오염시키지 않음
    """
    base_model = cfg("triton_model_name", required=True)
    shadow_model = cfg("triton_model_name_shadow", f"{base_model}_shadow")

    model = shadow_model if shadow else base_model
    repo = cfg("triton_repo_base", "/models")
    onnx_rel = cfg("triton_onnx_artifact_path", "onnx/model.onnx")

    if run_id:
        chosen_run_id = run_id
        deploy_version = int(datetime.now().strftime("%Y%m%d%H%M%S"))
        mode = "shadow"
        alias = (alias or "").strip() or "A"
    else:
        alias = (alias or "").strip() or cfg("mlflow_alias", "A")
        deploy_version, chosen_run_id = select_by_alias(base_model, alias)
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

    # ✅ materialize 시점부터 "이번 버전만 로드"하도록 specific 정책 고정
    write_or_update_config_policy(model_dir, version=int(deploy_version))

    ti.xcom_push(key="model", value=model)
    ti.xcom_push(key="model_dir", value=model_dir)
    ti.xcom_push(key="deploy_version", value=int(deploy_version))
    ti.xcom_push(key="run_id", value=chosen_run_id)
    ti.xcom_push(key="alias", value=alias)
    ti.xcom_push(key="deploy_mode", value=mode)

    # smoke 용
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
    """
    ✅ current.json 기록 + config.pbtxt version_policy specific 갱신
    (current.json은 운영/사람용, config.pbtxt가 Triton 동작의 진짜 스위치)
    """
    model_dir = ti.xcom_pull(task_ids="materialize_repo", key="model_dir")
    deploy_version = int(ti.xcom_pull(task_ids="materialize_repo", key="deploy_version"))
    deploy_mode = ti.xcom_pull(task_ids="materialize_repo", key="deploy_mode")

    # shadow 배포는 current.json 건드리지 않음
    if deploy_mode == "shadow":
        log.warning("[commit] skip current.json write for shadow deploy model_dir=%s", model_dir)
        return

    payload = {
        "active_version": deploy_version,
        "run_id": ti.xcom_pull(task_ids="materialize_repo", key="run_id"),
        "alias": ti.xcom_pull(task_ids="materialize_repo", key="alias"),
        "deploy_mode": deploy_mode,
        "updated_at_utc": utc_ts(),
    }

    path = os.path.join(model_dir, "current.json")
    _write_json(path, payload)

    # ✅ Triton에 "이번 버전만 로드" 강제 (핵심)
    write_or_update_config_policy(model_dir, version=deploy_version)

    log.info("[commit] OK path=%s version=%s", path, deploy_version)


def rollback_minimal(ti, **_):
    """
    실패시 자동 롤백(최소):
    - prev current.json 복구
    - 실패 버전 디렉토리 격리
    - unload/load
    - prev active_version이 있으면 그 버전으로 specific 정책 복구(핵심)
    """
    model = ti.xcom_pull(task_ids="materialize_repo", key="model") or cfg("triton_model_name", required=True)
    repo = cfg("triton_repo_base", "/models")
    model_dir = ti.xcom_pull(task_ids="materialize_repo", key="model_dir") or os.path.join(repo, model)
    deploy_v = ti.xcom_pull(task_ids="materialize_repo", key="deploy_version")
    prev = ti.xcom_pull(task_ids="snapshot_current", key="prev_current")

    triton = build_triton_http_url()
    log.warning("[ROLLBACK] start model=%s deploy_version=%s prev=%s", model, deploy_v, prev)

    # restore current.json
    if prev is not None:
        path = os.path.join(model_dir, "current.json")
        _write_json(path, prev)
        log.warning("[ROLLBACK] restored current.json")

        # ✅ prev active_version으로 specific 정책 복구
        try:
            av = prev.get("active_version")
            if av is not None:
                write_or_update_config_policy(model_dir, version=int(av))
        except Exception as e:
            log.warning("[ROLLBACK] policy restore failed: %s", e)

    # quarantine failed dir
    if deploy_v is not None:
        ver_dir = os.path.join(model_dir, str(deploy_v))
        if os.path.isdir(ver_dir):
            failed_dir = ver_dir + f".failed_{utc_ts()}"
            os.rename(ver_dir, failed_dir)
            log.warning("[ROLLBACK] moved failed dir: %s -> %s", ver_dir, failed_dir)

    # unload/load
    try:
        requests.post(f"{triton}/v2/repository/models/{model}/unload", timeout=10)
        r = requests.post(f"{triton}/v2/repository/models/{model}/load", timeout=10)
        log.warning("[ROLLBACK] reload status=%s body=%s", r.status_code, r.text[:200])
    except Exception as e:
        log.warning("[ROLLBACK] reload failed: %s", e)


def rollback_manual(model: str | None = None, deploy_version: int | None = None):
    """
    Manual rollback (ops/interview consistent):
    - deploy_version 주면:
        1) current.json active_version 갱신
        2) config.pbtxt version_policy specific = deploy_version
        3) unload/load
    - deploy_version 없으면:
        1) current.json 유지
        2) unload/load
    """
    model = model or cfg("triton_model_name", required=True)
    repo = cfg("triton_repo_base", "/models")
    model_dir = os.path.join(repo, model)
    os.makedirs(model_dir, exist_ok=True)

    path = os.path.join(model_dir, "current.json")
    cur = _read_json(path) or {}

    if deploy_version is not None:
        cur["active_version"] = int(deploy_version)
        cur["updated_at_utc"] = utc_ts()
        _write_json(path, cur)
        log.warning("[ROLLBACK_MANUAL] forced active_version=%s path=%s", deploy_version, path)

        # ✅ Triton 동작을 실제로 바꾸는 핵심
        write_or_update_config_policy(model_dir, version=int(deploy_version))
    else:
        log.warning("[ROLLBACK_MANUAL] no deploy_version -> reload only path=%s", path)

    triton = build_triton_http_url()
    # ✅ unload/load가 가장 확실
    requests.post(f"{triton}/v2/repository/models/{model}/unload", timeout=10)
    r = requests.post(f"{triton}/v2/repository/models/{model}/load", timeout=10)
    if r.status_code != 200:
        raise RuntimeError(f"[ROLLBACK_MANUAL] reload failed: {r.status_code} {r.text}")
    log.warning("[ROLLBACK_MANUAL] reload OK status=%s", r.status_code)

