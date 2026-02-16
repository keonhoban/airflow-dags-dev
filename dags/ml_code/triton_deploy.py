# dags/ml_code/triton_deploy.py
from __future__ import annotations

import os
import json
import shutil
import re
import time
from datetime import datetime, timezone
from typing import Optional

import requests
import mlflow
from mlflow.tracking import MlflowClient

from airflow.models import Variable
from airflow.utils.log.logging_mixin import LoggingMixin

log = LoggingMixin().log


# -----------------------
# Config helpers
# -----------------------
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


def utc_ts() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S")


def now_ts() -> str:
    # shadow 버전용 timestamp (Triton repo 폴더 이름)
    return datetime.now().strftime("%Y%m%d%H%M%S")


def build_triton_http_url() -> str:
    full = cfg("TRITON_HTTP_URL", None)
    if full:
        return full
    svc = cfg("TRITON_SERVICE", "triton")
    ns = cfg("TRITON_NAMESPACE", "triton-dev")
    port = cfg("TRITON_HTTP_PORT", "8000")
    return f"http://{svc}.{ns}.svc.cluster.local:{port}"


# -----------------------
# MLflow helpers
# -----------------------
def _mlflow_client() -> MlflowClient:
    uri = cfg("MLFLOW_TRACKING_URI", required=True)
    mlflow.set_tracking_uri(uri)
    return MlflowClient(tracking_uri=uri)


def get_run_params(run_id: str) -> dict:
    c = _mlflow_client()
    run = c.get_run(run_id)
    return dict(run.data.params or {})


def select_by_alias(model_name: str, alias: str):
    """
    ✅ Promotion SSOT = MLflow Registry version (int)
    """
    c = _mlflow_client()
    mv = c.get_model_version_by_alias(model_name, alias)
    return int(mv.version), str(mv.run_id)


def run_id_by_version(model_name: str, version: int) -> str:
    c = _mlflow_client()
    mv = c.get_model_version(model_name, str(int(version)))
    return str(mv.run_id)


# -----------------------
# Triton config.pbtxt builder
# -----------------------
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


def _atomic_write(path: str, content: str):
    tmp = path + ".tmp"
    with open(tmp, "w") as f:
        f.write(content)
    os.replace(tmp, path)


# -----------------------
# version_policy (핵심 스위치)
# -----------------------
_VERSION_POLICY_RE = re.compile(r"\nversion_policy\s*\{.*?\}\s*\n", re.DOTALL)


def _set_version_policy_specific(config_text: str, version: int) -> str:
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
    config_path = os.path.join(model_dir, "config.pbtxt")
    if not os.path.exists(config_path):
        raise RuntimeError(f"[config] missing config.pbtxt at {config_path}")

    with open(config_path, "r") as f:
        cur = f.read()

    updated = _set_version_policy_specific(cur, version)
    _atomic_write(config_path, updated)
    log.warning("[config] version_policy specific set to %s (%s)", version, config_path)


# -----------------------
# Tasks
# -----------------------
def snapshot_current(ti, **_):
    """
    ✅ Promotion 롤백용 snapshot은 base_model(best_model) 기준만 저장
    (shadow는 base_model과 분리되므로 snapshot/current.json 건드리지 않음)
    """
    base_model = cfg("triton_model_name", required=True)
    repo = cfg("triton_repo_base", "/models")
    model_dir = os.path.join(repo, base_model)
    path = os.path.join(model_dir, "current.json")

    prev = None
    if os.path.exists(path):
        try:
            with open(path, "r") as f:
                prev = json.load(f)
        except Exception as e:
            log.warning("[snapshot] read current.json failed: %s", e)

    ti.xcom_push(key="prev_current", value=prev)
    log.info("[snapshot] model=%s prev=%s", base_model, prev)


def materialize(
    ti,
    alias: str = "A",
    *,
    run_id: str | None = None,
    shadow: bool = False,
    **_,
):
    """
    ✅ B안(추천): promotion/shadow 완전 분리

    - promotion:
      - model = base_model (best_model)
      - deploy_version = MLflow registry version (int, 예: 33)
      - Triton version_policy specific = deploy_version
    - shadow:
      - model = shadow_model (best_model_shadow)
      - deploy_version = timestamp (예: 20260216091939)
      - run_id 기반 artifact 다운로드
      - Triton version_policy specific = deploy_version (shadow 모델 안에서만)
    """
    base_model = cfg("triton_model_name", required=True)
    repo = cfg("triton_repo_base", "/models")
    onnx_rel = cfg("triton_onnx_artifact_path", "onnx/model.onnx")

    shadow_model = cfg("triton_model_name_shadow", f"{base_model}_shadow")

    if shadow:
        if not run_id:
            raise ValueError("[materialize] shadow=True requires run_id")
        model = shadow_model
        chosen_run_id = run_id
        deploy_version = int(now_ts())  # timestamp
        mode = "shadow"
        used_alias = (alias or "").strip() or "A"
    else:
        used_alias = (alias or "").strip() or cfg("mlflow_alias", "A")
        model = base_model
        deploy_version, chosen_run_id = select_by_alias(base_model, used_alias)  # MLflow version
        mode = "promote"

    # artifact download
    uri = cfg("MLFLOW_TRACKING_URI", required=True)
    mlflow.set_tracking_uri(uri)
    local = mlflow.artifacts.download_artifacts(
        artifact_uri=f"runs:/{chosen_run_id}/{onnx_rel}"
    )

    # params -> shape/meta
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

    # copy model.onnx (atomic)
    dst = os.path.join(ver_dir, "model.onnx")
    tmp = dst + ".tmp"
    shutil.copyfile(local, tmp)
    os.replace(tmp, dst)

    # config.pbtxt write (atomic)
    os.makedirs(model_dir, exist_ok=True)
    _atomic_write(
        os.path.join(model_dir, "config.pbtxt"),
        build_config_pbtxt(model, in_name, out_prob, out_label, n_features, n_classes),
    )

    # ✅ 핵심: load 전에 version_policy specific을 반드시 고정
    write_or_update_config_policy(model_dir, version=int(deploy_version))
    log.warning("[materialize] version_policy set BEFORE load: model=%s version=%s mode=%s", model, deploy_version, mode)

    # xcom
    ti.xcom_push(key="model", value=model)
    ti.xcom_push(key="model_dir", value=model_dir)
    ti.xcom_push(key="deploy_version", value=int(deploy_version))
    ti.xcom_push(key="run_id", value=chosen_run_id)
    ti.xcom_push(key="alias", value=used_alias)
    ti.xcom_push(key="deploy_mode", value=mode)
    ti.xcom_push(key="n_features", value=n_features)
    ti.xcom_push(key="n_classes", value=n_classes)
    ti.xcom_push(key="onnx_input_name", value=in_name)

    log.info(
        "[materialize] mode=%s model=%s alias=@%s version=%s run_id=%s",
        mode, model, used_alias, deploy_version, chosen_run_id
    )


def triton_load(ti, **_):
    model = ti.xcom_pull(task_ids="materialize_repo", key="model")
    triton = build_triton_http_url()

    # ✅ 정석: unload -> load (config/version_policy 변경 확실 반영)
    try:
        requests.post(f"{triton}/v2/repository/models/{model}/unload", timeout=10)
    except Exception:
        pass

    time.sleep(0.5)

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
        "inputs": [{
            "name": in_name,
            "shape": [1, n_features],
            "datatype": "FP32",
            "data": [[0.0 for _ in range(n_features)]],
        }]
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
    ✅ Promotion 성공 시 base_model(best_model) 기준만 current.json 기록
    shadow(best_model_shadow)는 current.json을 가지지 않음(운영 혼선 방지)
    """
    model = ti.xcom_pull(task_ids="materialize_repo", key="model")
    base_model = cfg("triton_model_name", required=True)

    deploy_mode = ti.xcom_pull(task_ids="materialize_repo", key="deploy_mode")
    if deploy_mode == "shadow":
        log.warning("[commit] skip current.json for shadow model=%s", model)
        return

    if model != base_model:
        # 안전장치: promote인데 base_model이 아니면 이상 상태
        raise RuntimeError(f"[commit] unexpected promote model={model} (base_model={base_model})")

    model_dir = ti.xcom_pull(task_ids="materialize_repo", key="model_dir")
    deploy_version = int(ti.xcom_pull(task_ids="materialize_repo", key="deploy_version"))
    run_id = ti.xcom_pull(task_ids="materialize_repo", key="run_id")
    alias = ti.xcom_pull(task_ids="materialize_repo", key="alias")

    payload = {
        "active_version": deploy_version,  # MLflow version
        "run_id": run_id,
        "alias": alias,
        "deploy_mode": deploy_mode,
        "updated_at_utc": utc_ts(),
    }

    path = os.path.join(model_dir, "current.json")
    tmp = path + ".tmp"
    with open(tmp, "w") as f:
        json.dump(payload, f, indent=2)
    os.replace(tmp, path)

    # (중복 허용) policy 재적용
    write_or_update_config_policy(model_dir, version=deploy_version)
    log.info("[commit] OK version=%s path=%s", deploy_version, path)


def rollback_minimal(ti, **_):
    """
    실패 시 최소 롤백:
    - promote: snapshot current.json 복구 + 실패 버전 move + triton reload
    - shadow:  실패 버전 move + triton reload (current.json 없음)
    """
    model = ti.xcom_pull(task_ids="materialize_repo", key="model")
    model_dir = ti.xcom_pull(task_ids="materialize_repo", key="model_dir")
    deploy_v = ti.xcom_pull(task_ids="materialize_repo", key="deploy_version")
    deploy_mode = ti.xcom_pull(task_ids="materialize_repo", key="deploy_mode")
    prev = ti.xcom_pull(task_ids="snapshot_current", key="prev_current")

    triton = build_triton_http_url()
    log.warning("[ROLLBACK] start model=%s deploy_version=%s mode=%s", model, deploy_v, deploy_mode)

    base_model = cfg("triton_model_name", required=True)

    if deploy_mode != "shadow" and model == base_model and prev is not None:
        path = os.path.join(model_dir, "current.json")
        tmp = path + ".tmp"
        with open(tmp, "w") as f:
            json.dump(prev, f, indent=2)
        os.replace(tmp, path)
        log.warning("[ROLLBACK] restored current.json (promote)")

    if deploy_v is not None:
        ver_dir = os.path.join(model_dir, str(deploy_v))
        if os.path.isdir(ver_dir):
            failed_dir = ver_dir + f".failed_{utc_ts()}"
            os.rename(ver_dir, failed_dir)
            log.warning("[ROLLBACK] moved failed dir: %s -> %s", ver_dir, failed_dir)

    # best-effort reload
    try:
        try:
            requests.post(f"{triton}/v2/repository/models/{model}/unload", timeout=10)
        except Exception:
            pass
        time.sleep(0.5)
        r = requests.post(f"{triton}/v2/repository/models/{model}/load", timeout=10)
        log.warning("[ROLLBACK] reload status=%s body=%s", r.status_code, (r.text or "")[:200])
    except Exception as e:
        log.warning("[ROLLBACK] reload failed: %s", e)


def rebuild_config_for_version(model: str, version: int) -> str:
    """
    (선택 유틸) 특정 MLflow version에 맞춰 config 재생성 + policy specific 포함
    """
    run_id = run_id_by_version(model, int(version))
    params = get_run_params(run_id)

    n_features = int(params.get("n_features", "0") or "0")
    n_classes = int(params.get("n_classes", "0") or "0")
    in_name = params.get("onnx_input_name", "input")

    outs = _parse_output_names(params.get("onnx_output_names", ""))
    out_label = "label"
    out_prob = "probabilities"
    for cand in outs:
        if "label" in cand.lower():
            out_label = cand
        if "prob" in cand.lower():
            out_prob = cand

    if n_features <= 0 or n_classes <= 0:
        raise RuntimeError(f"[config] invalid n_features/n_classes: {n_features}/{n_classes}")

    base = build_config_pbtxt(model, in_name, out_prob, out_label, n_features, n_classes)
    clean = _set_version_policy_specific(base, int(version))
    return clean


def rollback_manual(model: str | None = None, deploy_version: int | None = None):
    """
    ✅ 수동 롤백은 base_model(best_model) 전용으로 사용하는 걸 권장
    """
    model = model or cfg("triton_model_name", required=True)
    repo = cfg("triton_repo_base", "/models")
    model_dir = os.path.join(repo, model)
    os.makedirs(model_dir, exist_ok=True)

    path = os.path.join(model_dir, "current.json")
    cur = {}
    if os.path.exists(path):
        try:
            with open(path, "r") as f:
                cur = json.load(f) or {}
        except Exception as e:
            log.warning("[rollback_manual] read current.json failed: %s", e)

    triton = build_triton_http_url()

    if deploy_version is not None:
        dv = int(deploy_version)
        cur["active_version"] = dv
        cur["run_id"] = run_id_by_version(model, dv)
        cur["deploy_mode"] = "rollback_manual"
        cur["updated_at_utc"] = utc_ts()

        tmp = path + ".tmp"
        with open(tmp, "w") as f:
            json.dump(cur, f, indent=2)
        os.replace(tmp, path)

        cfg_text = rebuild_config_for_version(model, dv)
        _atomic_write(os.path.join(model_dir, "config.pbtxt"), cfg_text)

        log.warning("[ROLLBACK_MANUAL] forced dv=%s run_id=%s", dv, cur["run_id"])
    else:
        log.warning("[ROLLBACK_MANUAL] no deploy_version -> reload only")

    # unload -> load
    try:
        requests.post(f"{triton}/v2/repository/models/{model}/unload", timeout=10)
    except Exception:
        pass
    time.sleep(0.5)

    r = requests.post(f"{triton}/v2/repository/models/{model}/load", timeout=10)
    if r.status_code != 200:
        raise RuntimeError(f"[ROLLBACK_MANUAL] reload failed: {r.status_code} {r.text}")
    log.warning("[ROLLBACK_MANUAL] reload OK status=%s", r.status_code)

