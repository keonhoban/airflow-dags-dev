# dags/ml_code/triton_actions.py
from __future__ import annotations

import os
import shutil
from datetime import datetime, timezone
from typing import Any, Dict, Optional, Tuple

import mlflow
from airflow.utils.log.logging_mixin import LoggingMixin

from ml_code.config import cfg, get_mlflow_client, get_tracking_uri
from mlops_lib.core.triton_config import (
    build_config_pbtxt,
    write_config_atomic,
    atomic_write,
)
from mlops_lib.infra.http import request_ok, request_json

log = LoggingMixin().log


def utc_ts() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S")


def now_ts() -> str:
    return datetime.now().strftime("%Y%m%d%H%M%S")


def build_triton_http_url() -> str:
    full = cfg("TRITON_HTTP_URL", None)
    if full:
        return str(full)
    svc = cfg("TRITON_SERVICE", "triton")
    ns = cfg("TRITON_NAMESPACE", "triton-dev")
    port = cfg("TRITON_HTTP_PORT", "8000")
    return f"http://{svc}.{ns}.svc.cluster.local:{port}"


# ---------- MLflow ----------
def get_run_params(run_id: str) -> Dict[str, str]:
    c = get_mlflow_client()
    run = c.get_run(run_id)
    return dict(run.data.params or {})


def select_by_alias(model_name: str, alias: str) -> Tuple[int, str]:
    c = get_mlflow_client()
    mv = c.get_model_version_by_alias(model_name, alias)
    return int(mv.version), str(mv.run_id)


def run_id_by_version(model_name: str, version: int) -> str:
    c = get_mlflow_client()
    mv = c.get_model_version(model_name, str(int(version)))
    return str(mv.run_id)


# ---------- Triton HTTP actions ----------
def triton_unload(model: str) -> None:
    triton = build_triton_http_url()
    try:
        request_ok("POST", f"{triton}/v2/repository/models/{model}/unload", timeout=10)
    except Exception as e:
        log.warning("[unload] ignore err=%s", e)


def triton_load(model: str) -> None:
    triton = build_triton_http_url()
    request_ok("POST", f"{triton}/v2/repository/models/{model}/load", timeout=10)
    log.info("[load] OK model=%s", model)


def triton_ready(model: str) -> None:
    triton = build_triton_http_url()
    request_ok("GET", f"{triton}/v2/models/{model}/ready", timeout=5)
    log.info("[ready] OK model=%s", model)


def triton_infer_smoke(model: str, *, in_name: str, n_features: int) -> Dict[str, Any]:
    triton = build_triton_http_url()
    if n_features <= 0:
        raise RuntimeError("[smoke] invalid n_features")

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
    res = request_json(
        "POST",
        f"{triton}/v2/models/{model}/infer",
        headers={"Content-Type": "application/json"},
        json_body=payload,
        timeout=10,
    )
    log.info("[smoke] OK model=%s resp=%s", model, str(res)[:300])
    return res


# ---------- Materialize ----------
def decide_deploy_target(
    *,
    base_model: str,
    alias: str,
    run_id: Optional[str],
    shadow: bool,
) -> Tuple[str, int, str, str, str]:
    """
    Returns: (model, deploy_version, chosen_run_id, used_alias, mode)
    """
    shadow_model = cfg("triton_model_name_shadow", f"{base_model}_shadow")

    if shadow:
        if not run_id:
            raise ValueError("[materialize] shadow=True requires run_id")
        return str(shadow_model), int(now_ts()), str(run_id), (alias or "").strip() or "A", "shadow"

    used_alias = (alias or "").strip() or cfg("mlflow_alias", "A")
    deploy_version, chosen_run_id = select_by_alias(str(base_model), str(used_alias))
    return str(base_model), int(deploy_version), str(chosen_run_id), str(used_alias), "promote"


def materialize_repo(
    *,
    model: str,
    deploy_version: int,
    run_id: str,
) -> Dict[str, Any]:
    """
    - 모델 artifact 다운로드 → /models/<model>/<deploy_version>/model.onnx
    - ✅ config.pbtxt는 version_policy 없이 '고정 형태'로 생성/유지
    """
    repo = cfg("triton_repo_base", "/models")
    onnx_rel = cfg("triton_onnx_artifact_path", "onnx/model.onnx")

    mlflow.set_tracking_uri(get_tracking_uri())
    local = mlflow.artifacts.download_artifacts(artifact_uri=f"runs:/{run_id}/{onnx_rel}")

    params = get_run_params(run_id)
    n_features = int(params.get("n_features", "0") or "0")
    n_classes = int(params.get("n_classes", "0") or "0")
    in_name = params.get("onnx_input_name", "input")

    # outputs
    outs_raw = params.get("onnx_output_names", "")
    outs = [x.strip() for x in outs_raw.split(",") if x.strip()]

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
        if len(outs) >= 2 and (out_label not in outs or out_prob not in outs):
            out_label, out_prob = outs[0], outs[1]

    if n_features <= 0 or n_classes <= 0:
        raise RuntimeError(f"[Triton] invalid n_features/n_classes: {n_features}/{n_classes}")

    model_dir = os.path.join(str(repo), str(model))
    ver_dir = os.path.join(model_dir, str(int(deploy_version)))
    os.makedirs(ver_dir, exist_ok=True)

    dst = os.path.join(ver_dir, "model.onnx")
    tmp = dst + ".tmp"
    shutil.copyfile(str(local), tmp)
    os.replace(tmp, dst)

    # ✅ config.pbtxt는 버전 강제 없이 생성/유지
    cfg_text = build_config_pbtxt(str(model), str(in_name), str(out_prob), str(out_label), n_features, n_classes)
    os.makedirs(model_dir, exist_ok=True)
    write_config_atomic(model_dir, cfg_text=cfg_text)

    return {
        "model": str(model),
        "model_dir": model_dir,
        "deploy_version": int(deploy_version),
        "run_id": str(run_id),
        "n_features": int(n_features),
        "n_classes": int(n_classes),
        "onnx_input_name": str(in_name),
    }


def rebuild_config_for_version(model: str, version: int) -> str:
    # ✅ version_policy를 쓰지 않으므로 "버전별 재생성"이 아니라 "고정 config 재생성"만 유지
    run_id = run_id_by_version(model, int(version))
    params = get_run_params(run_id)

    n_features = int(params.get("n_features", "0") or "0")
    n_classes = int(params.get("n_classes", "0") or "0")
    in_name = params.get("onnx_input_name", "input")

    outs_raw = params.get("onnx_output_names", "")
    outs = [x.strip() for x in outs_raw.split(",") if x.strip()]

    out_label = "label"
    out_prob = "probabilities"
    for cand in outs:
        if "label" in cand.lower():
            out_label = cand
        if "prob" in cand.lower():
            out_prob = cand

    if n_features <= 0 or n_classes <= 0:
        raise RuntimeError(f"[config] invalid n_features/n_classes: {n_features}/{n_classes}")

    return build_config_pbtxt(model, str(in_name), str(out_prob), str(out_label), n_features, n_classes)
