# dags/ml_code/triton_actions.py
from __future__ import annotations

import os
import shutil
from datetime import datetime, timezone
from functools import lru_cache
from typing import Any, Dict, Optional, Tuple

import mlflow
from airflow.utils.log.logging_mixin import LoggingMixin

from ml_code.config import cfg, get_mlflow_client, get_tracking_uri
from mlops_lib.core.triton_config import build_config_pbtxt, write_config_atomic
from mlops_lib.infra.http import request_ok, request_json
from mlops_lib.core.policy import (
    T_TRITON_UNLOAD,
    T_TRITON_LOAD,
    T_TRITON_READY,
    T_TRITON_INFER,
    triton_opt_config,
)

log = LoggingMixin().log


def utc_ts() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S")


def now_ts() -> str:
    return datetime.now().strftime("%Y%m%d%H%M%S")


def _ensure_tracking_uri() -> None:
    uri = get_tracking_uri()
    try:
        mlflow.set_tracking_uri(uri)
    except Exception as e:
        raise RuntimeError(f"[MLflow] set_tracking_uri failed uri={uri} err={e}") from e


@lru_cache(maxsize=1)
def build_triton_http_url() -> str:
    full = cfg("TRITON_HTTP_URL", None)
    if full:
        return str(full)

    svc = cfg("TRITON_SERVICE", "triton")
    ns = cfg("TRITON_NAMESPACE", "triton-dev")
    port = cfg("TRITON_HTTP_PORT", "8000")
    return f"http://{svc}.{ns}.svc.cluster.local:{port}"


# ---------- MLflow helpers ----------
def get_run_params(run_id: str) -> Dict[str, str]:
    _ensure_tracking_uri()
    c = get_mlflow_client()
    run = c.get_run(run_id)
    return dict(run.data.params or {})


def select_by_alias(model_name: str, alias: str) -> Tuple[int, str]:
    _ensure_tracking_uri()
    c = get_mlflow_client()
    mv = c.get_model_version_by_alias(model_name, alias)
    return int(mv.version), str(mv.run_id)


def run_id_by_version(model_name: str, version: int) -> str:
    _ensure_tracking_uri()
    c = get_mlflow_client()
    mv = c.get_model_version(model_name, str(int(version)))
    return str(mv.run_id)


# ---------- Triton HTTP actions ----------
def triton_unload(model: str) -> None:
    triton = build_triton_http_url()
    try:
        request_ok("POST", f"{triton}/v2/repository/models/{model}/unload", timeout=T_TRITON_UNLOAD)
    except Exception as e:
        log.warning("[unload] ignore model=%s err=%s", model, e)


def triton_load(model: str) -> None:
    triton = build_triton_http_url()
    request_ok("POST", f"{triton}/v2/repository/models/{model}/load", timeout=T_TRITON_LOAD)
    log.info("[load] OK model=%s", model)


def triton_ready(model: str) -> None:
    triton = build_triton_http_url()
    request_ok("GET", f"{triton}/v2/models/{model}/ready", timeout=T_TRITON_READY)
    log.info("[ready] OK model=%s", model)


def triton_infer_smoke(model: str, *, in_name: str, n_features: int) -> Dict[str, Any]:
    triton = build_triton_http_url()
    if n_features <= 0:
        raise RuntimeError(f"[smoke] invalid n_features={n_features} model={model}")

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
        timeout=T_TRITON_INFER,
    )
    log.info("[smoke] OK model=%s resp=%s", model, str(res)[:300])
    return res


# ---------- Materialize helpers ----------
def _parse_int(params: Dict[str, str], key: str, default: int = 0) -> int:
    try:
        return int(params.get(key, str(default)) or str(default))
    except Exception:
        return default


def _pick_outputs(outs_raw: str) -> Tuple[str, str]:
    outs = [x.strip() for x in (outs_raw or "").split(",") if x.strip()]

    out_label = "label"
    out_prob = "probabilities"

    if not outs:
        return out_prob, out_label

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

    return out_prob, out_label


def _download_onnx_artifact(run_id: str, onnx_rel: str) -> str:
    _ensure_tracking_uri()
    return str(mlflow.artifacts.download_artifacts(artifact_uri=f"runs:/{run_id}/{onnx_rel}"))


def _write_model_onnx(dst: str, local_path: str, *, model: str, deploy_version: int, run_id: str) -> None:
    try:
        tmp = dst + ".tmp"
        shutil.copyfile(str(local_path), tmp)
        os.replace(tmp, dst)
    except Exception as e:
        raise RuntimeError(
            f"[materialize] write model.onnx failed model={model} version={deploy_version} run_id={run_id} dst={dst} err={e}"
        ) from e


def decide_deploy_target(
    *,
    base_model: str,
    alias: str,
    run_id: Optional[str],
    shadow: bool,
) -> Tuple[str, int, str, str, str]:
    shadow_model = cfg("triton_model_name_shadow", f"{base_model}_shadow")

    if shadow:
        if not run_id:
            raise ValueError("[materialize] shadow=True requires run_id")
        used_alias = (alias or "").strip() or "A"
        return str(shadow_model), int(now_ts()), str(run_id), used_alias, "shadow"

    used_alias = (alias or "").strip() or cfg("mlflow_alias", "A")
    deploy_version, chosen_run_id = select_by_alias(str(base_model), str(used_alias))
    return str(base_model), int(deploy_version), str(chosen_run_id), str(used_alias), "promote"


def materialize_repo(*, model: str, deploy_version: int, run_id: str) -> Dict[str, Any]:
    repo = cfg("triton_repo_base", "/models")
    onnx_rel = cfg("triton_onnx_artifact_path", "onnx/model.onnx")

    local = _download_onnx_artifact(run_id, onnx_rel)

    params = get_run_params(run_id)
    n_features = _parse_int(params, "n_features", 0)
    n_classes = _parse_int(params, "n_classes", 0)
    in_name = params.get("onnx_input_name", "input") or "input"

    out_prob, out_label = _pick_outputs(params.get("onnx_output_names", "") or "")

    if n_features <= 0 or n_classes <= 0:
        raise RuntimeError(
            f"[Triton] invalid n_features/n_classes model={model} version={deploy_version} run_id={run_id} "
            f"n_features={n_features} n_classes={n_classes}"
        )

    model_dir = os.path.join(str(repo), str(model))
    ver_dir = os.path.join(model_dir, str(int(deploy_version)))
    os.makedirs(ver_dir, exist_ok=True)

    dst = os.path.join(ver_dir, "model.onnx")
    _write_model_onnx(dst, local, model=str(model), deploy_version=int(deploy_version), run_id=str(run_id))

    cfg_text = build_config_pbtxt(str(model), str(in_name), str(out_prob), str(out_label), n_features, n_classes, opt=triton_opt_config())
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
    run_id = run_id_by_version(model, int(version))
    params = get_run_params(run_id)

    n_features = _parse_int(params, "n_features", 0)
    n_classes = _parse_int(params, "n_classes", 0)
    in_name = params.get("onnx_input_name", "input") or "input"

    out_prob, out_label = _pick_outputs(params.get("onnx_output_names", "") or "")

    if n_features <= 0 or n_classes <= 0:
        raise RuntimeError(
            f"[config] invalid n_features/n_classes model={model} version={version} run_id={run_id} "
            f"n_features={n_features} n_classes={n_classes}"
        )

    return build_config_pbtxt(model, str(in_name), str(out_prob), str(out_label), n_features, n_classes, opt=triton_opt_config())
