# ml_code/triton_deploy.py

import os
import json
import shutil
import requests
from datetime import datetime, timezone
from zoneinfo import ZoneInfo

import mlflow
from mlflow.tracking import MlflowClient

from airflow.sdk import Variable
from airflow.utils.log.logging_mixin import LoggingMixin

log = LoggingMixin().log


# =========================
# Config loader (ENV > Variable > default)
# =========================
def cfg(key: str, default=None, *, required: bool = False):
    v = os.getenv(key)
    if v and str(v).strip():
        return v

    try:
        if default is None:
            v = Variable.get(key)
        else:
            v = Variable.get(key, default_var=str(default))
        if v and str(v).strip():
            return v
    except Exception:
        pass

    if required:
        raise RuntimeError(f"[Config] missing required key: {key}")
    return default


# =========================
# Time helpers
# =========================
def utc_ts():
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S")


def kst_ts():
    return datetime.now(ZoneInfo("Asia/Seoul")).strftime("%Y%m%dT%H%M%S")


# =========================
# MLflow: alias → (version, run_id)
# =========================
def select_by_alias(model_name: str, alias: str):
    tracking_uri = cfg("MLFLOW_TRACKING_URI", required=True)
    mlflow.set_tracking_uri(tracking_uri)

    client = MlflowClient(tracking_uri=tracking_uri)
    mv = client.get_model_version_by_alias(model_name, alias)

    return int(mv.version), mv.run_id


# =========================
# Task 1: materialize model repo
# =========================
def materialize(ti, **_):
    """
    MLflow alias 기준으로 ONNX artifact를 가져와
    Triton model-repo에 정수 버전 디렉터리로 배치
    """
    model = cfg("triton_model_name", required=True)          # ex) best_model
    alias = cfg("mlflow_alias", "A")                          # ex) A
    repo = cfg("triton_repo_base", "/models")                 # Triton PV mount
    onnx_rel = cfg("triton_onnx_artifact_path", "onnx/model.onnx")

    version, run_id = select_by_alias(model, alias)

    # Triton 규칙: 버전 디렉터리는 반드시 정수
    model_dir = os.path.join(repo, model)
    version_dir = os.path.join(model_dir, str(version))
    os.makedirs(version_dir, exist_ok=True)

    # MLflow → local → NFS
    local_path = mlflow.artifacts.download_artifacts(
        artifact_uri=f"runs:/{run_id}/{onnx_rel}"
    )
    dst = os.path.join(version_dir, "model.onnx")
    shutil.copyfile(local_path, dst)

    # XCom
    ti.xcom_push(key="model", value=model)
    ti.xcom_push(key="version", value=version)
    ti.xcom_push(key="run_id", value=run_id)
    ti.xcom_push(key="model_dir", value=model_dir)

    log.info(
        "[Triton][materialize] model=%s alias=@%s version=%s dst=%s",
        model, alias, version, dst
    )


# =========================
# Task 2: Triton load
# =========================
def triton_load(ti, **_):
    model = ti.xcom_pull(task_ids="materialize_repo", key="model")
    triton_url = cfg("triton_http_url", required=True)  # 내부 Service DNS

    url = f"{triton_url}/v2/repository/models/{model}/load"
    r = requests.post(url, timeout=10)

    if r.status_code != 200:
        raise RuntimeError(
            f"[Triton][load] failed: {r.status_code} {r.text}"
        )

    log.info("[Triton][load] OK model=%s", model)


# =========================
# Task 3: commit current.json (운영 메타)
# =========================
def commit_current(ti, **_):
    model_dir = ti.xcom_pull(task_ids="materialize_repo", key="model_dir")
    version = ti.xcom_pull(task_ids="materialize_repo", key="version")
    run_id = ti.xcom_pull(task_ids="materialize_repo", key="run_id")

    payload = {
        "active_version": version,
        "run_id": run_id,
        "updated_at_utc": utc_ts(),
    }

    path = os.path.join(model_dir, "current.json")
    with open(path + ".tmp", "w") as f:
        json.dump(payload, f, indent=2)

    os.replace(path + ".tmp", path)

    log.info("[Triton][commit] current.json updated → v%s", version)
