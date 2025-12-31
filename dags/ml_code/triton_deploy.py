# dags/ml_code/triton_deploy.py

import os, json, shutil
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
# NOTE:
# - 현재 best_model ONNX의 입출력을 /v2/models/best_model 결과에 맞춰 고정
# - 운영에서는 이 템플릿을 "모델별/버전별"로 분리하거나, ONNX에서 shape/name을 파싱해 생성하는 방식으로 확장 가능
CONFIG_TEMPLATE = """\
name: "{model}"
platform: "onnxruntime_onnx"

# iris logistic regression은 실시간 단건 추론이 목적이므로 batch 비활성(0)
max_batch_size: 0

input [
  {{
    name: "input"
    data_type: TYPE_FP32
    dims: [ 4 ]
  }}
]

output [
  {{
    name: "probabilities"
    data_type: TYPE_FP32
    dims: [ 3 ]
  }},
  {{
    name: "label"
    data_type: TYPE_INT64
    dims: [ 1 ]
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
    """
    클러스터 내부 통신용 Triton URL 생성
    - 기본: http://triton.triton-dev.svc.cluster.local:8000
    - 필요한 경우 ENV/Variable로 override 가능
    """
    svc = cfg("TRITON_SERVICE", "triton")
    ns = cfg("TRITON_NAMESPACE", "triton-dev")
    port = cfg("TRITON_HTTP_PORT", "8000")

    # 명시적으로 전체 URL을 주고 싶으면 이것만 세팅해도 됨(선택)
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
def materialize(ti, alias: str = "A", **_):
    """
    - UI params.alias 로 alias 주입 받음
    - Triton repository 규칙: {repo}/{model}/{version}/model.onnx
    - 실무형:
      - model root에 config.pbtxt 생성 (표준 구조)
      - version dir에 model.onnx 배치
    """
    model = cfg("triton_model_name", required=True)  # 예: best_model
    repo = cfg("triton_repo_base", "/models")        # Triton이 마운트한 model repo
    onnx_rel = cfg("triton_onnx_artifact_path", "onnx/model.onnx")  # runs:/.../onnx/model.onnx

    # alias는 DAG params가 1순위. (빈 문자열 방어)
    alias = (alias or "").strip() or cfg("mlflow_alias", "A")

    v, run_id = select_by_alias(model, alias)

    model_dir = os.path.join(repo, model)
    ver_dir = os.path.join(model_dir, str(v))  # ✅ Triton은 정수 버전 폴더 권장
    os.makedirs(ver_dir, exist_ok=True)

    # 1) MLflow artifact -> 로컬 다운로드 -> NFS 복사
    local = mlflow.artifacts.download_artifacts(artifact_uri=f"runs:/{run_id}/{onnx_rel}")
    dst = os.path.join(ver_dir, "model.onnx")
    shutil.copyfile(local, dst)

    # 2) config.pbtxt 생성 (✅ 표준: model root)
    os.makedirs(model_dir, exist_ok=True)
    config_path = os.path.join(model_dir, "config.pbtxt")
    with open(config_path, "w") as f:
        f.write(CONFIG_TEMPLATE.format(model=model))

    # XCom
    ti.xcom_push(key="model", value=model)
    ti.xcom_push(key="model_dir", value=model_dir)
    ti.xcom_push(key="deploy_version", value=v)
    ti.xcom_push(key="run_id", value=run_id)
    ti.xcom_push(key="alias", value=alias)

    log.info("[W6] materialize OK model=%s alias=@%s version=%s dst=%s", model, alias, v, dst)
    log.info("[W6] config.pbtxt created path=%s", config_path)


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

    # iris 샘플 1건 (input dims=4)
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

    resp = r.json()
    log.info("[W6] infer smoke OK model=%s resp=%s", model, resp)


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

    log.info("[W6] commit_current OK path=%s", path)
