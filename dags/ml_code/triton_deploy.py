# dags/ml_code/triton_deploy.py

import os
import json
import shutil
from datetime import datetime, timezone
from zoneinfo import ZoneInfo
from typing import Optional

import requests
import mlflow
from mlflow.tracking import MlflowClient

from airflow.sdk import Variable
from airflow.utils.log.logging_mixin import LoggingMixin

log = LoggingMixin().log


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
# Triton config.pbtxt generation
# -----------------------
def _to_int(v: Optional[str], default: int) -> int:
    try:
        if v is None:
            return default
        return int(str(v).strip())
    except Exception:
        return default


def _to_bool(v: Optional[str], default: bool) -> bool:
    if v is None:
        return default
    s = str(v).strip().lower()
    if s in ("1", "true", "yes", "y", "on"):
        return True
    if s in ("0", "false", "no", "n", "off"):
        return False
    return default


def generate_triton_config_pbtxt(
    *,
    model_name: str,
    input_name: str,
    input_dim: int,
    output_label_name: str,
    output_prob_name: str,
    num_classes: int,
    max_batch_size: int = 0,
    enable_dynamic_batching: bool = False,
    instance_count: int = 1,
) -> str:
    """
    가장 단순하지만 실무에서 잘 먹히는 ONNX 분류 모델 기본 템플릿
    - label: INT64 [1]
    - probabilities: FP32 [num_classes]
    """
    dyn = ""
    if enable_dynamic_batching and max_batch_size > 0:
        dyn = """
dynamic_batching {
  preferred_batch_size: [ 4, 8, 16 ]
  max_queue_delay_microseconds: 100
}
""".rstrip()

    # instance_group은 CPU-only 기준
    inst = f"""
instance_group [
  {{
    kind: KIND_CPU
    count: {instance_count}
  }}
]
""".rstrip()

    pbtxt = f"""
name: "{model_name}"
platform: "onnxruntime_onnx"
max_batch_size: {max_batch_size}

input [
  {{
    name: "{input_name}"
    data_type: TYPE_FP32
    dims: [ {input_dim} ]
  }}
]

output [
  {{
    name: "{output_label_name}"
    data_type: TYPE_INT64
    dims: [ 1 ]
  }},
  {{
    name: "{output_prob_name}"
    data_type: TYPE_FP32
    dims: [ {num_classes} ]
  }}
]

{dyn}

{inst}
""".strip()

    # 빈 줄 정리(가독성)
    lines = []
    prev_blank = False
    for line in pbtxt.splitlines():
        blank = (line.strip() == "")
        if blank and prev_blank:
            continue
        lines.append(line.rstrip())
        prev_blank = blank
    return "\n".join(lines) + "\n"


def ensure_config_pbtxt(
    *,
    model_dir: str,
    model_name: str,
    input_dim: int,
    num_classes: int,
):
    """
    {repo}/{model}/config.pbtxt 를 항상 보장
    - 기본 overwrite: true (실무에서는 '소스 오브 트루스는 DAG 코드'가 안전)
    - 필요하면 ENV/Variable로 제어
    """
    overwrite = _to_bool(cfg("TRITON_CONFIG_OVERWRITE", "true"), True)

    input_name = cfg("TRITON_INPUT_NAME", "input")
    output_label_name = cfg("TRITON_OUTPUT_LABEL_NAME", "label")
    output_prob_name = cfg("TRITON_OUTPUT_PROB_NAME", "probabilities")

    max_batch_size = _to_int(cfg("TRITON_MAX_BATCH_SIZE", "0"), 0)
    enable_dynamic_batching = _to_bool(cfg("TRITON_ENABLE_DYNAMIC_BATCHING", "false"), False)
    instance_count = _to_int(cfg("TRITON_INSTANCE_COUNT", "1"), 1)

    config_path = os.path.join(model_dir, "config.pbtxt")
    if os.path.exists(config_path) and not overwrite:
        log.info("[W6] config.pbtxt exists and overwrite disabled: %s", config_path)
        return

    config_text = generate_triton_config_pbtxt(
        model_name=model_name,
        input_name=input_name,
        input_dim=input_dim,
        output_label_name=output_label_name,
        output_prob_name=output_prob_name,
        num_classes=num_classes,
        max_batch_size=max_batch_size,
        enable_dynamic_batching=enable_dynamic_batching,
        instance_count=instance_count,
    )

    tmp = config_path + ".tmp"
    with open(tmp, "w") as f:
        f.write(config_text)
    os.replace(tmp, config_path)

    log.info("[W6] config.pbtxt written: %s (overwrite=%s)", config_path, overwrite)


# -----------------------
# Tasks
# -----------------------
def materialize(ti, alias: str = "A", **_):
    """
    - UI params.alias 로 alias 주입 받음
    - Triton repository 규칙:
      {repo}/{model}/{version}/model.onnx
      {repo}/{model}/config.pbtxt   (모델 공통 설정 파일)
    """
    model = cfg("triton_model_name", required=True)  # 예: best_model
    repo = cfg("triton_repo_base", "/models")        # Triton이 마운트한 model repo
    onnx_rel = cfg("triton_onnx_artifact_path", "onnx/model.onnx")  # runs:/.../onnx/model.onnx

    # config 생성에 필요한 "계약" 값들 (학습/ONNX export 쪽과 맞춰야 함)
    # iris 기준: input_dim=4, num_classes=3
    input_dim = _to_int(cfg("TRITON_INPUT_DIM", "4"), 4)
    num_classes = _to_int(cfg("TRITON_NUM_CLASSES", "3"), 3)

    # alias는 DAG params가 1순위. (빈 문자열 방어)
    alias = (alias or "").strip() or cfg("mlflow_alias", "A")

    v, run_id = select_by_alias(model, alias)

    model_dir = os.path.join(repo, model)
    ver_dir = os.path.join(model_dir, str(v))  # ✅ Triton은 정수 버전 폴더 권장
    os.makedirs(ver_dir, exist_ok=True)

    # (1) ONNX 다운로드 -> 버전 폴더에 복사
    local = mlflow.artifacts.download_artifacts(artifact_uri=f"runs:/{run_id}/{onnx_rel}")
    dst = os.path.join(ver_dir, "model.onnx")
    shutil.copyfile(local, dst)
    log.info("[W6] model.onnx copied: %s", dst)

    # (2) config.pbtxt 보장 (모델 디렉터리 바로 아래)
    ensure_config_pbtxt(
        model_dir=model_dir,
        model_name=model,
        input_dim=input_dim,
        num_classes=num_classes,
    )

    # xcom
    ti.xcom_push(key="model", value=model)
    ti.xcom_push(key="model_dir", value=model_dir)
    ti.xcom_push(key="deploy_version", value=v)
    ti.xcom_push(key="run_id", value=run_id)
    ti.xcom_push(key="alias", value=alias)

    log.info("[W6] materialize OK model=%s alias=@%s version=%s", model, alias, v)


def triton_load(ti, **_):
    model = ti.xcom_pull(task_ids="materialize_repo", key="model")
    alias = ti.xcom_pull(task_ids="materialize_repo", key="alias")
    triton = build_triton_http_url()

    log.info("[W6] triton_load model=%s alias=@%s triton=%s", model, alias, triton)

    # 주의: MODE_EXPLICIT이면 반드시 load 호출이 필요
    r = requests.post(f"{triton}/v2/repository/models/{model}/load", timeout=10)
    if r.status_code != 200:
        raise RuntimeError(f"load failed: {r.status_code} {r.text}")


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
