import os, json, shutil
from datetime import datetime, timezone
import requests
import mlflow
from mlflow.tracking import MlflowClient
from airflow.sdk import Variable
from airflow.utils.log.logging_mixin import LoggingMixin

log = LoggingMixin().log

def kst_ts():
    return datetime.now(ZoneInfo("Asia/Seoul")).strftime("%Y%m%dT%H%M%S")

def select_prod(model_name: str, tag_key="stage", tag_value="Production"):
    uri = Variable.get("MLFLOW_TRACKING_URI")
    mlflow.set_tracking_uri(uri)
    c = MlflowClient(tracking_uri=uri)

    prod = []
    for mv in c.search_model_versions(f"name='{model_name}'"):
        if (mv.tags or {}).get(tag_key) == tag_value:
            prod.append(mv)

    if not prod:
        raise RuntimeError(f"no prod version: {model_name} tags.{tag_key}={tag_value}")

    prod.sort(key=lambda x: int(x.version), reverse=True)
    mv = prod[0]
    return mv.version, mv.run_id

def materialize(ti, **_):
    model = Variable.get("triton_model_name")              # ex) simple_model
    repo  = Variable.get("triton_repo_base", "/models")    # /models
    onnx_rel = Variable.get("triton_onnx_artifact_path", "onnx/model.onnx")  # 고정

    v, run_id = select_prod(model)

    deploy = f"v{v}_{kst_ts()}"
    model_dir = os.path.join(repo, model)
    ver_dir = os.path.join(model_dir, deploy)
    os.makedirs(ver_dir, exist_ok=True)

    # MLflow artifact -> 로컬 다운로드 -> NFS로 복사 (최종 파일명 model.onnx로 표준화)
    local = mlflow.artifacts.download_artifacts(artifact_uri=f"runs:/{run_id}/{onnx_rel}")
    dst = os.path.join(ver_dir, "model.onnx")
    shutil.copyfile(local, dst)

    ti.xcom_push(key="model", value=model)
    ti.xcom_push(key="model_dir", value=model_dir)
    ti.xcom_push(key="deploy", value=deploy)
    ti.xcom_push(key="mlflow_version", value=int(v))
    ti.xcom_push(key="run_id", value=run_id)

    log.info("[W6] materialize OK model=%s deploy=%s dst=%s", model, deploy, dst)

def triton_load(ti, **_):
    model = ti.xcom_pull(task_ids="materialize_repo", key="model")
    triton = Variable.get("triton_http_url")  # ex) http://triton.triton-dev...:8000

    r = requests.post(f"{triton}/v2/repository/models/{model}/load", timeout=5)
    if r.status_code != 200:
        raise RuntimeError(f"load failed: {r.status_code} {r.text}")

def commit_current(ti, **_):
    model_dir = ti.xcom_pull(task_ids="materialize_repo", key="model_dir")
    payload = {
        "active_version": ti.xcom_pull(task_ids="materialize_repo", key="deploy"),
        "mlflow_version": ti.xcom_pull(task_ids="materialize_repo", key="mlflow_version"),
        "run_id": ti.xcom_pull(task_ids="materialize_repo", key="run_id"),
        "updated_at_utc": utc_ts(),
    }
    path = os.path.join(model_dir, "current.json")
    with open(path + ".tmp", "w") as f:
        json.dump(payload, f, indent=2)
    os.replace(path + ".tmp", path)

