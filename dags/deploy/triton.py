from __future__ import annotations
import os, shutil, json, time
import requests
from pathlib import Path
from airflow.models.taskinstance import TaskInstance

from utils.config import triton_http_url, triton_model_repo, model_name
from utils.slack import info

# (중요) wrapper 이름 충돌 방지: triton_snapshot_current 로 외부에 노출
def snapshot_current(ti: TaskInstance):
    repo = Path(triton_model_repo()) / model_name()
    snap_dir = repo / "_snapshot"
    snap_dir.mkdir(parents=True, exist_ok=True)

    current = repo / "current"
    if current.exists() and current.is_dir():
        # current 내용을 snapshot으로 복사
        dst = snap_dir / f"current_{int(time.time())}"
        shutil.copytree(current, dst, dirs_exist_ok=True)
        ti.xcom_push(key="snapshot_path", value=str(dst))
    else:
        ti.xcom_push(key="snapshot_path", value=None)

def materialize_from_shared_repo(ti: TaskInstance, target_version: str):
    """
    여기서는 "이미 /models/best_model/<version>/model.onnx 가 존재한다"는 전제.
    (건호님 환경에서 find로 확인됨)
    current -> target_version symlink(or copy) 정책만 정하면 됩니다.
    """
    repo = Path(triton_model_repo()) / model_name()
    ver_dir = repo / str(target_version)
    if not ver_dir.exists():
        raise FileNotFoundError(f"triton model version dir not found: {ver_dir}")

    current = repo / "current"
    if current.exists():
        if current.is_symlink() or current.is_file():
            current.unlink()
        else:
            shutil.rmtree(current)

    # 심볼릭 링크가 가장 깔끔 (권한/FS에 따라 copy로 바꿔도 됨)
    current.symlink_to(ver_dir, target_is_directory=True)
    ti.xcom_push(key="deployed_version", value=str(target_version))

def triton_load_ready_smoke(target_model: str):
    base = triton_http_url()
    # explicit mode 기준: load 호출 필요
    r = requests.post(f"{base}/v2/repository/models/{target_model}/load", timeout=10)
    r.raise_for_status()

    r = requests.get(f"{base}/v2/models/{target_model}/ready", timeout=10)
    r.raise_for_status()

    # smoke infer는 건호님이 이미 성공한 payload 방식 그대로(입력 피처 수는 모델에 맞게)
    # 여기서는 예시로 3개
    payload = {
        "inputs": [{
            "name": "input",
            "shape": [1, 3],
            "datatype": "FP32",
            "data": [[0.0, 0.0, 0.0]],
        }]
    }
    r = requests.post(f"{base}/v2/models/{target_model}/infer", json=payload, timeout=10)
    r.raise_for_status()

def commit_current(ti: TaskInstance):
    # current 버전을 “커밋 마커”로 남기고 싶으면 여기에 기록
    v = ti.xcom_pull(key="deployed_version", task_ids="triton_materialize_and_smoke")
    info("Triton commit", deployed_version=v)

def rollback_minimal(ti: TaskInstance):
    snap = ti.xcom_pull(key="snapshot_path", task_ids="triton_snapshot_current")
    if not snap:
        return
    repo = Path(triton_model_repo()) / model_name()
    current = repo / "current"
    if current.exists():
        if current.is_symlink() or current.is_file():
            current.unlink()
        else:
            shutil.rmtree(current)
    # snapshot의 "폴더"를 current로 복구(copy)
    shutil.copytree(snap, current, dirs_exist_ok=True)

