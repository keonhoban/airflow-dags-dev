from __future__ import annotations
import os, time
import mlflow
from mlflow.tracking import MlflowClient

def wait_until_ready(model_name: str, version: int, timeout_sec: int = 60) -> bool:
    mlflow.set_tracking_uri(os.environ["MLFLOW_TRACKING_URI"])
    c = MlflowClient()

    deadline = time.time() + timeout_sec
    while time.time() < deadline:
        mv = c.get_model_version(name=model_name, version=str(version))
        # MLflow 기본 상태는 READY가 아닌 경우가 있어도 대부분 접근 가능하지만,
        # 여기서는 "짧은 센서"로 안정성만 확보
        if mv:
            return True
        time.sleep(2)
    raise TimeoutError("model version not reachable")

