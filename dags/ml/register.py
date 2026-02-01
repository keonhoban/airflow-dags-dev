from __future__ import annotations
import os
import mlflow
from mlflow.tracking import MlflowClient

def register_and_set_alias(run_id: str, model_name: str, alias: str) -> int:
    mlflow.set_tracking_uri(os.environ["MLFLOW_TRACKING_URI"])
    c = MlflowClient()

    mv = c.create_model_version(
        name=model_name,
        source=f"runs:/{run_id}/model",
        run_id=run_id,
    )
    version = int(mv.version)

    # alias -> version
    c.set_registered_model_alias(name=model_name, alias=alias, version=str(version))
    return version

