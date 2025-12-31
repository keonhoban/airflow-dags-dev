# ml_code/train_model.py

import os
import tempfile
import mlflow
import mlflow.sklearn

from sklearn.datasets import load_iris
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

from mlflow.tracking import MlflowClient
from ml_code.config import get_tracking_uri, get_experiment_name
from airflow.utils.log.logging_mixin import LoggingMixin

logger = LoggingMixin().log


def _export_onnx_sklearn(clf, n_features: int, out_path: str):
    """
    sklearn LogisticRegression -> ONNX export
    """
    try:
        from skl2onnx import convert_sklearn
        from skl2onnx.common.data_types import FloatTensorType
    except Exception as e:
        raise RuntimeError(
            "ONNX export requires 'skl2onnx' (and 'onnx') installed in the worker image."
        ) from e

    initial_type = [("input", FloatTensorType([None, n_features]))]
    onnx_model = convert_sklearn(clf, initial_types=initial_type)

    with open(out_path, "wb") as f:
        f.write(onnx_model.SerializeToString())


def train_model(C, max_iter):
    tracking_uri = get_tracking_uri()
    logger.info(f"[DEBUG] ✅ tracking_uri: {tracking_uri}")

    mlflow.set_tracking_uri(tracking_uri)
    client = MlflowClient(tracking_uri=tracking_uri)

    experiment_name = get_experiment_name()
    experiment = client.get_experiment_by_name(experiment_name)

    if experiment:
        experiment_id = experiment.experiment_id
        logger.info(f"[DEBUG] ✅ 기존 실험 ID 사용: {experiment_id}")
    else:
        experiment_id = client.create_experiment(experiment_name)
        logger.info(f"[DEBUG] ✅ 새 실험 생성 ID: {experiment_id}")

    X, y = load_iris(return_X_y=True)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

    with mlflow.start_run(experiment_id=experiment_id) as run:
        run_id = run.info.run_id
        logger.info(f"[DEBUG] ✅ run_id: {run_id}")

        clf = LogisticRegression(C=C, max_iter=max_iter)
        clf.fit(X_train, y_train)

        y_pred = clf.predict(X_test)
        acc = accuracy_score(y_test, y_pred)

        mlflow.log_param("C", C)
        mlflow.log_param("max_iter", max_iter)
        mlflow.log_metric("accuracy", acc)

        # 1) 기존대로 MLflow model 패키지 저장 (model/)
        mlflow.sklearn.log_model(clf, "model")

        # 2) Triton용 ONNX 저장 (onnx/model.onnx)
        #    - 배포 DAG에서 triton_onnx_artifact_path="onnx/model.onnx"로 고정 가능
        try:
            with tempfile.TemporaryDirectory() as td:
                onnx_path = os.path.join(td, "model.onnx")
                _export_onnx_sklearn(clf, n_features=X_train.shape[1], out_path=onnx_path)
                mlflow.log_artifact(onnx_path, artifact_path="onnx")
                logger.info("[ONNX] ✅ exported & logged: onnx/model.onnx")
        except Exception as e:
            # 운영적으로는 여기서 fail 시키는 게 더 안전
            # (배포 파이프라인 계약이 'onnx/model.onnx 필수'라면 학습 성공해도 배포 불가)
            logger.error(f"[ONNX] ❌ export/log 실패: {e}")
            raise

        return acc, run_id
