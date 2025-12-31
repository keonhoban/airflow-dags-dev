# ml_code/train_model.py 
import os
import mlflow
import mlflow.sklearn
from sklearn.datasets import load_iris
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from mlflow.tracking import MlflowClient
from ml_code.config import get_tracking_uri, get_experiment_name
from airflow.utils.log.logging_mixin import LoggingMixin

# ONNX 추가
import numpy as np
from skl2onnx import convert_sklearn
from skl2onnx.common.data_types import FloatTensorType

logger = LoggingMixin().log

def export_onnx_and_log_artifact(clf, X_train):
    """
    runs:/<run_id>/onnx/model.onnx 로 항상 저장되도록 보장
    Triton 호환을 위해 확률 출력이 SEQUENCE가 아니라 TENSOR가 되도록 zipmap 비활성화
    """
    initial_type = [("input", FloatTensorType([None, X_train.shape[1]]))]

    onnx_model = convert_sklearn(
        clf,
        initial_types=initial_type,
        options={id(clf): {"zipmap": False}}  # ⭐ 핵심
    )

    onnx_path = "/tmp/model.onnx"
    with open(onnx_path, "wb") as f:
        f.write(onnx_model.SerializeToString())

    mlflow.log_artifact(onnx_path, artifact_path="onnx")
    logger.info("[ONNX] logged: onnx/model.onnx")


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

    try:
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

            # 기존 MLflow Model 패키지
            mlflow.sklearn.log_model(clf, "model")

            # ONNX도 같이 저장
            export_onnx_and_log_artifact(clf, X_train)

            return acc, run_id

    except Exception as e:
        logger.error(f"[ERROR] ❌ 학습 또는 로깅 중 오류: {e}")
        raise
