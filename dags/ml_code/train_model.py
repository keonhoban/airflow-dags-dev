# ml_code/train_model.py

import mlflow
import mlflow.sklearn
from sklearn.datasets import load_iris
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from mlflow.tracking import MlflowClient
from ml_code.config import get_tracking_uri
from ml_code.config import get_experiment_name

def train_model(C, max_iter):
    # ✅ tracking_uri 한 번만 로드
    tracking_uri = get_tracking_uri()
    print(f"[DEBUG] ✅ tracking_uri: {tracking_uri}")
    mlflow.set_tracking_uri(tracking_uri)
    client = MlflowClient(tracking_uri=tracking_uri)

    # ✅ 실험 확보
    experiment_name = get_experiment_name()
    experiment = client.get_experiment_by_name(experiment_name)

    if experiment:
        experiment_id = experiment.experiment_id
        print(f"[DEBUG] ✅ 기존 실험 ID 사용: {experiment_id}")
    else:
        experiment_id = client.create_experiment(experiment_name)
        print(f"[DEBUG] ✅ 새 실험 생성 ID: {experiment_id}")

    # ✅ 데이터셋 준비
    X, y = load_iris(return_X_y=True)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

    # ✅ Run 시작
    try: 
        with mlflow.start_run(experiment_id=experiment_id) as run:
            run_id = run.info.run_id
            print(f"[DEBUG] ✅ run_id: {run_id}")
    
            # ✅ 학습
            clf = LogisticRegression(C=C, max_iter=max_iter)
            clf.fit(X_train, y_train)
            y_pred = clf.predict(X_test)
            acc = accuracy_score(y_test, y_pred)
    
            # ✅ 로깅
            mlflow.log_param("C", C)
            mlflow.log_param("max_iter", max_iter)
            mlflow.log_metric("accuracy", acc)
            mlflow.sklearn.log_model(clf, "model")
    
            return acc, run_id
    except Exception as e:
        print(f"[ERROR] ❌ 학습 또는 로깅 중 오류: {e}")
        raise
