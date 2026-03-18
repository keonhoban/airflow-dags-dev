"""
DAG integrity tests.

목적:
- DAG 파일에 import/문법 오류가 없는지 CI 단계에서 확인
- 기대하는 DAG ID가 실제로 등록되는지 검증
- DagBag 로딩 에러 0개를 강제

실행 방법:
    pytest tests/test_dag_integrity.py -v

전제:
    AIRFLOW_HOME 또는 환경에 맞는 airflow DB가 초기화되어 있어야 합니다.
    CI 환경에서는 sqlite + airflow db init 으로 충분합니다.
"""
from __future__ import annotations

import os
import pytest
from airflow.models import DagBag

DAGS_FOLDER = os.path.join(os.path.dirname(__file__), "..", "dags")

EXPECTED_DAG_IDS = {
    "e2e_full",
    "dp_feature_pipeline",
    "feast_materialize_dev",
    "feast_full_refresh_dev_manual",
    "rollback_manual",
}


@pytest.fixture(scope="module")
def dagbag():
    """DagBag을 한 번만 로드해서 모든 테스트에서 공유한다."""
    return DagBag(dag_folder=DAGS_FOLDER, include_examples=False)


def test_no_import_errors(dagbag):
    """DAG 파일 import 에러가 0개여야 한다."""
    errors = dagbag.import_errors
    assert errors == {}, (
        "DAG import 에러 발생:\n"
        + "\n".join(f"  {path}: {err}" for path, err in errors.items())
    )


def test_expected_dag_ids_exist(dagbag):
    """기대하는 DAG ID가 모두 DagBag에 등록되어 있어야 한다."""
    loaded = set(dagbag.dag_ids)
    missing = EXPECTED_DAG_IDS - loaded
    assert not missing, f"DagBag에서 누락된 DAG ID: {missing}"


@pytest.mark.parametrize("dag_id", sorted(EXPECTED_DAG_IDS))
def test_dag_has_tasks(dagbag, dag_id):
    """각 DAG에 태스크가 1개 이상 존재해야 한다."""
    dag = dagbag.get_dag(dag_id)
    if dag is None:
        pytest.skip(f"DAG '{dag_id}' not loaded (import error가 별도 테스트에서 잡힘)")
    assert len(dag.tasks) > 0, f"DAG '{dag_id}'에 태스크가 없습니다"


def test_e2e_full_key_task_ids(dagbag):
    """e2e_full DAG의 핵심 task_id가 존재하는지 확인한다."""
    dag = dagbag.get_dag("e2e_full")
    if dag is None:
        pytest.skip("e2e_full DAG not loaded")

    task_ids = {t.task_id for t in dag.tasks}
    required = {
        "train_and_evaluate",
        "check_result",
        "observe_post_deploy_metrics",
        "rollback_minimal",
    }
    missing = required - task_ids
    assert not missing, f"e2e_full에서 누락된 task_id: {missing}"


def test_e2e_full_max_active_runs(dagbag):
    """e2e_full은 동시 실행을 막기 위해 max_active_runs=1이어야 한다."""
    dag = dagbag.get_dag("e2e_full")
    if dag is None:
        pytest.skip("e2e_full DAG not loaded")
    assert dag.max_active_runs == 1, (
        f"e2e_full.max_active_runs={dag.max_active_runs} (expected 1)"
    )


def test_e2e_full_catchup_disabled(dagbag):
    """e2e_full은 과거 실행을 자동으로 채우지 않아야 한다."""
    dag = dagbag.get_dag("e2e_full")
    if dag is None:
        pytest.skip("e2e_full DAG not loaded")
    assert dag.catchup is False, "e2e_full.catchup should be False"
