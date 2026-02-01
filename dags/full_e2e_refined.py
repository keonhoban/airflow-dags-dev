from __future__ import annotations

from datetime import datetime, timedelta
from pendulum import timezone

from airflow import DAG
from airflow.operators.python import PythonOperator, BranchPythonOperator
from airflow.operators.empty import EmptyOperator
from airflow.operators.bash import BashOperator
from airflow.sensors.python import PythonSensor

from mlops.slack import alert_slack
from mlops.dp.tasks import (
    task_extract_raw_data,
    task_validate_data,
    task_build_features,
    task_store_features,
    task_summarize_run,
)
from mlops.ml.train import task_train_and_evaluate
from mlops.ml.register import task_register_model
from mlops.ml.sensor import sensor_model_ready
from mlops.ml.fastapi_reload import task_fastapi_reload
from mlops.triton.deploy import (
    task_snapshot_current,
    task_materialize_repo,
    task_triton_load,
    task_triton_ready,
    task_triton_infer_smoke,
    task_commit_current,
    task_rollback_minimal,
)
from mlops.config import cfg


kst = timezone("Asia/Seoul")

default_args = {
    "start_date": datetime(2025, 1, 1, tzinfo=kst),
    "retries": 1,
    "retry_delay": timedelta(minutes=2),
}

with DAG(
    dag_id="full_e2e_refined",
    default_args=default_args,
    schedule=None,
    catchup=False,
    max_active_runs=1,
    tags=["e2e", "refined", "feast", "triton", "mlops"],
    on_failure_callback=alert_slack,
) as dag:

    # -----------------------
    # Data pipeline
    # -----------------------
    extract_raw = PythonOperator(
        task_id="extract_raw_data",
        python_callable=task_extract_raw_data,
    )

    validate = PythonOperator(
        task_id="validate_data",
        python_callable=task_validate_data,
    )

    build = PythonOperator(
        task_id="build_features",
        python_callable=task_build_features,
    )

    store = PythonOperator(
        task_id="store_features",
        python_callable=task_store_features,
    )

    feast_apply = BashOperator(
        task_id="feast_apply",
        bash_command="""
        set -euo pipefail
        cd /opt/airflow/dags/repo/dags/feast_repo
        feast apply
        """.strip(),
    )

    feast_materialize = BashOperator(
        task_id="feast_materialize",
        bash_command="""
        set -euo pipefail
        cd /opt/airflow/dags/repo/dags/feast_repo
        feast materialize "{{ macros.ds_add(ds, -1) }}T00:00:00" "{{ ds }}T23:59:59"
        """.strip(),
    )

    # -----------------------
    # Train
    # -----------------------
    train = PythonOperator(
        task_id="train_and_evaluate",
        python_callable=task_train_and_evaluate,
    )

    # -----------------------
    # Branch: promote or shadow
    # -----------------------
    def _branch(**context):
        ti = context["ti"]
        acc = float(ti.xcom_pull(task_ids="train_and_evaluate", key="accuracy") or 0.0)
        thr = float(cfg("accuracy_threshold", "0.0"))
        return "promote_start" if acc >= thr else "shadow_start"

    branch = BranchPythonOperator(
        task_id="branch_promote_or_shadow",
        python_callable=_branch,
    )

    promote_start = EmptyOperator(task_id="promote_start")
    shadow_start = EmptyOperator(task_id="shadow_start")

    # -----------------------
    # Promote path
    # -----------------------
    register = PythonOperator(
        task_id="register_model",
        python_callable=task_register_model,
    )

    model_ready = PythonSensor(
        task_id="wait_model_ready",
        python_callable=sensor_model_ready,
        poke_interval=10,
        timeout=180,
        mode="reschedule",
    )

    # -----------------------
    # Shadow path (no registry change)
    # -----------------------
    # just pass

    # -----------------------
    # Deploy (promotion/shadow each has its own chain)
    # -----------------------
    snap_p = PythonOperator(
        task_id="snapshot_current_promotion",
        python_callable=task_snapshot_current,
    )
    mat_p = PythonOperator(
        task_id="materialize_repo_promotion",
        python_callable=task_materialize_repo,
    )
    load_p = PythonOperator(
        task_id="triton_load_promotion",
        python_callable=task_triton_load,
    )
    ready_p = PythonOperator(
        task_id="triton_ready_promotion",
        python_callable=task_triton_ready,
    )
    smoke_p = PythonOperator(
        task_id="triton_smoke_promotion",
        python_callable=task_triton_infer_smoke,
    )

    commit = PythonOperator(
        task_id="commit_current",
        python_callable=task_commit_current,
    )
    reload_api = PythonOperator(
        task_id="fastapi_reload",
        python_callable=task_fastapi_reload,
    )

    rb_p = PythonOperator(
        task_id="rollback_promotion",
        python_callable=task_rollback_minimal,
    )

    snap_s = PythonOperator(
        task_id="snapshot_current_shadow",
        python_callable=task_snapshot_current,
    )
    mat_s = PythonOperator(
        task_id="materialize_repo_shadow",
        python_callable=task_materialize_repo,
    )
    load_s = PythonOperator(
        task_id="triton_load_shadow",
        python_callable=task_triton_load,
    )
    ready_s = PythonOperator(
        task_id="triton_ready_shadow",
        python_callable=task_triton_ready,
    )
    smoke_s = PythonOperator(
        task_id="triton_smoke_shadow",
        python_callable=task_triton_infer_smoke,
    )

    rb_s = PythonOperator(
        task_id="rollback_shadow",
        python_callable=task_rollback_minimal,
    )

    summarize = PythonOperator(
        task_id="summarize_run",
        python_callable=task_summarize_run,
        trigger_rule="all_done",
    )

    # -----------------------
    # Wiring
    # -----------------------
    extract_raw >> validate >> build >> store >> feast_apply >> feast_materialize
    feast_materialize >> train >> branch
    branch >> promote_start >> register >> model_ready >> snap_p
    branch >> shadow_start >> snap_s

    # promotion deploy chain
    snap_p >> mat_p >> load_p >> ready_p >> smoke_p >> commit >> reload_api

    # shadow deploy chain (no commit/reload)
    snap_s >> mat_s >> load_s >> ready_s >> smoke_s

    # rollback only for deploy stages (NOT commit/reload)
    # promotion rollback
    [mat_p, load_p, ready_p, smoke_p] >> rb_p
    # shadow rollback
    [mat_s, load_s, ready_s, smoke_s] >> rb_s

    # summary
    [reload_api, smoke_s, rb_p, rb_s] >> summarize

