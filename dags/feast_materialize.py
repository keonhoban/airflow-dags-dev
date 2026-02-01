from __future__ import annotations

from datetime import datetime, timedelta
from typing import Literal

import pendulum
from airflow import DAG
from airflow.models import Variable
from airflow.providers.cncf.kubernetes.operators.pod import KubernetesPodOperator
from kubernetes.client import models as k8s

KST = pendulum.timezone("Asia/Seoul")

DEFAULT_ARGS = {"owner": "airflow", "retries": 1, "retry_delay": timedelta(minutes=3)}

Mode = Literal["incremental", "full"]

# ---- Airflow Variables ----
FEAST_IMAGE = Variable.get("FEAST_IMAGE", default_var="hoizz/feast-server:0.40.1-s3fs")
AWS_REGION = Variable.get("AWS_REGION", default_var="ap-northeast-2")

# ✅ dev/prod repo 분리 권장
FEAST_REPO_CONFIGMAP_DEV = Variable.get("FEAST_REPO_CONFIGMAP_DEV", default_var="feast-repo-dev")
FEAST_REPO_CONFIGMAP_PROD = Variable.get("FEAST_REPO_CONFIGMAP_PROD", default_var="feast-repo-prod")

AWS_CRED_SECRET_DEV = Variable.get("FEAST_AWS_CRED_SECRET_DEV", default_var="aws-credentials-secret")
AWS_CRED_SECRET_PROD = Variable.get("FEAST_AWS_CRED_SECRET_PROD", default_var="aws-credentials-secret")

AWS_PROFILE_DEV = Variable.get("FEAST_AWS_PROFILE_DEV", default_var="rotator-dev")
AWS_PROFILE_PROD = Variable.get("FEAST_AWS_PROFILE_PROD", default_var="rotator-prod")

FEAST_FULL_START = Variable.get("FEAST_FULL_START", default_var="2026-01-01T00:00:00")
ULIMIT_NOFILE = int(Variable.get("FEAST_ULIMIT_NOFILE", default_var="8192"))

# non-root 안전 경로
AWS_DIR = "/tmp/.aws"
AWS_SHARED_CREDENTIALS_FILE = f"{AWS_DIR}/credentials"


def _repo_volumes(repo_configmap_name: str):
    vol_repo_src = k8s.V1Volume(
        name="feast-repo-src",
        config_map=k8s.V1ConfigMapVolumeSource(name=repo_configmap_name),
    )
    vm_repo_src = k8s.V1VolumeMount(
        name="feast-repo-src",
        mount_path="/feast-repo-src",
        read_only=True,
    )

    vol_repo_work = k8s.V1Volume(
        name="feast-repo-work",
        empty_dir=k8s.V1EmptyDirVolumeSource(),
    )
    vm_repo_work = k8s.V1VolumeMount(
        name="feast-repo-work",
        mount_path="/feast-repo",
        read_only=False,
    )
    return [vol_repo_src, vol_repo_work], [vm_repo_src, vm_repo_work]


def _aws_cred_volume(secret_name: str):
    vol_aws = k8s.V1Volume(
        name="aws-credentials",
        secret=k8s.V1SecretVolumeSource(secret_name=secret_name),
    )
    vm_aws = k8s.V1VolumeMount(
        name="aws-credentials",
        mount_path=AWS_DIR,
        read_only=True,
    )
    return vol_aws, vm_aws


def _build_script(mode: Mode, full_start: str) -> str:
    lines: list[str] = [
        "set -eu",
        "set -x",
        f"ulimit -n {ULIMIT_NOFILE} || true",
        "",
        "rm -rf /feast-repo/*",
        "cp -aL /feast-repo-src/* /feast-repo/",
        "",
        "ls -al /feast-repo",
        "test -f /feast-repo/feature_store.yaml",
        "test -f /feast-repo/repo.py",
        "",
        "cd /feast-repo",
        "feast apply",
        "",
    ]

    if mode == "incremental":
        lines += [
            'END="$(date -u +"%Y-%m-%dT%H:%M:%S")"',
            'feast materialize-incremental "$END"',
        ]
    else:
        lines += [
            f'START="{full_start}"',
            'END="$(date -u +"%Y-%m-%dT%H:%M:%S")"',
            'echo "materialize $START -> $END"',
            'feast materialize "$START" "$END"',
        ]

    return "\n".join(lines) + "\n"


def _feast_task(
    *,
    dag: DAG,
    task_id: str,
    namespace: str,
    repo_configmap: str,
    aws_secret: str,
    aws_profile: str,
    mode: Mode,
) -> KubernetesPodOperator:
    repo_vols, repo_mounts = _repo_volumes(repo_configmap)
    vol_aws, vm_aws = _aws_cred_volume(aws_secret)
    script = _build_script(mode=mode, full_start=FEAST_FULL_START)

    return KubernetesPodOperator(
        dag=dag,
        task_id=task_id,
        name=task_id,
        namespace=namespace,
        image=FEAST_IMAGE,
        cmds=["/bin/sh", "-c"],
        arguments=[script],
        env_vars={
            "AWS_SHARED_CREDENTIALS_FILE": AWS_SHARED_CREDENTIALS_FILE,
            "AWS_PROFILE": aws_profile,
            "AWS_DEFAULT_REGION": AWS_REGION,
            "AWS_REGION": AWS_REGION,
        },
        volumes=[*repo_vols, vol_aws],
        volume_mounts=[*repo_mounts, vm_aws],
        get_logs=True,
        is_delete_operator_pod=True,
        startup_timeout_seconds=300,
    )


with DAG(
    dag_id="feast_materialize",
    start_date=datetime(2026, 2, 1, tzinfo=KST),
    schedule="*/30 * * * *",
    catchup=False,
    default_args=DEFAULT_ARGS,
    tags=["feature-store", "feast", "materialize"],
) as dag:
    feast_dev_incremental = _feast_task(
        dag=dag,
        task_id="feast_dev_incremental",
        namespace="feature-store-dev",
        repo_configmap=FEAST_REPO_CONFIGMAP_DEV,
        aws_secret=AWS_CRED_SECRET_DEV,
        aws_profile=AWS_PROFILE_DEV,
        mode="incremental",
    )

    feast_prod_incremental = _feast_task(
        dag=dag,
        task_id="feast_prod_incremental",
        namespace="feature-store-prod",
        repo_configmap=FEAST_REPO_CONFIGMAP_PROD,
        aws_secret=AWS_CRED_SECRET_PROD,
        aws_profile=AWS_PROFILE_PROD,
        mode="incremental",
    )

    feast_dev_incremental >> feast_prod_incremental


with DAG(
    dag_id="feast_full_refresh_manual",
    start_date=datetime(2026, 2, 1, tzinfo=KST),
    schedule=None,
    catchup=False,
    default_args=DEFAULT_ARGS,
    tags=["feature-store", "feast", "recovery"],
) as dag_full:
    feast_dev_full = _feast_task(
        dag=dag_full,
        task_id="feast_dev_full_refresh",
        namespace="feature-store-dev",
        repo_configmap=FEAST_REPO_CONFIGMAP_DEV,
        aws_secret=AWS_CRED_SECRET_DEV,
        aws_profile=AWS_PROFILE_DEV,
        mode="full",
    )

    feast_prod_full = _feast_task(
        dag=dag_full,
        task_id="feast_prod_full_refresh",
        namespace="feature-store-prod",
        repo_configmap=FEAST_REPO_CONFIGMAP_PROD,
        aws_secret=AWS_CRED_SECRET_PROD,
        aws_profile=AWS_PROFILE_PROD,
        mode="full",
    )

    feast_dev_full >> feast_prod_full

