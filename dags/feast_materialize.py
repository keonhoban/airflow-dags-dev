from __future__ import annotations

from datetime import datetime, timedelta
from typing import Literal

import pendulum
from airflow import DAG
from airflow.models import Variable
from airflow.providers.cncf.kubernetes.operators.pod import KubernetesPodOperator
from kubernetes.client import models as k8s

KST = pendulum.timezone("Asia/Seoul")

DEFAULT_ARGS = {
    "owner": "airflow",
    "retries": 1,
    "retry_delay": timedelta(minutes=3),
}

FEAST_IMAGE = Variable.get("FEAST_IMAGE", default_var="hoizz/feast-server:0.40.1-s3fs")
AWS_REGION = Variable.get("AWS_REGION", default_var="ap-northeast-2")
FEAST_REPO_CONFIGMAP = Variable.get("FEAST_REPO_CONFIGMAP", default_var="feast-repo")

AWS_CRED_SECRET_DEV = Variable.get("FEAST_AWS_CRED_SECRET_DEV", default_var="aws-credentials-secret")
AWS_CRED_SECRET_PROD = Variable.get("FEAST_AWS_CRED_SECRET_PROD", default_var="aws-credentials-secret")

AWS_PROFILE_DEV = Variable.get("FEAST_AWS_PROFILE_DEV", default_var="rotator-dev")
AWS_PROFILE_PROD = Variable.get("FEAST_AWS_PROFILE_PROD", default_var="rotator-prod")

FEAST_FULL_START = Variable.get("FEAST_FULL_START", default_var="2026-01-01T00:00:00")

# fsnotify/FD 한계 완화 (필요시 Airflow Variable로 튜닝)
ULIMIT_NOFILE = int(Variable.get("FEAST_ULIMIT_NOFILE", default_var="8192"))

Mode = Literal["incremental", "full"]


def _build_repo_volumes(*, repo_configmap_name: str):
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


def _build_aws_cred_volume(*, aws_cred_secret_name: str):
    vol_aws = k8s.V1Volume(
        name="aws-credentials",
        secret=k8s.V1SecretVolumeSource(secret_name=aws_cred_secret_name),
    )
    vm_aws = k8s.V1VolumeMount(
        name="aws-credentials",
        mount_path="/root/.aws",
        read_only=True,
    )
    return vol_aws, vm_aws


def _build_cmd(*, mode: Mode, full_start: str) -> str:
    """
    - ConfigMap atomic writer(..data) 오염 방지
    - 문자열 결합으로 커맨드가 붙는 사고(apply#) 재발 방지: 항상 라인 단위 join
    - fsnotify open files 완화: ulimit (가능하면)
    """
    lines: list[str] = [
        "set -euo pipefail",
        "set -x",
        "",
        f"ulimit -n {ULIMIT_NOFILE} || true",
        "",
        # workdir 완전 정리 (숨김 포함)
        "find /feast-repo -mindepth 1 -maxdepth 1 -exec rm -rf {} +",
        "",
        # ConfigMap -> workdir 복사
        # '/.' 금지: ..data/..timestamp 유입 가능
        "cp -aL /feast-repo-src/* /feast-repo/",
        "",
        # 방어: atomic writer 디렉토리 유입 시 즉시 실패
        "test ! -e /feast-repo/..data",
        "",
        "ls -al /feast-repo",
        "test -f /feast-repo/feature_store.yaml",
        "test -f /feast-repo/repo.py",
        "",
        "cd /feast-repo",
        "",
        # ✅ 반드시 단독 라인 (apply# 사고 재발 방지)
        "feast apply",
        "",
    ]

    if mode == "incremental":
        lines += [
            "# incremental: 마지막 materialize 시점부터 now까지",
            'feast materialize-incremental "$(date -u +"%Y-%m-%dT%H:%M:%S")"',
        ]
    elif mode == "full":
        lines += [
            "# full refresh: START -> now",
            f'START="{full_start}"',
            'END="$(date -u +"%Y-%m-%dT%H:%M:%S")"',
            'echo "materialize $START -> $END"',
            'feast materialize "$START" "$END"',
        ]
    else:
        raise ValueError(f"unknown mode: {mode}")

    # join으로만 합치면 커맨드가 붙을 일이 없습니다.
    return "\n".join(lines) + "\n"


def _feast_task(
    *,
    dag: DAG,
    task_id: str,
    namespace: str,
    aws_cred_secret_name: str,
    aws_profile: str,
    mode: Mode,
) -> KubernetesPodOperator:
    volumes_repo, mounts_repo = _build_repo_volumes(repo_configmap_name=FEAST_REPO_CONFIGMAP)
    vol_aws, vm_aws = _build_aws_cred_volume(aws_cred_secret_name=aws_cred_secret_name)

    cmd = _build_cmd(mode=mode, full_start=FEAST_FULL_START)

    return KubernetesPodOperator(
        dag=dag,
        task_id=task_id,
        name=task_id,
        namespace=namespace,
        image=FEAST_IMAGE,
        cmds=["/bin/sh", "-c"],
        arguments=[cmd],
        env_vars={
            "AWS_SHARED_CREDENTIALS_FILE": "/root/.aws/credentials",
            "AWS_PROFILE": aws_profile,
            "AWS_DEFAULT_REGION": AWS_REGION,
            "AWS_REGION": AWS_REGION,
        },
        volumes=[*volumes_repo, vol_aws],
        volume_mounts=[*mounts_repo, vm_aws],
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
        aws_cred_secret_name=AWS_CRED_SECRET_DEV,
        aws_profile=AWS_PROFILE_DEV,
        mode="incremental",
    )

    feast_prod_incremental = _feast_task(
        dag=dag,
        task_id="feast_prod_incremental",
        namespace="feature-store-prod",
        aws_cred_secret_name=AWS_CRED_SECRET_PROD,
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
        aws_cred_secret_name=AWS_CRED_SECRET_DEV,
        aws_profile=AWS_PROFILE_DEV,
        mode="full",
    )

    feast_prod_full = _feast_task(
        dag=dag_full,
        task_id="feast_prod_full_refresh",
        namespace="feature-store-prod",
        aws_cred_secret_name=AWS_CRED_SECRET_PROD,
        aws_profile=AWS_PROFILE_PROD,
        mode="full",
    )

    feast_dev_full >> feast_prod_full

