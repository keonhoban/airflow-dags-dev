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

# ---- Config (Airflow Variables로 튜닝 가능) ----
FEAST_IMAGE = Variable.get("FEAST_IMAGE", default_var="hoizz/feast-server:0.40.1-s3fs")
AWS_REGION = Variable.get("AWS_REGION", default_var="ap-northeast-2")
FEAST_REPO_CONFIGMAP = Variable.get("FEAST_REPO_CONFIGMAP", default_var="feast-repo")

AWS_CRED_SECRET_DEV = Variable.get("FEAST_AWS_CRED_SECRET_DEV", default_var="aws-credentials-secret")
AWS_CRED_SECRET_PROD = Variable.get("FEAST_AWS_CRED_SECRET_PROD", default_var="aws-credentials-secret")

AWS_PROFILE_DEV = Variable.get("FEAST_AWS_PROFILE_DEV", default_var="rotator-dev")
AWS_PROFILE_PROD = Variable.get("FEAST_AWS_PROFILE_PROD", default_var="rotator-prod")

FEAST_FULL_START = Variable.get("FEAST_FULL_START", default_var="2026-01-01T00:00:00")

ULIMIT_NOFILE = int(Variable.get("FEAST_ULIMIT_NOFILE", default_var="8192"))

Mode = Literal["incremental", "full"]


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
        mount_path="/root/.aws",
        read_only=True,
    )
    return vol_aws, vm_aws


def _build_script(mode: Mode, full_start: str) -> str:
    """
    핵심 목표:
    - /bin/sh 환경에서도 깨지지 않기 (pipefail 금지)
    - bash가 있으면 pipefail 활성화 (자동 폴백)
    - ConfigMap atomic writer(..data) 오염 방지
    - 커맨드 붙는 사고 방지: 라인 join
    """
    lines: list[str] = [
        # 1) bash 존재하면 bash로 재실행 (pipefail 가능), 아니면 sh로 계속
        'if command -v bash >/dev/null 2>&1; then exec bash -lc "$0"; fi',
        "",
        # 2) sh 안전 옵션 (pipefail 없음)
        "set -eu",
        "set -x",
        "",
        # 3) ulimit (가능하면)
        f"ulimit -n {ULIMIT_NOFILE} || true",
        "",
        # 4) workdir 정리
        "find /feast-repo -mindepth 1 -maxdepth 1 -exec rm -rf {} +",
        "",
        # 5) ConfigMap -> workdir 복사
        "cp -aL /feast-repo-src/* /feast-repo/",
        "",
        # 6) 방어
        "test ! -e /feast-repo/..data",
        "ls -al /feast-repo",
        "test -f /feast-repo/feature_store.yaml",
        "test -f /feast-repo/repo.py",
        "",
        "cd /feast-repo",
        "",
        # 7) Feast apply (단독 라인)
        "feast apply",
        "",
    ]

    if mode == "incremental":
        lines += [
            "# incremental: 마지막 materialize 시점부터 now까지",
            'feast materialize-incremental "$(date -u +"%Y-%m-%dT%H:%M:%S")"',
        ]
    else:
        lines += [
            "# full refresh: START -> now",
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
    aws_secret: str,
    aws_profile: str,
    mode: Mode,
) -> KubernetesPodOperator:
    repo_vols, repo_mounts = _repo_volumes(FEAST_REPO_CONFIGMAP)
    vol_aws, vm_aws = _aws_cred_volume(aws_secret)

    script = _build_script(mode=mode, full_start=FEAST_FULL_START)

    # /bin/sh로 실행하되, 스크립트 첫 줄에서 bash 있으면 자동으로 bash로 재실행
    return KubernetesPodOperator(
        dag=dag,
        task_id=task_id,
        name=task_id,
        namespace=namespace,
        image=FEAST_IMAGE,
        cmds=["/bin/sh", "-c"],
        arguments=[script],
        env_vars={
            "AWS_SHARED_CREDENTIALS_FILE": "/root/.aws/credentials",
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
        aws_secret=AWS_CRED_SECRET_DEV,
        aws_profile=AWS_PROFILE_DEV,
        mode="incremental",
    )

    feast_prod_incremental = _feast_task(
        dag=dag,
        task_id="feast_prod_incremental",
        namespace="feature-store-prod",
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
        aws_secret=AWS_CRED_SECRET_DEV,
        aws_profile=AWS_PROFILE_DEV,
        mode="full",
    )

    feast_prod_full = _feast_task(
        dag=dag_full,
        task_id="feast_prod_full_refresh",
        namespace="feature-store-prod",
        aws_secret=AWS_CRED_SECRET_PROD,
        aws_profile=AWS_PROFILE_PROD,
        mode="full",
    )

    feast_dev_full >> feast_prod_full

