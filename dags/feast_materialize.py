from __future__ import annotations

from datetime import datetime, timedelta
from typing import Literal

import pendulum
from airflow import DAG
from airflow.models import Variable
from airflow.providers.cncf.kubernetes.operators.pod import KubernetesPodOperator
from kubernetes.client import models as k8s

KST = pendulum.timezone("Asia/Seoul")
Mode = Literal["incremental", "full"]

DEFAULT_ARGS = {
    "owner": "airflow",
    "retries": 1,
    "retry_delay": timedelta(minutes=3),
}

# =========================
# Minimal Config (Variables)
# =========================
# "있으면 사용하고, 없으면 안전한 기본값"만 남깁니다.
FEAST_IMAGE = Variable.get("FEAST_IMAGE", default_var="hoizz/feast-server:0.40.1-s3fs")

# feature-store 네임스페이스에 존재하는 ConfigMap 이름 (charts/feast/templates/feast-repo-configmap.yaml)
FEAST_REPO_CONFIGMAP = Variable.get("FEAST_REPO_CONFIGMAP", default_var="feast-repo")

AWS_REGION = Variable.get("AWS_REGION", default_var="ap-northeast-2")

# Airflow가 Pod를 띄울 때 사용할 AWS credential Secret 이름(각 feature-store ns에 있어야 함)
AWS_CRED_SECRET_DEV = Variable.get("FEAST_AWS_CRED_SECRET_DEV", default_var="aws-credentials-secret")
AWS_CRED_SECRET_PROD = Variable.get("FEAST_AWS_CRED_SECRET_PROD", default_var="aws-credentials-secret")

AWS_PROFILE_DEV = Variable.get("FEAST_AWS_PROFILE_DEV", default_var="rotator-dev")
AWS_PROFILE_PROD = Variable.get("FEAST_AWS_PROFILE_PROD", default_var="rotator-prod")

# Full refresh 시작점(수동 DAG)
FEAST_FULL_START = Variable.get("FEAST_FULL_START", default_var="2026-01-01T00:00:00")

# (선택) KPO pod가 사용할 ServiceAccount (RBAC와 맞춰야 함)
KPO_SERVICE_ACCOUNT = Variable.get("FEAST_KPO_SERVICE_ACCOUNT", default_var=None)

# (선택) 리소스 제한 (과하면 제거 가능)
CPU_REQUEST = Variable.get("FEAST_CPU_REQUEST", default_var="200m")
MEM_REQUEST = Variable.get("FEAST_MEM_REQUEST", default_var="512Mi")
CPU_LIMIT = Variable.get("FEAST_CPU_LIMIT", default_var="1000m")
MEM_LIMIT = Variable.get("FEAST_MEM_LIMIT", default_var="2Gi")


def _repo_volumes(repo_configmap_name: str):
    """
    ConfigMap은 k8s atomic writer(..data) 때문에 직접 작업dir로 쓰면 사고가 납니다.
    - /feast-repo-src: ConfigMap read-only mount
    - /feast-repo: emptyDir(workdir)
    """
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
    """
    주의:
    - Secret에는 /root/.aws/credentials 로 읽힐 'credentials' 키가 존재해야 합니다.
    """
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
    실무 기준 핵심만:
    - sh-only (bash/pipefail 의존 제거)
    - repo 복사 + 필수파일 검증 + apply + materialize
    """
    lines: list[str] = [
        "set -eu",
        "set -x",
        "",
        # 작업dir 정리
        "rm -rf /feast-repo/*",
        "",
        # ConfigMap -> workdir 복사
        "cp -aL /feast-repo-src/. /feast-repo/",
        "",
        # 방어 (필수 파일 존재)
        "test -f /feast-repo/feature_store.yaml",
        "test -f /feast-repo/repo.py",
        "cd /feast-repo",
        "",
        # apply
        "feast apply",
        "",
    ]

    if mode == "incremental":
        lines += [
            'NOW="$(date -u +"%Y-%m-%dT%H:%M:%S")"',
            'echo "materialize-incremental -> $NOW"',
            'feast materialize-incremental "$NOW"',
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
    aws_secret: str,
    aws_profile: str,
    mode: Mode,
) -> KubernetesPodOperator:
    repo_vols, repo_mounts = _repo_volumes(FEAST_REPO_CONFIGMAP)
    vol_aws, vm_aws = _aws_cred_volume(aws_secret)

    script = _build_script(mode=mode, full_start=FEAST_FULL_START)

    resources = k8s.V1ResourceRequirements(
        requests={"cpu": CPU_REQUEST, "memory": MEM_REQUEST},
        limits={"cpu": CPU_LIMIT, "memory": MEM_LIMIT},
    )

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
        container_resources=resources,
        labels={
            "app": "feast-materialize",
            "env": "dev" if "dev" in namespace else "prod",
        },
        service_account_name=KPO_SERVICE_ACCOUNT or None,
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

