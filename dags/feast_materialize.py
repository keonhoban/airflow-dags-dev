from __future__ import annotations

from datetime import datetime, timedelta
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

# 공통: Feast 작업 Pod에서 사용할 이미지
FEAST_IMAGE = Variable.get("FEAST_IMAGE", default_var="hoizz/feast-server:0.40.1-s3fs")
AWS_REGION = Variable.get("AWS_REGION", default_var="ap-northeast-2")

# 공통: repo configmap 이름
FEAST_REPO_CONFIGMAP = Variable.get("FEAST_REPO_CONFIGMAP", default_var="feast-repo")

# 공통: AWS credentials secret 이름 (dev/prod가 다르면 변수로 분리)
AWS_CRED_SECRET_DEV = Variable.get("FEAST_AWS_CRED_SECRET_DEV", default_var="aws-credentials-secret")
AWS_CRED_SECRET_PROD = Variable.get("FEAST_AWS_CRED_SECRET_PROD", default_var="aws-credentials-secret")

# ✅ 핵심: credentials 파일 안에 default가 없고 [rotator-dev]/[rotator-prod]만 있으므로 profile을 명시해야 함
AWS_PROFILE_DEV = Variable.get("FEAST_AWS_PROFILE_DEV", default_var="rotator-dev")
AWS_PROFILE_PROD = Variable.get("FEAST_AWS_PROFILE_PROD", default_var="rotator-prod")


def _feast_task(
    *,
    dag: DAG,
    task_id: str,
    namespace: str,
    aws_cred_secret_name: str,
    aws_profile: str,
    mode: str,
) -> KubernetesPodOperator:
    """
    mode:
      - "incremental": apply + materialize-incremental(now)
      - "full": apply + materialize(START, now)  # 복구용
    """

    # ConfigMap source mount (숨김 디렉토리 구조 포함)
    vol_repo_src = k8s.V1Volume(
        name="feast-repo-src",
        config_map=k8s.V1ConfigMapVolumeSource(name=FEAST_REPO_CONFIGMAP),
    )
    vm_repo_src = k8s.V1VolumeMount(
        name="feast-repo-src",
        mount_path="/feast-repo-src",
        read_only=True,
    )

    # Clean working dir (실행은 여기서)
    vol_repo_work = k8s.V1Volume(
        name="feast-repo-work",
        empty_dir=k8s.V1EmptyDirVolumeSource(),
    )
    vm_repo_work = k8s.V1VolumeMount(
        name="feast-repo-work",
        mount_path="/feast-repo",
        read_only=False,
    )

    # AWS credentials secret 을 /root/.aws 로 마운트
    vol_aws = k8s.V1Volume(
        name="aws-credentials",
        secret=k8s.V1SecretVolumeSource(secret_name=aws_cred_secret_name),
    )
    vm_aws = k8s.V1VolumeMount(
        name="aws-credentials",
        mount_path="/root/.aws",
        read_only=True,
    )

    # 실행 커맨드 구성
    if mode == "incremental":
        cmd = r"""
        set -eux
        rm -rf /feast-repo/*
        cp -R /feast-repo-src/* /feast-repo/
        cd /feast-repo
        feast apply
        feast materialize-incremental "$(date -u +"%Y-%m-%dT%H:%M:%S")"
        """
    elif mode == "full":
        start = Variable.get("FEAST_FULL_START", default_var="2026-01-01T00:00:00")
        cmd = rf"""
        set -eux
        rm -rf /feast-repo/*
        cp -R /feast-repo-src/* /feast-repo/
        cd /feast-repo
        feast apply
        START="{start}"
        END="$(date -u +"%Y-%m-%dT%H:%M:%S")"
        echo "materialize $START -> $END"
        feast materialize "$START" "$END"
        """
    else:
        raise ValueError(f"unknown mode: {mode}")

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
        volumes=[vol_repo_src, vol_repo_work, vol_aws],
        volume_mounts=[vm_repo_src, vm_repo_work, vm_aws],
        get_logs=True,
        is_delete_operator_pod=True,
        startup_timeout_seconds=300,
    )

with DAG(
    dag_id="feast_materialize",
    start_date=datetime(2026, 2, 1, tzinfo=KST),
    schedule="*/30 * * * *",  # 30분마다
    catchup=False,
    default_args=DEFAULT_ARGS,
    tags=["feature-store", "feast", "materialize"],
) as dag:
    # dev incremental
    feast_dev_incremental = _feast_task(
        dag=dag,
        task_id="feast_dev_incremental",
        namespace="feature-store-dev",
        aws_cred_secret_name=AWS_CRED_SECRET_DEV,
        aws_profile=AWS_PROFILE_DEV,
        mode="incremental",
    )

    # prod incremental
    feast_prod_incremental = _feast_task(
        dag=dag,
        task_id="feast_prod_incremental",
        namespace="feature-store-prod",
        aws_cred_secret_name=AWS_CRED_SECRET_PROD,
        aws_profile=AWS_PROFILE_PROD,
        mode="incremental",
    )

    # 처음엔 dev → prod 순서가 운영적으로 안전
    feast_dev_incremental >> feast_prod_incremental


with DAG(
    dag_id="feast_full_refresh_manual",
    start_date=datetime(2026, 2, 1, tzinfo=KST),
    schedule=None,  # 수동 실행 전용
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

