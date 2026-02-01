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

# ---- Variables (운영에서 바뀌는 값은 전부 Variable로) ----
FEAST_IMAGE = Variable.get("FEAST_IMAGE", default_var="hoizz/feast-server:0.40.1-s3fs")

FEAST_REPO_CONFIGMAP = Variable.get("FEAST_REPO_CONFIGMAP", default_var="feast-repo")

AWS_REGION = Variable.get("AWS_REGION", default_var="ap-northeast-2")
AWS_CRED_SECRET_DEV = Variable.get("FEAST_AWS_CRED_SECRET_DEV", default_var="aws-credentials-secret")
AWS_CRED_SECRET_PROD = Variable.get("FEAST_AWS_CRED_SECRET_PROD", default_var="aws-credentials-secret")

# credentials 안에 default가 없고 rotator-*만 있는 구조를 전제로 profile 명시
AWS_PROFILE_DEV = Variable.get("FEAST_AWS_PROFILE_DEV", default_var="rotator-dev")
AWS_PROFILE_PROD = Variable.get("FEAST_AWS_PROFILE_PROD", default_var="rotator-prod")

# materialize-incremental의 “from” 시점은 “now”가 아니라 “이전부터 now”가 일반적이지만,
# 건호님은 30분마다 갱신 목적이라 now를 유지하되, 필요하면 Variable로 바꿀 수 있게 열어둠.
INCREMENTAL_FROM_EXPR = Variable.get(
    "FEAST_INCREMENTAL_FROM_EXPR",
    default_var='$(date -u +"%Y-%m-%dT%H:%M:%S")',
)

FULL_START = Variable.get("FEAST_FULL_START", default_var="2026-01-01T00:00:00")


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
      - "incremental": feast apply + feast materialize-incremental(<from>)
      - "full":        feast apply + feast materialize(START, END)  # 복구용
    """

    # --- Volumes ---
    # ConfigMap은 atomic-writer로 symlink 구조(..data/..)를 씁니다.
    # 따라서 /feast-repo-src 에서 작업하면 import/파일 탐색이 꼬일 수 있어,
    # emptyDir(/feast-repo)로 "실파일 복사" 후 거기서 실행합니다.
    vol_repo_src = k8s.V1Volume(
        name="feast-repo-src",
        config_map=k8s.V1ConfigMapVolumeSource(name=FEAST_REPO_CONFIGMAP),
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

    vol_aws = k8s.V1Volume(
        name="aws-credentials",
        secret=k8s.V1SecretVolumeSource(secret_name=aws_cred_secret_name),
    )
    vm_aws = k8s.V1VolumeMount(
        name="aws-credentials",
        mount_path="/root/.aws",
        read_only=True,
    )

    # --- Commands ---
    # ✅ 핵심: cp -aL 로 symlink를 dereference 해서 실제 파일 내용으로 복사
    if mode == "incremental":
        cmd = rf"""
        set -eux

        # inotify/open files 메시지 완화(불가하면 그냥 진행)
        ulimit -n 4096 || true

        rm -rf /feast-repo/*
        cp -aL /feast-repo-src/. /feast-repo/

        # 디버깅 가능한 최소 증거 남기기
        ls -al /feast-repo
        test -f /feast-repo/feature_store.yaml

        cd /feast-repo
        feast apply
        feast materialize-incremental "{INCREMENTAL_FROM_EXPR}"
        """
    elif mode == "full":
        cmd = rf"""
        set -eux

        ulimit -n 4096 || true

        rm -rf /feast-repo/*
        cp -aL /feast-repo-src/. /feast-repo/

        ls -al /feast-repo
        test -f /feast-repo/feature_store.yaml

        cd /feast-repo
        feast apply

        START="{FULL_START}"
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


# -----------------------------
# DAG 1) 주기적 materialize-incremental
# -----------------------------
with DAG(
    dag_id="feast_materialize",
    start_date=datetime(2026, 2, 1, tzinfo=KST),
    schedule="*/30 * * * *",  # 30분마다
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

    # 운영 안전: dev 성공 후 prod
    feast_dev_incremental >> feast_prod_incremental


# -----------------------------
# DAG 2) 복구용 full refresh (수동)
# -----------------------------
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

