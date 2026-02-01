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

# ──────────────────────────────────────────────────────────────────────────────
# Airflow Variables (운영에서 교체 가능한 값만 노출)
# ──────────────────────────────────────────────────────────────────────────────
FEAST_IMAGE = Variable.get("FEAST_IMAGE", default_var="hoizz/feast-server:0.40.1-s3fs")
AWS_REGION = Variable.get("AWS_REGION", default_var="ap-northeast-2")

# Feast repo는 ConfigMap으로 배포됨 (dev/prod 공통 이름 or 환경별로 분리 가능)
FEAST_REPO_CONFIGMAP = Variable.get("FEAST_REPO_CONFIGMAP", default_var="feast-repo")

# AWS credentials secret 이름 (dev/prod 별도 운영 가능)
AWS_CRED_SECRET_DEV = Variable.get("FEAST_AWS_CRED_SECRET_DEV", default_var="aws-credentials-secret")
AWS_CRED_SECRET_PROD = Variable.get("FEAST_AWS_CRED_SECRET_PROD", default_var="aws-credentials-secret")

# ✅ credentials 파일에 [default] 가 없고 [rotator-dev]/[rotator-prod]만 있는 구조라면 profile 명시 필수
AWS_PROFILE_DEV = Variable.get("FEAST_AWS_PROFILE_DEV", default_var="rotator-dev")
AWS_PROFILE_PROD = Variable.get("FEAST_AWS_PROFILE_PROD", default_var="rotator-prod")

# full refresh 기본 시작점 (복구/재적재)
FEAST_FULL_START = Variable.get("FEAST_FULL_START", default_var="2026-01-01T00:00:00")


Mode = Literal["incremental", "full"]


def _build_repo_volumes(
    *,
    repo_configmap_name: str,
) -> tuple[list[k8s.V1Volume], list[k8s.V1VolumeMount]]:
    """
    ConfigMap repo는 Kubernetes atomic writer 구조로 ..data/..timestamp 디렉토리가 생깁니다.
    이 디렉토리를 그대로 작업 디렉토리로 쓰면 (특히 "." 복사) import 오염이 발생할 수 있어
    emptyDir(워크)로 복사 후 실행합니다.
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


def _build_aws_cred_volume(
    *,
    aws_cred_secret_name: str,
) -> tuple[k8s.V1Volume, k8s.V1VolumeMount]:
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
    ✅ 실무형 방어 포인트
    - ConfigMap atomic writer 숨김 디렉토리(..data/..timestamp) 오염 방지:
      cp "/feast-repo-src/." 금지, cp "/feast-repo-src/*" 사용
    - workdir 청소는 숨김 포함으로 강제
    - 핵심 파일 존재 검사로 실패 지점을 고정
    """
    common = r"""
set -eux

# open files/inotify 메시지 완화(불가하면 그냥 진행)
ulimit -n 4096 || true

# 1) workdir 완전 정리 (숨김 포함)
find /feast-repo -mindepth 1 -maxdepth 1 -exec rm -rf {} +

# 2) ConfigMap repo를 workdir로 복사
#    - 주의: "/." 복사는 ..data/..timestamp(숨김)까지 포함될 수 있음 → import 오염
cp -aL /feast-repo-src/* /feast-repo/

# 3) 방어: 숨김 atomic writer 디렉토리가 섞였으면 즉시 실패
test ! -e /feast-repo/..data

# 4) 최소 증거 및 핵심 파일 검증
ls -al /feast-repo
test -f /feast-repo/feature_store.yaml
test -f /feast-repo/repo.py

cd /feast-repo
feast apply
""".strip()

    if mode == "incremental":
        return common + r"""

# incremental: 마지막 시점부터 now까지 누적 반영
feast materialize-incremental "$(date -u +"%Y-%m-%dT%H:%M:%S")"
""".strip()

    if mode == "full":
        # 복구/재적재: START -> now
        return common + rf"""

START="{full_start}"
END="$(date -u +"%Y-%m-%dT%H:%M:%S")"
echo "materialize $START -> $END"
feast materialize "$START" "$END"
""".strip()

    raise ValueError(f"unknown mode: {mode}")


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


# ──────────────────────────────────────────────────────────────────────────────
# DAG 1) 30분마다 incremental (dev -> prod)
# ──────────────────────────────────────────────────────────────────────────────
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

    # 운영적으로 안전한 순서: dev 성공 후 prod 반영
    feast_dev_incremental >> feast_prod_incremental


# ──────────────────────────────────────────────────────────────────────────────
# DAG 2) 수동 실행 full refresh (복구/재적재, dev -> prod)
# ──────────────────────────────────────────────────────────────────────────────
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

