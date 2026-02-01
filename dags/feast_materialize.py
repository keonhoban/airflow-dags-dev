# dags/feast_materialize_dev.py
from __future__ import annotations

from datetime import datetime, timedelta
import pendulum

from airflow import DAG
from airflow.models import Variable
from airflow.providers.cncf.kubernetes.operators.pod import KubernetesPodOperator
from kubernetes.client import models as k8s

KST = pendulum.timezone("Asia/Seoul")

# =========================
# 최소 설정 (필요하면 Variable로 override)
# =========================
FEAST_IMAGE = Variable.get("FEAST_IMAGE", default_var="hoizz/feast-server:0.40.1-s3fs")
FEAST_REPO_CONFIGMAP = Variable.get("FEAST_REPO_CONFIGMAP", default_var="feast-repo")

AWS_REGION = Variable.get("AWS_REGION", default_var="ap-northeast-2")
AWS_PROFILE_DEV = Variable.get("FEAST_AWS_PROFILE_DEV", default_var="rotator-dev")
AWS_CRED_SECRET_DEV = Variable.get("FEAST_AWS_CRED_SECRET_DEV", default_var="aws-credentials-secret")

# full refresh 시작점(복구/초기화용)
FEAST_FULL_START = Variable.get("FEAST_FULL_START", default_var="2026-01-01T00:00:00")

# /bin/sh에서도 안전하게 동작하도록 pipefail 금지
ULIMIT_NOFILE = int(Variable.get("FEAST_ULIMIT_NOFILE", default_var="8192"))

DEFAULT_ARGS = {
    "owner": "airflow",
    "retries": 1,
    "retry_delay": timedelta(minutes=3),
}

# =========================
# K8s Volumes
# =========================
def _repo_volumes(configmap_name: str):
    # ConfigMap은 atomic writer로 ..data 링크가 생길 수 있으니,
    # emptyDir(work)로 복사해서 실행한다.
    vol_src = k8s.V1Volume(
        name="feast-repo-src",
        config_map=k8s.V1ConfigMapVolumeSource(name=configmap_name),
    )
    vm_src = k8s.V1VolumeMount(
        name="feast-repo-src",
        mount_path="/feast-repo-src",
        read_only=True,
    )

    vol_work = k8s.V1Volume(
        name="feast-repo-work",
        empty_dir=k8s.V1EmptyDirVolumeSource(),
    )
    vm_work = k8s.V1VolumeMount(
        name="feast-repo-work",
        mount_path="/feast-repo",
        read_only=False,
    )

    return [vol_src, vol_work], [vm_src, vm_work]


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


# =========================
# Script builder (sh-safe)
# =========================
def _build_script(*, mode: str, full_start: str) -> str:
    """
    목표:
    - /bin/sh 환경에서 깨지지 않게 (pipefail 금지)
    - bash 있으면 bash로 자동 재실행 (선택)
    - ConfigMap atomic writer(..data) 오염 방지
    """
    lines: list[str] = [
        # bash 있으면 bash -lc로 승급 (있을 때만)
        'if command -v bash >/dev/null 2>&1; then exec bash -lc "$0"; fi',
        "",
        "set -eu",
        "set -x",
        f"ulimit -n {ULIMIT_NOFILE} || true",
        "",
        # workdir clean
        "find /feast-repo -mindepth 1 -maxdepth 1 -exec rm -rf {} +",
        # copy configmap -> workdir (symlink 제거 위해 -aL)
        "cp -aL /feast-repo-src/* /feast-repo/",
        "",
        # atomic writer 방어
        "test ! -e /feast-repo/..data",
        "test -f /feast-repo/feature_store.yaml",
        "test -f /feast-repo/repo.py",
        "ls -al /feast-repo",
        "cd /feast-repo",
        "",
        "feast apply",
        "",
    ]

    if mode == "incremental":
        lines += [
            'feast materialize-incremental "$(date -u +\"%Y-%m-%dT%H:%M:%S\")"',
        ]
    elif mode == "full":
        lines += [
            f'START="{full_start}"',
            'END="$(date -u +\"%Y-%m-%dT%H:%M:%S\")"',
            'echo "materialize $START -> $END"',
            'feast materialize "$START" "$END"',
        ]
    else:
        raise ValueError(f"invalid mode: {mode}")

    return "\n".join(lines) + "\n"


def _feast_kpo(*, dag: DAG, task_id: str, mode: str) -> KubernetesPodOperator:
    repo_vols, repo_mounts = _repo_volumes(FEAST_REPO_CONFIGMAP)
    vol_aws, vm_aws = _aws_cred_volume(AWS_CRED_SECRET_DEV)

    script = _build_script(mode=mode, full_start=FEAST_FULL_START)

    return KubernetesPodOperator(
        dag=dag,
        task_id=task_id,
        name=task_id,
        namespace="feature-store-dev",          # ✅ dev만
        image=FEAST_IMAGE,
        cmds=["/bin/sh", "-c"],
        arguments=[script],
        env_vars={
            "AWS_SHARED_CREDENTIALS_FILE": "/root/.aws/credentials",
            "AWS_PROFILE": AWS_PROFILE_DEV,
            "AWS_DEFAULT_REGION": AWS_REGION,
            "AWS_REGION": AWS_REGION,
        },
        volumes=[*repo_vols, vol_aws],
        volume_mounts=[*repo_mounts, vm_aws],
        get_logs=True,
        is_delete_operator_pod=True,
        startup_timeout_seconds=300,
    )


# =========================
# DAG 1) 주기 incremental (dev only)
# =========================
with DAG(
    dag_id="feast_materialize_dev",
    start_date=datetime(2026, 2, 1, tzinfo=KST),
    schedule="*/30 * * * *",
    catchup=False,
    default_args=DEFAULT_ARGS,
    tags=["feature-store", "feast", "dev", "materialize"],
) as dag:
    feast_incremental = _feast_kpo(dag=dag, task_id="feast_dev_incremental", mode="incremental")

# =========================
# DAG 2) 수동 full refresh (dev only)
# =========================
with DAG(
    dag_id="feast_full_refresh_dev_manual",
    start_date=datetime(2026, 2, 1, tzinfo=KST),
    schedule=None,
    catchup=False,
    default_args=DEFAULT_ARGS,
    tags=["feature-store", "feast", "dev", "recovery"],
) as dag_full:
    feast_full = _feast_kpo(dag=dag_full, task_id="feast_dev_full_refresh", mode="full")

