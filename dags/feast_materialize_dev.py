# dags/feast_materialize_dev.py
from __future__ import annotations

from datetime import datetime, timedelta
import pendulum

from airflow import DAG
from airflow.providers.cncf.kubernetes.operators.pod import KubernetesPodOperator
from kubernetes.client import models as k8s

KST = pendulum.timezone("Asia/Seoul")

# =========================
# ✅ 운영/면접/유지보수 기준: 바뀔만한 값만 상수로 고정
# =========================
NAMESPACE = "feature-store-dev"

FEAST_IMAGE = "hoizz/feast-server:0.40.1-s3fs"

# ConfigMap: feature_store.yaml + repo.py 포함
FEAST_REPO_CONFIGMAP = "feast-repo"

# AWS credentials secret: /root/.aws/credentials 로 마운트
AWS_CRED_SECRET = "aws-credentials-secret"
AWS_PROFILE = "rotator-dev"
AWS_REGION = "ap-northeast-2"

# Full refresh 시작점 (복구/초기화용)
FULL_REFRESH_START_ISO = "2026-01-01T00:00:00"

# 30분 주기 incremental
SCHEDULE = "*/30 * * * *"

DEFAULT_ARGS = {
    "owner": "airflow",
    "retries": 1,
    "retry_delay": timedelta(minutes=3),
}


# =========================
# K8s volumes
# - ConfigMap은 ..data/..timestamp atomic writer 구조가 있어
#   emptyDir에 복사 후 실행
# =========================
def _repo_volumes():
    vol_src = k8s.V1Volume(
        name="feast-repo-src",
        config_map=k8s.V1ConfigMapVolumeSource(name=FEAST_REPO_CONFIGMAP),
    )
    vm_src = k8s.V1VolumeMount(
        name="feast-repo-src",
        mount_path="/repo-src",
        read_only=True,
    )

    vol_work = k8s.V1Volume(
        name="feast-repo-work",
        empty_dir=k8s.V1EmptyDirVolumeSource(),
    )
    vm_work = k8s.V1VolumeMount(
        name="feast-repo-work",
        mount_path="/repo",
        read_only=False,
    )

    return [vol_src, vol_work], [vm_src, vm_work]


def _aws_cred_volume():
    vol = k8s.V1Volume(
        name="aws-credentials",
        secret=k8s.V1SecretVolumeSource(secret_name=AWS_CRED_SECRET),
    )
    vm = k8s.V1VolumeMount(
        name="aws-credentials",
        mount_path="/root/.aws",
        read_only=True,
    )
    return vol, vm


# =========================
# sh-safe scripts
# - dotfile(..data 등) 복사 방지
# - symlink 실체화(-L)
# =========================
def _script_common_preamble() -> str:
    return r"""
set -eu
set -x

# workdir clean
rm -rf /repo/* || true

# ✅ 핵심: ConfigMap mount의 숨김 디렉토리(..data, ..2026_...)는 복사하지 않기
# - * 로 dotfile 제외
# - -L 로 symlink 실체화
cp -aL /repo-src/* /repo/

# 혹시 남아있으면 제거(안전망)
find /repo -maxdepth 1 -name '..*' -exec rm -rf {} + || true

cd /repo
test -f feature_store.yaml
test -f repo.py
""".strip()


def _script_incremental() -> str:
    return (
        _script_common_preamble()
        + r"""

feast apply
feast materialize-incremental "$(date -u +'%Y-%m-%dT%H:%M:%S')"
""".strip()
    )


def _script_full(start_iso: str) -> str:
    return (
        _script_common_preamble()
        + rf"""

feast apply
START="{start_iso}"
END="$(date -u +'%Y-%m-%dT%H:%M:%S')"
echo "materialize $START -> $END"
feast materialize "$START" "$END"
""".strip()
    )


def _kpo(task_id: str, script: str) -> KubernetesPodOperator:
    repo_vols, repo_mounts = _repo_volumes()
    aws_vol, aws_mount = _aws_cred_volume()

    return KubernetesPodOperator(
        task_id=task_id,
        name=task_id,
        namespace=NAMESPACE,
        image=FEAST_IMAGE,
        cmds=["/bin/sh", "-c"],
        arguments=[script],
        env_vars={
            "AWS_SHARED_CREDENTIALS_FILE": "/root/.aws/credentials",
            "AWS_PROFILE": AWS_PROFILE,
            "AWS_DEFAULT_REGION": AWS_REGION,
            "AWS_REGION": AWS_REGION,
        },
        volumes=[*repo_vols, aws_vol],
        volume_mounts=[*repo_mounts, aws_mount],
        get_logs=True,
        is_delete_operator_pod=True,
        startup_timeout_seconds=300,
    )


# =========================
# DAG 1) Incremental (주기)
# =========================
with DAG(
    dag_id="feast_materialize_dev",
    start_date=datetime(2026, 2, 1, tzinfo=KST),
    schedule=SCHEDULE,
    catchup=False,
    default_args=DEFAULT_ARGS,
    tags=["feast", "feature-store", "dev"],
) as dag:
    _kpo("feast_materialize_incremental", _script_incremental())


# =========================
# DAG 2) Full refresh (수동)
# =========================
with DAG(
    dag_id="feast_full_refresh_dev_manual",
    start_date=datetime(2026, 2, 1, tzinfo=KST),
    schedule=None,
    catchup=False,
    default_args=DEFAULT_ARGS,
    tags=["feast", "feature-store", "dev", "recovery"],
) as dag_full:
    _kpo("feast_materialize_full", _script_full(start_iso=FULL_REFRESH_START_ISO))

