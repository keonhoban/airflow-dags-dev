# dags/feast_materialize_dev.py
from __future__ import annotations

from datetime import datetime, timedelta
import pendulum

from airflow import DAG
from airflow.providers.cncf.kubernetes.operators.pod import KubernetesPodOperator
from kubernetes.client import models as k8s

KST = pendulum.timezone("Asia/Seoul")

# =========================
# 운영에서 바뀔만한 값만 "상수"로 고정
# (변수/옵션 과다 제거: 면접/유지보수에 유리)
# =========================
NAMESPACE = "feature-store-dev"
FEAST_IMAGE = "hoizz/feast-server:0.40.1-s3fs"

FEAST_REPO_CONFIGMAP = "feast-repo"           # feature_store.yaml + repo.py 들어있는 CM
AWS_CRED_SECRET = "aws-credentials-secret"    # /root/.aws/credentials
AWS_REGION = "ap-northeast-2"
AWS_PROFILE = "rotator-dev"

DEFAULT_ARGS = {
    "owner": "airflow",
    "retries": 1,
    "retry_delay": timedelta(minutes=3),
}


def _feast_repo_volumes():
    """
    ConfigMap은 ..data(atomic writer) 링크 구조가 생길 수 있어
    실행 디렉토리(emptyDir)에 복사 후 실행한다.
    """
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


def _script_incremental() -> str:
    """
    /bin/sh 기준으로만 작성 (bash 의존 제거)
    - repo 복사
    - feast apply
    - feast materialize-incremental now(UTC)
    """
    return r"""
set -eu

# repo workdir 준비
rm -rf /repo/* || true
cp -a /repo-src/. /repo/

cd /repo
test -f feature_store.yaml
test -f repo.py

feast apply
feast materialize-incremental "$(date -u +'%Y-%m-%dT%H:%M:%S')"
""".strip()


def _script_full(start_iso: str) -> str:
    return rf"""
set -eu

rm -rf /repo/* || true
cp -a /repo-src/. /repo/

cd /repo
test -f feature_store.yaml
test -f repo.py

feast apply
START="{start_iso}"
END="$(date -u +'%Y-%m-%dT%H:%M:%S')"
echo "materialize $START -> $END"
feast materialize "$START" "$END"
""".strip()


def _kpo(task_id: str, script: str) -> KubernetesPodOperator:
    repo_vols, repo_mounts = _feast_repo_volumes()
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
# DAG 1) Incremental (30분 주기)
# =========================
with DAG(
    dag_id="feast_materialize_dev",
    start_date=datetime(2026, 2, 1, tzinfo=KST),
    schedule="*/30 * * * *",
    catchup=False,
    default_args=DEFAULT_ARGS,
    tags=["feast", "feature-store", "dev"],
) as dag:
    _kpo("feast_materialize_incremental", _script_incremental())


# =========================
# DAG 2) Full refresh (수동 실행)
# - 운영 복구/초기화용 (정상 운영에서는 거의 안 씀)
# =========================
with DAG(
    dag_id="feast_full_refresh_dev_manual",
    start_date=datetime(2026, 2, 1, tzinfo=KST),
    schedule=None,
    catchup=False,
    default_args=DEFAULT_ARGS,
    tags=["feast", "feature-store", "dev", "recovery"],
) as dag_full:
    _kpo("feast_materialize_full", _script_full(start_iso="2026-01-01T00:00:00"))

