# dags/feast_materialize_dev.py
from __future__ import annotations

from datetime import datetime, timedelta
import pendulum

from airflow import DAG
from airflow.providers.cncf.kubernetes.operators.pod import KubernetesPodOperator
from kubernetes.client import models as k8s

KST = pendulum.timezone("Asia/Seoul")

# ---- 고정 값 (면접/유지보수 관점: 옵션 최소화) ----
NAMESPACE = "feature-store-dev"

FEAST_IMAGE = "hoizz/feast-server:0.40.1-s3fs"
FEAST_REPO_CONFIGMAP = "feast-repo"           # ✅ 반드시 존재해야 함
AWS_CRED_SECRET = "aws-credentials-secret"    # ✅ 기존 secretName과 일치

AWS_REGION = "ap-northeast-2"
AWS_PROFILE = "rotator-dev"

FULL_REFRESH_START = "2026-01-01T00:00:00"

DEFAULT_ARGS = {
    "owner": "airflow",
    "retries": 1,
    "retry_delay": timedelta(minutes=3),
}

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

def _script(mode: str) -> str:
    base = r"""
set -eu

rm -rf /repo/* || true

# ConfigMap mount의 숨김 디렉토리(..data, ..2026_...)는 복사하지 않기:
# 1) 와일드카드(*)로 dotfile 제외
# 2) -L 로 symlink 실체화
cp -aL /repo-src/* /repo/ || true

# 혹시 남아있으면 제거(안전망)
find /repo -maxdepth 1 -name '..*' -exec rm -rf {} + || true

cd /repo
test -f feature_store.yaml
test -f repo.py

feast apply
""".strip()

    if mode == "incremental":
        return base + "\n" + r"""feast materialize-incremental "$(date -u +'%Y-%m-%dT%H:%M:%S')" """.strip()

    if mode == "full":
        return base + "\n" + rf"""
START="{FULL_REFRESH_START}"
END="$(date -u +'%Y-%m-%dT%H:%M:%S')"
echo "materialize $START -> $END"
feast materialize "$START" "$END"
""".strip()

    raise ValueError(f"invalid mode: {mode}")

def _kpo(task_id: str, mode: str) -> KubernetesPodOperator:
    repo_vols, repo_mounts = _repo_volumes()
    aws_vol, aws_mount = _aws_cred_volume()

    return KubernetesPodOperator(
        task_id=task_id,
        name=task_id,
        namespace=NAMESPACE,
        image=FEAST_IMAGE,
        cmds=["/bin/sh", "-c"],
        arguments=[_script(mode)],
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

# 1) 30분마다 incremental
with DAG(
    dag_id="feast_materialize_dev",
    start_date=datetime(2026, 2, 1, tzinfo=KST),
    schedule="*/30 * * * *",
    catchup=False,
    default_args=DEFAULT_ARGS,
    tags=["feature-store", "feast", "dev"],
) as dag:
    _kpo("feast_materialize_incremental", mode="incremental")

# 2) 수동 full refresh (복구/초기화)
with DAG(
    dag_id="feast_full_refresh_dev_manual",
    start_date=datetime(2026, 2, 1, tzinfo=KST),
    schedule=None,
    catchup=False,
    default_args=DEFAULT_ARGS,
    tags=["feature-store", "feast", "dev", "recovery"],
) as dag_full:
    _kpo("feast_materialize_full", mode="full")

