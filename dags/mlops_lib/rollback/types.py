# dags/mlops_lib/rollback/types.py
from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Any, Optional

from airflow.utils.log.logging_mixin import LoggingMixin

from ml_code.config import cfg

log = LoggingMixin().log


def env_required(key: str) -> str:
    v = os.environ.get(key)
    if not v:
        raise RuntimeError(f"missing env: {key}")
    return v


@dataclass(frozen=True)
class Ctx:
    model: str
    model_dir: str
    deploy_version: int
    alias: str
    reason: str

    triton_http_url: str
    fastapi_base_url: str
    fastapi_token: str

    fastapi_wait_timeout_sec: int = 30
    fastapi_wait_interval_sec: int = 2
    fastapi_converge_min_pods: int = 2

    triton_ready_timeout_sec: int = 60
    triton_ready_interval_sec: int = 2

    def to_dict(self) -> dict[str, Any]:
        return {
            "model": self.model,
            "model_dir": self.model_dir,
            "deploy_version": int(self.deploy_version),
            "alias": self.alias,
            "reason": self.reason,
            "triton_http_url": self.triton_http_url,
            "fastapi_base_url": self.fastapi_base_url,
            "fastapi_token": self.fastapi_token,
            "fastapi_wait_timeout_sec": int(self.fastapi_wait_timeout_sec),
            "fastapi_wait_interval_sec": int(self.fastapi_wait_interval_sec),
            "fastapi_converge_min_pods": int(self.fastapi_converge_min_pods),
            "triton_ready_timeout_sec": int(self.triton_ready_timeout_sec),
            "triton_ready_interval_sec": int(self.triton_ready_interval_sec),
        }

    @staticmethod
    def from_dict(d: dict[str, Any]) -> "Ctx":
        return Ctx(
            model=str(d["model"]),
            model_dir=str(d["model_dir"]),
            deploy_version=int(d["deploy_version"]),
            alias=str(d["alias"]),
            reason=str(d.get("reason") or "manual rollback"),
            triton_http_url=str(d["triton_http_url"]).rstrip("/"),
            fastapi_base_url=str(d["fastapi_base_url"]).rstrip("/"),
            fastapi_token=str(d["fastapi_token"]),
            fastapi_wait_timeout_sec=int(d.get("fastapi_wait_timeout_sec") or 30),
            fastapi_wait_interval_sec=int(d.get("fastapi_wait_interval_sec") or 2),
            fastapi_converge_min_pods=int(d.get("fastapi_converge_min_pods") or 2),
            triton_ready_timeout_sec=int(d.get("triton_ready_timeout_sec") or 60),
            triton_ready_interval_sec=int(d.get("triton_ready_interval_sec") or 2),
        )


def build_ctx_from_airflow(dag_run=None) -> Ctx:
    conf = dict(getattr(dag_run, "conf", None) or {})

    model = str(cfg("triton_model_name", required=True))
    repo = str(cfg("triton_repo_base", "/models"))
    model_dir = os.path.join(repo, model)
    os.makedirs(model_dir, exist_ok=True)

    deploy_version = int(conf.get("deploy_version") or cfg("rollback_deploy_version", required=True))
    alias = str(conf.get("alias") or "A")
    reason = str(conf.get("reason") or "manual rollback")

    triton_http_url = os.environ.get("TRITON_HTTP_URL") or str(
        cfg("triton_http_url", "http://triton.triton-dev.svc.cluster.local:8000")
    )
    fastapi_base_url = env_required("FASTAPI_BASE_URL").rstrip("/")
    fastapi_token = env_required("RELOAD_SECRET_TOKEN")

    ctx = Ctx(
        model=model,
        model_dir=model_dir,
        deploy_version=int(deploy_version),
        alias=alias,
        reason=reason,
        triton_http_url=str(triton_http_url).rstrip("/"),
        fastapi_base_url=fastapi_base_url,
        fastapi_token=fastapi_token,
        fastapi_wait_timeout_sec=int(conf.get("fastapi_wait_timeout_sec") or 30),
        fastapi_wait_interval_sec=int(conf.get("fastapi_wait_interval_sec") or 2),
        fastapi_converge_min_pods=int(conf.get("fastapi_converge_min_pods") or 2),
        triton_ready_timeout_sec=int(conf.get("triton_ready_timeout_sec") or 60),
        triton_ready_interval_sec=int(conf.get("triton_ready_interval_sec") or 2),
    )

    log.info("[rollback_manual] ctx=%s", ctx)
    return ctx
