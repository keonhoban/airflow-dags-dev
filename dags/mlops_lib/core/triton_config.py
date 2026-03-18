# dags/mlops_lib/core/triton_config.py
from __future__ import annotations

import os
from typing import TYPE_CHECKING, Optional

from airflow.utils.log.logging_mixin import LoggingMixin

if TYPE_CHECKING:
    from mlops_lib.core.policy import TritonOptConfig

log = LoggingMixin().log


def atomic_write(path: str, content: str) -> None:
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        f.write(content)
    os.replace(tmp, path)


def _build_dynamic_batching_block(opt: "TritonOptConfig") -> str:
    """
    dynamic_batching 블록을 생성한다.

    Triton server-side batching 동작:
      - preferred_batch_size: 이 크기가 채워지면 즉시 dispatch
      - max_queue_delay_microseconds: 채워지지 않아도 이 시간 후 dispatch
      - max_batch_size (모델 레벨): 이 값 초과 요청은 서버가 거부

    운영 고려사항:
      - preferred_batch_size가 너무 크면 queue delay가 길어져 p50 latency 악화
      - max_queue_delay_us 5000 (5ms)는 p95 latency SLO 800ms 기준에서 허용 가능한 범위
      - GPU 없는 환경(CPU inference)에서는 batching 효과가 제한적이므로 비활성화 권장
    """
    sizes_str = ", ".join(str(s) for s in opt.preferred_batch_sizes)
    return f"""
dynamic_batching {{
  preferred_batch_size: [ {sizes_str} ]
  max_queue_delay_microseconds: {opt.max_queue_delay_us}
}}"""


def _build_instance_group_block(opt: "TritonOptConfig") -> str:
    """
    instance_group 블록을 생성한다.

    Triton 인스턴스 배치 전략:
      - KIND_GPU: GPU 디바이스에 count개의 모델 인스턴스를 올림
        → 멀티 GPU 서버에서는 gpus 필드로 특정 GPU 지정 가능
      - KIND_CPU: CPU 추론. GPU 없는 환경 또는 경량 모델에 사용
      - count > 1: 동시 요청 처리 병렬도 향상 (메모리 N배 사용)

    운영 고려사항:
      - ONNX Runtime 기반 모델은 GPU 메모리 프로파일링 후 count 조정 필요
      - count=1 (기본)은 안전한 시작점이며 Prometheus로 GPU utilization 확인 후 증가
    """
    return f"""
instance_group [
  {{
    kind: {opt.instance_group_kind}
    count: {opt.instance_group_count}
  }}
]"""


def build_config_pbtxt(
    model: str,
    in_name: str,
    out_prob: str,
    out_label: str,
    n_features: int,
    n_classes: int,
    opt: Optional["TritonOptConfig"] = None,
) -> str:
    """
    Triton config.pbtxt 문자열을 생성한다.

    ✅ 설계 원칙: version_policy를 절대 넣지 않는다.
       version_policy가 있으면 Triton이 디렉토리를 스캔해 최고 버전을 자동 선택한다.
       이 시스템은 current.json을 SSOT로 두고 배포/롤백 시점에만 버전을 변경하므로,
       Triton 자동 선택이 current.json과 충돌하는 상황을 방지하기 위해 version_policy를 제거한다.

    Args:
        model      : Triton 모델명 (config.pbtxt의 name 필드)
        in_name    : ONNX 입력 텐서명
        out_prob   : ONNX 출력 텐서명 (확률값)
        out_label  : ONNX 출력 텐서명 (클래스 레이블)
        n_features : 입력 feature 차원 수
        n_classes  : 출력 클래스 수
        opt        : TritonOptConfig. None이면 dynamic_batching/instance_group 블록 생략.
                     Variable "triton_dynamic_batching_enabled" / "triton_instance_group_enabled"
                     으로 런타임 제어. 미설정 시 기본값 false (블록 없음).
    """
    # dynamic_batching 활성화 시 max_batch_size > 0 필요.
    # 0이면 Triton이 unbatched 모드로 동작해 dynamic_batching이 무시된다.
    dyn_enabled = opt is not None and opt.dynamic_batching_enabled
    max_batch_size = opt.max_batch_size if dyn_enabled else 0

    # 선택적 블록 조립
    opt_blocks = ""
    if dyn_enabled:
        opt_blocks += _build_dynamic_batching_block(opt)
        log.info(
            "[config] dynamic_batching enabled: preferred=%s delay_us=%d max_batch=%d",
            opt.preferred_batch_sizes, opt.max_queue_delay_us, max_batch_size,
        )
    if opt is not None and opt.instance_group_enabled:
        opt_blocks += _build_instance_group_block(opt)
        log.info(
            "[config] instance_group enabled: kind=%s count=%d",
            opt.instance_group_kind, opt.instance_group_count,
        )

    return f'''name: "{model}"
platform: "onnxruntime_onnx"
max_batch_size: {max_batch_size}

input [
  {{
    name: "{in_name}"
    data_type: TYPE_FP32
    dims: [ -1, {n_features} ]
  }}
]

output [
  {{
    name: "{out_prob}"
    data_type: TYPE_FP32
    dims: [ -1, {n_classes} ]
  }},
  {{
    name: "{out_label}"
    data_type: TYPE_INT64
    dims: [ -1 ]
  }}
]{opt_blocks}
'''


def write_config_atomic(model_dir: str, *, cfg_text: str) -> None:
    """
    ✅ config.pbtxt는 '항상 같은 형태(입출력/플랫폼)'로만 유지한다.
    - version_policy 없음
    - 현재 활성 버전 선택은 디렉토리 구조 / current.json / load payload 등으로 관리
    """
    config_path = os.path.join(model_dir, "config.pbtxt")
    atomic_write(config_path, cfg_text.rstrip() + "\n")
    log.warning("[config] config.pbtxt written (NO version_policy) path=%s", config_path)
