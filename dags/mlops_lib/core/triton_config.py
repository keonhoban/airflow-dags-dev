# dags/mlops_lib/core/triton_config.py
from __future__ import annotations

import os
from airflow.utils.log.logging_mixin import LoggingMixin

log = LoggingMixin().log


def atomic_write(path: str, content: str) -> None:
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        f.write(content)
    os.replace(tmp, path)


def build_config_pbtxt(
    model: str,
    in_name: str,
    out_prob: str,
    out_label: str,
    n_features: int,
    n_classes: int,
) -> str:
    # ✅ 핵심: version_policy를 절대 넣지 않는다.
    # (특정 버전 강제 -> 버전 디렉토리 깨지면 Triton load가 즉시 죽는 구조)
    return f'''name: "{model}"
platform: "onnxruntime_onnx"
max_batch_size: 0

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
]
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
