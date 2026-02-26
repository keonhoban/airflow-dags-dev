# dags/mlops_lib/core/triton_config.py
from __future__ import annotations

import os
from typing import List

from airflow.utils.log.logging_mixin import LoggingMixin

log = LoggingMixin().log


def atomic_write(path: str, content: str) -> None:
    tmp = path + ".tmp"
    with open(tmp, "w") as f:
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


def parse_output_names(raw: str) -> List[str]:
    return [x.strip() for x in (raw or "").split(",") if x.strip()]


def strip_version_policy_blocks(text: str) -> str:
    """
    version_policy { ... } 블록 제거 (brace counting 방식).
    """
    key = "version_policy"
    if key not in text:
        return text

    out = text
    while True:
        i = out.find(key)
        if i == -1:
            break

        start = out.rfind("\n", 0, i)
        start = 0 if start == -1 else start

        brace_open = out.find("{", i)
        if brace_open == -1:
            line_end = out.find("\n", i)
            out = out[:start] if line_end == -1 else (out[:start] + out[line_end + 1 :])
            break

        depth = 0
        j = brace_open
        end = None
        while j < len(out):
            ch = out[j]
            if ch == "{":
                depth += 1
            elif ch == "}":
                depth -= 1
                if depth == 0:
                    end = j + 1
                    break
            j += 1

        if end is None:
            line_end = out.find("\n", i)
            out = out[:start] if line_end == -1 else (out[:start] + out[line_end + 1 :])
            break

        tail = out[end:]
        if tail.startswith("\n"):
            tail = tail[1:]
        out = out[:start].rstrip() + "\n" + tail.lstrip()

    return out


def strip_trailing_lonely_braces(text: str) -> str:
    lines = text.splitlines()
    out = []
    removed = 0
    for ln in lines:
        if ln.strip() == "}":
            removed += 1
            continue
        out.append(ln)
    if removed:
        log.warning("[config] stripped lonely brace lines=%s", removed)
    return "\n".join(out) + ("\n" if text.endswith("\n") else "")


def set_version_policy_specific(config_text: str, version: int) -> str:
    vp = (
        "\nversion_policy {\n"
        "  specific {\n"
        f"    versions: [ {int(version)} ]\n"
        "  }\n"
        "}\n"
    )
    base = strip_version_policy_blocks(config_text)
    base = strip_trailing_lonely_braces(base).rstrip()
    return base + vp + "\n"


def write_config_with_policy_atomic(model_dir: str, *, base_cfg: str, version: int) -> None:
    config_path = os.path.join(model_dir, "config.pbtxt")
    final_cfg = set_version_policy_specific(base_cfg, int(version))
    atomic_write(config_path, final_cfg)
    log.warning("[config] config.pbtxt written (policy specific=%s) path=%s", version, config_path)


def write_or_update_config_policy(model_dir: str, *, version: int) -> None:
    config_path = os.path.join(model_dir, "config.pbtxt")
    if not os.path.exists(config_path):
        raise RuntimeError(f"[config] missing config.pbtxt at {config_path}")

    with open(config_path, "r") as f:
        cur = f.read()

    updated = set_version_policy_specific(cur, int(version))
    atomic_write(config_path, updated)
    log.warning("[config] version_policy specific set to %s (%s)", version, config_path)
