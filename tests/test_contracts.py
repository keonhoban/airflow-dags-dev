# tests/test_contracts.py
"""
Cross-repo API contract tests.

DAG 코드가 가정하는 외부 서비스 엔드포인트가 contracts/api_contracts.json과
일치하는지 검증. 인프라 변경 시 호환성 깨짐을 CI에서 조기 감지한다.
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest

CONTRACTS_PATH = Path(__file__).parent.parent / "contracts" / "api_contracts.json"
DAGS_ROOT = Path(__file__).parent.parent / "dags"


@pytest.fixture(scope="module")
def contracts():
    with open(CONTRACTS_PATH, encoding="utf-8") as f:
        return json.load(f)


def _collect_python_files(root: Path):
    return list(root.rglob("*.py"))


class TestTritonContracts:

    def test_triton_load_endpoint_pattern(self, contracts):
        pattern = "/v2/repository/models/"
        found = any(pattern in f.read_text(encoding="utf-8", errors="ignore")
                     for f in _collect_python_files(DAGS_ROOT))
        assert found, f"Triton load/unload endpoint pattern '{pattern}' not found in DAG code"

    def test_triton_ready_endpoint_pattern(self, contracts):
        pattern = "/v2/models/"
        found = any(pattern in f.read_text(encoding="utf-8", errors="ignore")
                     for f in _collect_python_files(DAGS_ROOT))
        assert found, "Triton model ready/infer endpoint not found in DAG code"

    def test_triton_ports_match(self, contracts):
        assert contracts["triton"]["ports"]["http"] == 8000

    def test_triton_explicit_mode(self, contracts):
        assert contracts["triton"]["model_control_mode"] == "explicit"


class TestFastAPIContracts:

    def test_reload_endpoint_pattern(self, contracts):
        pattern = "/variant/"
        found = any(pattern in f.read_text(encoding="utf-8", errors="ignore")
                     and "reload" in f.read_text(encoding="utf-8", errors="ignore")
                     for f in _collect_python_files(DAGS_ROOT))
        assert found, "FastAPI reload endpoint pattern '/variant/.../reload' not found"

    def test_models_endpoint_pattern(self, contracts):
        expected = contracts["fastapi"]["endpoints"]["models"]
        found = any(expected in f.read_text(encoding="utf-8", errors="ignore")
                     for f in _collect_python_files(DAGS_ROOT))
        assert found, f"FastAPI models endpoint '{expected}' not found in DAG code"


class TestContractFileIntegrity:

    def test_contract_file_exists(self):
        assert CONTRACTS_PATH.exists()

    def test_contract_is_valid_json(self, contracts):
        assert isinstance(contracts, dict)
        assert "triton" in contracts
        assert "fastapi" in contracts
        assert "prometheus" in contracts
