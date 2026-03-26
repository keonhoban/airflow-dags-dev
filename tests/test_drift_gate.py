# tests/test_drift_gate.py
"""
drift_gate.py 비즈니스 로직 단위 테스트.

_ks_stat: KS 통계량 계산 정확성
_pick_numeric_columns: 수치형 컬럼 선택 로직
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from mlops_lib.quality.drift_gate import _ks_stat, _ks_pvalue, _pick_numeric_columns


class TestKsStat:
    """_ks_stat: Two-sample KS statistic (D) 계산 검증."""

    def test_identical_distributions_returns_zero(self):
        x = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        assert _ks_stat(x, x) == 0.0

    def test_completely_separated_distributions_returns_one(self):
        x = np.array([1.0, 2.0, 3.0])
        y = np.array([10.0, 11.0, 12.0])
        assert _ks_stat(x, y) == pytest.approx(1.0)

    def test_known_ks_value(self):
        """scipy.stats.ks_2samp과 일치하는 값 검증."""
        np.random.seed(42)
        x = np.random.normal(0, 1, 200)
        y = np.random.normal(0, 1, 200)
        d = _ks_stat(x, y)
        # 같은 분포에서 뽑았으므로 D는 작아야 함
        assert 0.0 <= d < 0.2

    def test_shifted_distribution_detects_drift(self):
        np.random.seed(42)
        x = np.random.normal(0, 1, 500)
        y = np.random.normal(2, 1, 500)  # 평균 2만큼 shift
        d = _ks_stat(x, y)
        assert d > 0.5  # 명확한 shift는 큰 D값

    def test_empty_array_returns_zero(self):
        x = np.array([1.0, 2.0])
        empty = np.array([])
        assert _ks_stat(x, empty) == 0.0
        assert _ks_stat(empty, x) == 0.0

    def test_single_element_arrays(self):
        x = np.array([1.0])
        y = np.array([2.0])
        d = _ks_stat(x, y)
        assert 0.0 <= d <= 1.0

    def test_symmetry(self):
        np.random.seed(42)
        x = np.random.normal(0, 1, 100)
        y = np.random.normal(1, 1, 100)
        assert _ks_stat(x, y) == pytest.approx(_ks_stat(y, x))


class TestKsPvalue:
    """_ks_pvalue: asymptotic p-value approximation."""

    def test_identical_distributions_high_pvalue(self):
        np.random.seed(42)
        x = np.random.normal(0, 1, 200)
        y = np.random.normal(0, 1, 200)
        d = _ks_stat(x, y)
        p = _ks_pvalue(d, len(x), len(y))
        assert p > 0.05

    def test_shifted_distributions_low_pvalue(self):
        np.random.seed(42)
        x = np.random.normal(0, 1, 500)
        y = np.random.normal(2, 1, 500)
        d = _ks_stat(x, y)
        p = _ks_pvalue(d, len(x), len(y))
        assert p < 0.001

    def test_empty_arrays_return_one(self):
        assert _ks_pvalue(0.5, 0, 100) == 1.0
        assert _ks_pvalue(0.5, 100, 0) == 1.0

    def test_zero_d_returns_one(self):
        assert _ks_pvalue(0.0, 100, 100) == 1.0

    def test_pvalue_bounded_zero_one(self):
        p = _ks_pvalue(0.99, 1000, 1000)
        assert 0.0 <= p <= 1.0


class TestPickNumericColumns:
    """_pick_numeric_columns: 수치형 컬럼만 선택, max_cols 제한."""

    def test_selects_numeric_only(self):
        df = pd.DataFrame({
            "num1": [1.0, 2.0],
            "str_col": ["a", "b"],
            "num2": [3, 4],
            "bool_col": [True, False],
        })
        cols = _pick_numeric_columns(df, max_cols=10)
        assert "num1" in cols
        assert "num2" in cols
        assert "str_col" not in cols

    def test_max_cols_limit(self):
        df = pd.DataFrame({f"col_{i}": [float(i)] for i in range(10)})
        cols = _pick_numeric_columns(df, max_cols=3)
        assert len(cols) == 3

    def test_empty_dataframe(self):
        df = pd.DataFrame()
        cols = _pick_numeric_columns(df, max_cols=5)
        assert cols == []
