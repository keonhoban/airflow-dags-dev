# tests/test_train_validation.py
"""
train_model.py 데이터 검증 로직 단위 테스트.

_validate_training_data: 학습 전 데이터 품질 검증
- 최소 행 수, 클래스 수, 클래스 불균형 체크
- TrainSkippableError vs 일반 에러 분리
"""
from __future__ import annotations

import pandas as pd
import pytest

from ml_code.train_model import TrainSkippableError, _validate_training_data
from mlops_lib.dp.feature_schema import FEATURE_COLS, LABEL_COL


def _make_df(n: int = 100, n_classes: int = 2) -> pd.DataFrame:
    """테스트용 DataFrame 생성."""
    data = {col: list(range(n)) for col in FEATURE_COLS}
    data[LABEL_COL] = [i % n_classes for i in range(n)]
    return pd.DataFrame(data)


class TestValidateTrainingData:
    """_validate_training_data 검증 로직 테스트."""

    def test_valid_data_returns_labels(self):
        df = _make_df(100, 2)
        y = _validate_training_data(df)
        assert len(y) == 100
        assert sorted(y.unique().tolist()) == [0, 1]

    def test_missing_feature_columns_raises(self):
        df = pd.DataFrame({"wrong_col": [1, 2], LABEL_COL: [0, 1]})
        with pytest.raises(TrainSkippableError, match="컬럼 누락"):
            _validate_training_data(df)

    def test_missing_label_column_raises(self):
        data = {col: list(range(50)) for col in FEATURE_COLS}
        df = pd.DataFrame(data)  # label 없음
        with pytest.raises(TrainSkippableError, match="label 컬럼"):
            _validate_training_data(df)

    def test_too_few_rows_raises(self):
        df = _make_df(10, 2)  # 20 미만
        with pytest.raises(TrainSkippableError, match="rows=10"):
            _validate_training_data(df)

    def test_single_class_raises(self):
        df = _make_df(50, 1)  # 단일 클래스
        with pytest.raises(TrainSkippableError, match="클래스 부족"):
            _validate_training_data(df)

    def test_class_imbalance_raises(self):
        """한 클래스에 2개 미만이면 TrainSkippableError."""
        data = {col: list(range(50)) for col in FEATURE_COLS}
        # 48개는 클래스 0, 2개는 클래스 1 → min count = 2 < 3
        data[LABEL_COL] = [0] * 48 + [1] * 2
        df = pd.DataFrame(data)
        with pytest.raises(TrainSkippableError, match="클래스 불균형"):
            _validate_training_data(df)

    def test_boundary_exactly_20_rows_passes(self):
        df = _make_df(20, 2)
        y = _validate_training_data(df)
        assert len(y) == 20

    def test_multiclass_works(self):
        df = _make_df(90, 3)
        y = _validate_training_data(df)
        assert sorted(y.unique().tolist()) == [0, 1, 2]
