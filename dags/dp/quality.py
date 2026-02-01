from __future__ import annotations
import pandas as pd

class DataNotTrainable(Exception):
    pass

def assert_trainable(df: pd.DataFrame, label_col: str = "label"):
    if label_col not in df.columns:
        raise DataNotTrainable(f"missing label column: {label_col}")

    if df.shape[0] < 50:
        raise DataNotTrainable(f"too few rows: {df.shape[0]} (need >= 50 for stable demo)")

    vc = df[label_col].value_counts(dropna=False)
    if vc.shape[0] < 2:
        raise DataNotTrainable(f"only one class in label: {vc.to_dict()}")

    # constant feature check (exclude id/label/timestamp)
    ignore = {"user_id", label_col, "event_timestamp"}
    for c in df.columns:
        if c in ignore:
            continue
        if df[c].nunique(dropna=False) <= 1:
            raise DataNotTrainable(f"constant feature: {c}")

