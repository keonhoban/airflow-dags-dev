from __future__ import annotations
import pandas as pd
import numpy as np

def build_features_with_label(raw: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    """
    raw columns expected:
      user_id, event_time, event_type, amount, session_length_sec, is_premium ...
    output:
      features: user_id + feature cols + label
      meta: schema_hash inputs etc (caller fills)
    """
    raw = raw.copy()
    raw["event_time"] = pd.to_datetime(raw["event_time"], utc=True, errors="coerce")
    raw["amount"] = pd.to_numeric(raw.get("amount", 0), errors="coerce").fillna(0)

    # per-user aggregates
    g = raw.groupby("user_id", dropna=False)

    feats = pd.DataFrame({
        "user_id": g.size().index,
        "f_total_events_7d": g.size().values,
        "f_avg_session_sec_7d": g["session_length_sec"].mean().fillna(0).values,
        "f_total_amount_7d": g["amount"].sum().values,
        "f_purchase_cnt_7d": g.apply(lambda x: (x["event_type"] == "purchase").sum()).values,
        "event_timestamp": raw["event_time"].max(),
    })

    # label (3-class): 0=no spend, 1=low, 2=high
    total_amount = feats["f_total_amount_7d"].values
    nonzero = total_amount[total_amount > 0]
    if len(nonzero) >= 4:
        p75 = np.quantile(nonzero, 0.75)
    else:
        p75 = float(nonzero.max()) if len(nonzero) else 0.0

    label = np.zeros(len(feats), dtype=int)
    label[(total_amount > 0) & (total_amount <= p75)] = 1
    label[total_amount > p75] = 2
    feats["label"] = label

    # minimal sanity
    feats.replace([np.inf, -np.inf], np.nan, inplace=True)
    feats.fillna(0, inplace=True)

    meta = {
        "n_rows": int(feats.shape[0]),
        "n_features": int(feats.shape[1] - 2),  # user_id, label 제외
        "n_classes": int(feats["label"].nunique()),
    }
    return feats, meta

