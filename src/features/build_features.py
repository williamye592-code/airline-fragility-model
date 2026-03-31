import numpy as np
import pandas as pd

from src.utils.config import (
    DATE_COL,
    START_TIME_COL,
    TARGET_COL,
    FEATURE_COLS,
)


def _extract_hour(value):
    """
    Robustly parse hour from examples like:
    '13:45', '1345', '930', '09', etc.
    """
    if pd.isna(value):
        return np.nan

    s = str(value).strip()

    if s == "" or s.lower() == "nan":
        return np.nan

    if ":" in s:
        try:
            hour = int(s.split(":")[0])
            return hour if 0 <= hour <= 23 else np.nan
        except Exception:
            return np.nan

    digits = "".join(ch for ch in s if ch.isdigit())

    if digits == "":
        return np.nan

    if len(digits) >= 3:
        try:
            hour = int(digits[:-2])
            return hour if 0 <= hour <= 23 else np.nan
        except Exception:
            return np.nan


    if len(digits) <= 2:
        try:
            hour = int(digits)
            return hour if 0 <= hour <= 23 else np.nan
        except Exception:
            return np.nan

    return np.nan


def add_time_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    df["seq_month"] = df[DATE_COL].dt.month
    df["seq_dayofweek"] = df[DATE_COL].dt.dayofweek
    df["is_weekend"] = df["seq_dayofweek"].isin([5, 6]).astype(int)

    if START_TIME_COL in df.columns:
        df["seq_start_hour"] = df[START_TIME_COL].apply(_extract_hour)
    else:
        df["seq_start_hour"] = np.nan

    return df


def build_modeling_table(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df = add_time_features(df)
    return df


def get_X_y(df: pd.DataFrame):
    X = df[FEATURE_COLS].copy()
    y = df[TARGET_COL].copy()
    return X, y