import numpy as np
import pandas as pd

from src.utils.config import (
    DATE_COL,
    START_TIME_COL,
    TARGET_SOURCE_COL,
    TARGET_COL,
)


def parse_dates(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    if DATE_COL in df.columns:
        df[DATE_COL] = pd.to_datetime(df[DATE_COL], errors="coerce")

    if START_TIME_COL in df.columns:
        df[START_TIME_COL] = df[START_TIME_COL].astype(str).str.strip()

    return df


def create_target(df: pd.DataFrame) -> pd.DataFrame:
    """
    y = 1 if spoiled / partially spoiled
    y = 0 if NOT SPOILED
    """
    df = df.copy()

    spoilage_clean = (
        df[TARGET_SOURCE_COL]
        .astype(str)
        .str.upper()
        .str.strip()
    )

    df[TARGET_COL] = np.where(spoilage_clean == "NOT SPOILED", 0, 1)
    return df


def basic_cleaning(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()


    if DATE_COL in df.columns:
        df = df.dropna(subset=[DATE_COL])


    for col in ["FLEET", "BASE", "DIVISION", "FLIGHT_PATTERN"]:
        if col in df.columns:
            df[col] = (
                df[col]
                .astype(str)
                .str.strip()
                .replace({"": "UNKNOWN", "nan": "UNKNOWN", "None": "UNKNOWN"})
            )

    return df


def sort_by_time(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df = df.sort_values(DATE_COL).reset_index(drop=True)
    return df


def preprocess_raw_data(df: pd.DataFrame) -> pd.DataFrame:
    df = parse_dates(df)
    df = create_target(df)
    df = basic_cleaning(df)
    df = sort_by_time(df)
    return df