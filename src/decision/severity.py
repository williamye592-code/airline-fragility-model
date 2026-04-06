from __future__ import annotations

from typing import Dict, Iterable
import pandas as pd


def _safe_get(row: pd.Series, col: str, default: float = 0.0) -> float:
    """
    Safely get a numeric value from a row.
    Returns default if column is missing or value is NaN / non-numeric.
    """
    if col not in row.index:
        return default
    val = row[col]
    try:
        if pd.isna(val):
            return default
        return float(val)
    except Exception:
        return default


def build_severity_proxy(
    df: pd.DataFrame,
    config: Dict[str, float] | None = None
) -> pd.Series:
    """
    Build a sequence-level disruption severity proxy.

    This is NOT passenger-level missed-connection cost.
    It is a sequence-level operational severity score used for
    decision modeling under limited observability.

    The function is intentionally robust to missing columns.
    If a column is absent, its contribution is treated as 0.

    Suggested columns from your current sequence-level data:
        - SEQ_TTL_LEGS
        - TOTAL_BLOCKED_HRS
        - TOTAL_SPOILED_HRS
        - SEQ_TTL_FLTTIME
        - MAX_LEGS_PER_DAY
        - SEQ_DUTY_DAYS
        - SEQ_CAL_DAYS
        - IN_SEQ_DHD
        - LAYOVER
        - MORETHAN2_321_LEGS
        - SEQ_START_HRS
    """

    default_config = {
        # base
        "base": 10.0,

        # linear weights
        "w_seq_ttl_legs": 2.0,
        "w_total_blocked_hrs": 1.2,
        "w_total_spoiled_hrs": 2.0,
        "w_seq_ttl_flttime": 0.8,
        "w_max_legs_per_day": 1.5,
        "w_seq_duty_days": 1.0,
        "w_seq_cal_days": 0.8,

        # binary / categorical-style add-ons
        "w_in_seq_dhd": 3.0,
        "w_layover": 1.5,
        "w_morethan2_321_legs": 2.5,

        # time-of-day uplift
        "early_hour_bonus": 2.0,   # if very early start
        "late_hour_bonus": 2.0,    # if very late start
    }

    if config is not None:
        default_config.update(config)

    def score_row(row: pd.Series) -> float:
        s = default_config["base"]

        s += default_config["w_seq_ttl_legs"] * _safe_get(row, "SEQ_TTL_LEGS")
        s += default_config["w_total_blocked_hrs"] * _safe_get(row, "TOTAL_BLOCKED_HRS")
        s += default_config["w_total_spoiled_hrs"] * _safe_get(row, "TOTAL_SPOILED_HRS")
        s += default_config["w_seq_ttl_flttime"] * _safe_get(row, "SEQ_TTL_FLTTIME")
        s += default_config["w_max_legs_per_day"] * _safe_get(row, "MAX_LEGS_PER_DAY")
        s += default_config["w_seq_duty_days"] * _safe_get(row, "SEQ_DUTY_DAYS")
        s += default_config["w_seq_cal_days"] * _safe_get(row, "SEQ_CAL_DAYS")

        if _safe_get(row, "IN_SEQ_DHD") > 0:
            s += default_config["w_in_seq_dhd"]

        if _safe_get(row, "LAYOVER") > 0:
            s += default_config["w_layover"]

        if _safe_get(row, "MORETHAN2_321_LEGS") > 0:
            s += default_config["w_morethan2_321_legs"]

        start_hr = _safe_get(row, "SEQ_START_HRS", default=-1)
        if 0 <= start_hr <= 6:
            s += default_config["early_hour_bonus"]
        if 20 <= start_hr <= 23:
            s += default_config["late_hour_bonus"]

        # keep strictly positive
        return max(s, 1.0)

    return df.apply(score_row, axis=1)


def build_beta_proxy(
    df: pd.DataFrame,
    base_beta: float = 0.06
) -> pd.Series:
    """
    Build a simple heterogeneous hold-effect parameter beta_i.

    Interpretation:
        p_after_hold(h) = p0 * exp(-beta_i * h)

    Larger beta_i => holding helps more.

    We proxy this using sequence tightness / complexity indicators.
    """
    betas = []

    for _, row in df.iterrows():
        beta = base_beta

        legs = _safe_get(row, "SEQ_TTL_LEGS")
        blocked = _safe_get(row, "TOTAL_BLOCKED_HRS")
        max_legs_per_day = _safe_get(row, "MAX_LEGS_PER_DAY")
        spoiled_hrs = _safe_get(row, "TOTAL_SPOILED_HRS")
        start_hr = _safe_get(row, "SEQ_START_HRS", default=-1)

        if legs >= 3:
            beta += 0.01
        if blocked >= 10:
            beta += 0.01
        if max_legs_per_day >= 3:
            beta += 0.01
        if spoiled_hrs > 0:
            beta += 0.015
        if 0 <= start_hr <= 6:
            beta += 0.005

        betas.append(beta)

    return pd.Series(betas, index=df.index)