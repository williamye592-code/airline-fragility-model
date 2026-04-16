from __future__ import annotations

import numpy as np
import pandas as pd


# =========================================================
# Basic assumptions for the first-pass decision prototype
# =========================================================

# Approximate cost of holding an aircraft for 1 minute
DELAY_COST_PER_MIN = 100.0

# First-pass scenario assumption for a disruption event
DISRUPTION_COST = 5000.0

# Candidate hold actions (in minutes)
HOLD_OPTIONS = list(range(0, 16))


# =========================================================
# Severity proxy
# =========================================================
def build_severity_proxy(df: pd.DataFrame) -> pd.Series:
    """
    Build a simple sequence-level severity proxy.

    This is NOT passenger-level impact.
    It is a rough operational severity multiplier used to
    differentiate low-impact vs high-impact disruptions.

    Higher value => disruption is more costly for this sequence.
    """
    severity = pd.Series(1.0, index=df.index, dtype=float)

    if "SEQ_TTL_LEGS" in df.columns:
        severity += 0.10 * pd.to_numeric(df["SEQ_TTL_LEGS"], errors="coerce").fillna(0)

    if "SEQ_TTL_FLTTIME" in df.columns:
        severity += 0.03 * pd.to_numeric(df["SEQ_TTL_FLTTIME"], errors="coerce").fillna(0)

    if "MAX_LEGS_PER_DAY" in df.columns:
        severity += 0.10 * pd.to_numeric(df["MAX_LEGS_PER_DAY"], errors="coerce").fillna(0)

    if "LAYOVER" in df.columns:
        severity += 0.03 * pd.to_numeric(df["LAYOVER"], errors="coerce").fillna(0)

    if "seq_start_hour" in df.columns:
        start_hr = pd.to_numeric(df["seq_start_hour"], errors="coerce")
        # Early morning / late night mild uplift
        severity += np.where((start_hr <= 6) | (start_hr >= 20), 0.10, 0.0)
    
    if "SEQ_TTL_LEGS" in df.columns:
        legs = pd.to_numeric(df["SEQ_TTL_LEGS"], errors="coerce").fillna(0)
        severity += np.where(legs >= 4, 0.30, 0.0)

    return severity


# =========================================================
# Hold-effect proxy
# =========================================================
def build_beta_proxy(df: pd.DataFrame) -> pd.Series:
    """
    Build a simple beta parameter that controls how effective
    holding is at reducing disruption probability.

    We model:
        p_after_hold = p0 * exp(-beta * hold_minutes)

    Larger beta => hold helps more.
    """
    beta = pd.Series(0.015, index=df.index, dtype=float)

    if "SEQ_TTL_LEGS" in df.columns:
        legs = pd.to_numeric(df["SEQ_TTL_LEGS"], errors="coerce").fillna(0)
        beta += np.where(legs >= 3, 0.005, 0.0)

    if "MAX_LEGS_PER_DAY" in df.columns:
        max_legs = pd.to_numeric(df["MAX_LEGS_PER_DAY"], errors="coerce").fillna(0)
        beta += np.where(max_legs >= 3, 0.005, 0.0)

    if "MIN_FLYTIME_PER_LEG" in df.columns:
        min_leg = pd.to_numeric(df["MIN_FLYTIME_PER_LEG"], errors="coerce").fillna(0)
        median_val = min_leg[min_leg.notna()].median() if min_leg.notna().any() else 0
        beta += np.where(min_leg <= median_val, 0.005, 0.0)

    return beta


# =========================================================
# Probability update under hold
# =========================================================
def p_after_hold(p0: np.ndarray, hold_minutes: float, beta: np.ndarray) -> np.ndarray:
    """
    Scenario-based approximation:
        p_hold = p0 * exp(-beta * hold_minutes)

    This is not causal identification.
    It is a first-pass decision modeling assumption.
    """
    return p0 * np.exp(-beta * hold_minutes)


# =========================================================
# Expected cost
# =========================================================
def expected_cost(
    p0: np.ndarray,
    hold_minutes: float,
    beta: np.ndarray,
    severity_multiplier: np.ndarray,
    delay_cost_per_min: float = DELAY_COST_PER_MIN,
    disruption_cost: float = DISRUPTION_COST,
) -> np.ndarray:
    """
    Expected cost = hold cost + expected disruption cost

    hold cost = hold_minutes * delay_cost_per_min
    disruption cost = p_after_hold * disruption_cost * severity_multiplier
    """
    hold_component = hold_minutes * delay_cost_per_min
    p_hold = p_after_hold(p0, hold_minutes, beta)
    disruption_component = p_hold * disruption_cost * severity_multiplier

    # Nonlinear penalty for long holds:
    # small holds are manageable, but longer holds create
    # additional propagation / network / operational cost
    gamma = 20.0
    hold_penalty = gamma * (hold_minutes ** 2)

    return hold_component + disruption_component + hold_penalty


# =========================================================
# Main decision table builder
# =========================================================
def build_decision_table(
    X_input: pd.DataFrame,
    p0: np.ndarray,
    hold_options: list[int] | None = None,
    delay_cost_per_min: float = DELAY_COST_PER_MIN,
    disruption_cost: float = DISRUPTION_COST,
) -> pd.DataFrame:
    """
    Build a per-sequence decision table.

    Inputs:
        X_input: feature table for the scored samples
        p0: predicted disruption probability from the ML model

    Outputs:
        DataFrame containing:
        - predicted risk
        - severity proxy
        - beta proxy
        - expected cost under each hold option
        - recommended hold time
        - expected benefit vs no-hold
    """
    if hold_options is None:
        hold_options = HOLD_OPTIONS

    out = X_input.copy().reset_index(drop=True)

    # Ensure p0 is a numpy array aligned with out
    p0 = np.asarray(p0, dtype=float)
    out["predicted_risk"] = p0

    severity = build_severity_proxy(out).to_numpy(dtype=float)
    beta = build_beta_proxy(out).to_numpy(dtype=float)

    out["severity_proxy"] = severity
    out["beta_proxy"] = beta

    cost_cols = []

    for h in hold_options:
        col = f"cost_hold_{h}"
        out[col] = expected_cost(
            p0=p0,
            hold_minutes=h,
            beta=beta,
            severity_multiplier=severity,
            delay_cost_per_min=delay_cost_per_min,
            disruption_cost=disruption_cost,
        )
        cost_cols.append(col)

    out["recommended_hold"] = (
        out[cost_cols]
        .idxmin(axis=1)
        .str.replace("cost_hold_", "", regex=False)
        .astype(int)
    )

    out["recommended_cost"] = out[cost_cols].min(axis=1)
    out["cost_no_hold"] = out["cost_hold_0"]
    out["expected_benefit_vs_no_hold"] = out["cost_no_hold"] - out["recommended_cost"]

    return out