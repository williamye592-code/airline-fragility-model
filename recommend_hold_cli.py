from __future__ import annotations

import argparse
from pathlib import Path
import joblib
import numpy as np
import pandas as pd

from src.decision.cost_policy import build_decision_table
from src.utils.config import MODELS_DIR


FEATURE_COLS = [
    "FLEET",
    "BASE",
    "DIVISION",
    "FLIGHT_PATTERN",
    "TOTAL_BLOCKED_HRS",
    "SEQ_CAL_DAYS",
    "SEQ_DUTY_DAYS",
    "SEQ_TTL_FLTTIME",
    "MIN_FLYTIME_PER_LEG",
    "MAX_LEGS_PER_DAY",
    "SEQ_TTL_LEGS",
    "LAYOVER",
    "seq_month",
    "seq_dayofweek",
    "seq_start_hour",
    "is_weekend",
]


def parse_args():
    parser = argparse.ArgumentParser(
        description="Recommend an integer-minute hold time for one airline sequence."
    )

    parser.add_argument("--fleet", type=str, required=True)
    parser.add_argument("--base", type=str, required=True)
    parser.add_argument("--division", type=str, required=True)
    parser.add_argument("--flight_pattern", type=str, required=True)

    parser.add_argument("--total_blocked_hrs", type=float, required=True)
    parser.add_argument("--seq_cal_days", type=float, required=True)
    parser.add_argument("--seq_duty_days", type=float, required=True)
    parser.add_argument("--seq_ttl_flttime", type=float, required=True)
    parser.add_argument("--min_flytime_per_leg", type=float, required=True)
    parser.add_argument("--max_legs_per_day", type=float, required=True)
    parser.add_argument("--seq_ttl_legs", type=float, required=True)
    parser.add_argument("--layover", type=float, required=True)

    parser.add_argument("--seq_month", type=int, required=True)
    parser.add_argument("--seq_dayofweek", type=int, required=True)
    parser.add_argument("--seq_start_hour", type=int, required=True)
    parser.add_argument("--is_weekend", type=int, required=True)

    parser.add_argument("--delay_cost_per_min", type=float, default=100.0)
    parser.add_argument("--disruption_cost", type=float, default=3000.0)
    parser.add_argument("--max_hold", type=int, default=15)

    return parser.parse_args()


def build_single_input_df(args) -> pd.DataFrame:
    row = {
        "FLEET": args.fleet,
        "BASE": args.base,
        "DIVISION": args.division,
        "FLIGHT_PATTERN": args.flight_pattern,
        "TOTAL_BLOCKED_HRS": args.total_blocked_hrs,
        "SEQ_CAL_DAYS": args.seq_cal_days,
        "SEQ_DUTY_DAYS": args.seq_duty_days,
        "SEQ_TTL_FLTTIME": args.seq_ttl_flttime,
        "MIN_FLYTIME_PER_LEG": args.min_flytime_per_leg,
        "MAX_LEGS_PER_DAY": args.max_legs_per_day,
        "SEQ_TTL_LEGS": args.seq_ttl_legs,
        "LAYOVER": args.layover,
        "seq_month": args.seq_month,
        "seq_dayofweek": args.seq_dayofweek,
        "seq_start_hour": args.seq_start_hour,
        "is_weekend": args.is_weekend,
    }
    return pd.DataFrame([row], columns=FEATURE_COLS)


def load_logistic_model():
    """
    Adjust the filename below if your saved model file uses a different name.
    """
    candidate_paths = [
        MODELS_DIR / "logistic_regression.joblib",
        MODELS_DIR / "logit_pipeline.joblib",
        MODELS_DIR / "logistic_pipeline.joblib",
    ]

    for path in candidate_paths:
        if path.exists():
            return joblib.load(path), path

    raise FileNotFoundError(
        f"Could not find a saved logistic model in {MODELS_DIR}. "
        f"Tried: {[str(p) for p in candidate_paths]}"
    )


def main():
    args = parse_args()

    model, model_path = load_logistic_model()
    print(f"Loaded model from: {model_path}")

    X_one = build_single_input_df(args)

    p0 = model.predict_proba(X_one)[:, 1]

    hold_options = list(range(0, args.max_hold + 1))

    decision_df = build_decision_table(
        X_input=X_one,
        p0=p0,
        hold_options=hold_options,
        delay_cost_per_min=args.delay_cost_per_min,
        disruption_cost=args.disruption_cost,
    )

    row = decision_df.iloc[0]

    print("\n=== Recommendation Result ===")
    print(f"Predicted disruption risk: {row['predicted_risk']:.4f}")
    print(f"Severity proxy: {row['severity_proxy']:.4f}")
    print(f"Beta proxy: {row['beta_proxy']:.4f}")
    print(f"Recommended hold: {int(row['recommended_hold'])} minutes")
    print(f"Recommended expected cost: {row['recommended_cost']:.2f}")
    print(f"Expected benefit vs no-hold: {row['expected_benefit_vs_no_hold']:.2f}")

    print("\n=== Cost by Hold Minute ===")
    for h in hold_options:
        col = f"cost_hold_{h}"
        print(f"hold={h:2d} min -> cost={row[col]:.2f}")


if __name__ == "__main__":
    main()