from __future__ import annotations

from pathlib import Path
import json
import pandas as pd
import numpy as np

from src.data.load_data import load_raw_data
from src.data.preprocess import preprocess_raw_data
from src.features.build_features import build_modeling_table, get_X_y
from src.models.train_baseline import (
    time_based_split,
    build_rf_pipeline,
    train_and_evaluate_model,
)
from src.decision.severity import build_severity_proxy, build_beta_proxy


BASE_DIR = Path(__file__).resolve().parent
OUTPUT_TABLES_DIR = BASE_DIR / "outputs" / "tables"
OUTPUT_MODELS_DIR = BASE_DIR / "outputs" / "models"

OUTPUT_TABLES_DIR.mkdir(parents=True, exist_ok=True)
OUTPUT_MODELS_DIR.mkdir(parents=True, exist_ok=True)


# ----------------------------
# decision / cost assumptions
# ----------------------------

HOLD_MINUTES = 5.0
HOLD_COST_PER_MIN = 1.0


def p_after_hold(p0: np.ndarray, beta: np.ndarray, hold_minutes: float) -> np.ndarray:
    """
    Counterfactual approximation:
        p_hold = p0 * exp(-beta * h)

    This is a scenario-based engineering assumption, not a causal estimate.
    """
    return p0 * np.exp(-beta * hold_minutes)


def expected_cost_no_hold(p0: np.ndarray, severity: np.ndarray) -> np.ndarray:
    return p0 * severity


def expected_cost_hold(
    p0: np.ndarray,
    severity: np.ndarray,
    beta: np.ndarray,
    hold_minutes: float,
    hold_cost_per_min: float,
) -> np.ndarray:
    hold_cost = hold_minutes * hold_cost_per_min
    p_hold = p_after_hold(p0, beta, hold_minutes)
    return hold_cost + p_hold * severity


def add_decision_columns(
    scored_df: pd.DataFrame,
    hold_minutes: float = HOLD_MINUTES,
    hold_cost_per_min: float = HOLD_COST_PER_MIN,
) -> pd.DataFrame:
    out = scored_df.copy()

    p0 = out["p_spoilage"].to_numpy(dtype=float)
    sev = out["severity_proxy"].to_numpy(dtype=float)
    beta = out["beta_hold_effect"].to_numpy(dtype=float)

    out["p_spoilage_hold_5"] = p_after_hold(p0, beta, hold_minutes)
    out["expected_cost_no_hold"] = expected_cost_no_hold(p0, sev)
    out["expected_cost_hold_5"] = expected_cost_hold(
        p0, sev, beta, hold_minutes, hold_cost_per_min
    )
    out["benefit_hold_5"] = out["expected_cost_no_hold"] - out["expected_cost_hold_5"]
    out["recommend_hold_5"] = (out["benefit_hold_5"] > 0).astype(int)

    return out


def main():
    print("STEP 1: load raw data")
    raw_df = load_raw_data()

    print("STEP 2: preprocess")
    clean_df = preprocess_raw_data(raw_df)

    print("STEP 3: build modeling table")
    modeling_df = build_modeling_table(clean_df)

    print("STEP 4: get X, y")
    X, y = get_X_y(modeling_df)

    # Keep a copy for later merge-back
    full_df = modeling_df.copy()

    print("STEP 5: time-based split")
    X_train, X_test, y_train, y_test = time_based_split(X, y)

    # Recover the corresponding rows from modeling_df via index
    train_idx = X_train.index
    test_idx = X_test.index

    train_df = full_df.loc[train_idx].copy()
    test_df = full_df.loc[test_idx].copy()

    print("STEP 6: train RF as main scoring model")
    rf_pipeline = build_rf_pipeline()
    trained_rf, rf_results = train_and_evaluate_model(
        "random_forest",
        rf_pipeline,
        X_train, y_train, X_test, y_test
    )

    print("RF test results:")
    print(rf_results)

    print("STEP 7: generate out-of-sample probabilities on test set")
    p_test = trained_rf.predict_proba(X_test)[:, 1]
    pred_test = (p_test >= 0.5).astype(int)

    scored_test_df = test_df.copy()
    scored_test_df["y_true"] = y_test.values
    scored_test_df["p_spoilage"] = p_test
    scored_test_df["pred_label"] = pred_test

    print("STEP 8: build severity + beta proxy")
    scored_test_df["severity_proxy"] = build_severity_proxy(scored_test_df)
    scored_test_df["beta_hold_effect"] = build_beta_proxy(scored_test_df)

    print("STEP 9: build decision columns")
    decision_test_df = add_decision_columns(scored_test_df)

    print("STEP 10: save outputs")
    table_path = OUTPUT_TABLES_DIR / "decision_ready_test.csv"
    decision_test_df.to_csv(table_path, index=False)

    summary = {
        "n_test": int(len(decision_test_df)),
        "avg_p_spoilage": float(decision_test_df["p_spoilage"].mean()),
        "avg_severity_proxy": float(decision_test_df["severity_proxy"].mean()),
        "avg_expected_cost_no_hold": float(decision_test_df["expected_cost_no_hold"].mean()),
        "avg_expected_cost_hold_5": float(decision_test_df["expected_cost_hold_5"].mean()),
        "hold_5_recommend_rate": float(decision_test_df["recommend_hold_5"].mean()),
        "total_benefit_hold_5": float(decision_test_df["benefit_hold_5"].sum()),
    }

    summary_path = OUTPUT_TABLES_DIR / "decision_summary.json"
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print(f"Saved: {table_path}")
    print(f"Saved: {summary_path}")
    print("DONE.")


if __name__ == "__main__":
    main()