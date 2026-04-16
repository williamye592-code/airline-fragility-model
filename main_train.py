import pandas as pd;
from src.data.load_data import load_raw_data
from src.data.preprocess import preprocess_raw_data
from src.features.build_features import build_modeling_table, get_X_y
from src.models.train_baseline import (
    time_based_split,
    build_logistic_pipeline,
    build_rf_pipeline,
    train_and_evaluate_model,
    extract_logistic_feature_importance,
    extract_rf_feature_importance,
)
from src.models.evaluate import plot_roc_curve, plot_pr_curve
from src.utils.config import TABLES_DIR, PLOTS_DIR
from src.utils.helpers import print_section, save_dataframe, summarize_target
from src.decision.cost_policy import build_decision_table


def main():

    print_section("STEP 1: Load raw data")
    df = load_raw_data()
    print(f"Raw shape: {df.shape}")


    print_section("STEP 2: Preprocess raw data")
    df = preprocess_raw_data(df)
    print(f"Post-preprocess shape: {df.shape}")
    
    print("\nLeakage sanity check:")
    check_cols = [c for c in ["SPOILAGE", "TOTAL_SPOILED_HRS"] if c in df.columns]
    if len(check_cols) == 2:
        print(df[check_cols].head())
        print(df.groupby("SPOILAGE")["TOTAL_SPOILED_HRS"].describe())


    print_section("STEP 3: Build modeling table")
    df = build_modeling_table(df)
    print(f"Modeling table shape: {df.shape}")


    print_section("STEP 4: Prepare X and y")
    X, y = get_X_y(df)
    print(f"X shape: {X.shape}")
    print(f"y shape: {y.shape}")

    target_summary = summarize_target(y)
    print("\nTarget distribution:")
    print(target_summary)
    save_dataframe(target_summary, TABLES_DIR / "target_distribution.csv", index=False)


    print_section("STEP 5: Time-based split")
    X_train, X_test, y_train, y_test = time_based_split(X, y)
    print(f"Train shape: {X_train.shape}, Test shape: {X_test.shape}")


    print_section("STEP 6: Train Logistic Regression")
    logit_pipeline = build_logistic_pipeline()
    trained_logit, logit_results = train_and_evaluate_model(
        "logistic_regression",
        logit_pipeline,
        X_train, y_train, X_test, y_test
    )
    print(logit_results)

    logit_importance = extract_logistic_feature_importance(trained_logit)
    save_dataframe(logit_importance, TABLES_DIR / "logistic_feature_importance.csv", index=False)
    print("\nTop logistic features:")
    print(logit_importance.head(10))

    plot_roc_curve(
        trained_logit,
        X_test,
        y_test,
        PLOTS_DIR / "logistic_roc.png",
        "Logistic Regression ROC Curve",
    )
    plot_pr_curve(
        trained_logit,
        X_test,
        y_test,
        PLOTS_DIR / "logistic_pr.png",
        "Logistic Regression PR Curve",
    )


    print_section("STEP 7: Train Random Forest")
    rf_pipeline = build_rf_pipeline()
    trained_rf, rf_results = train_and_evaluate_model(
        "random_forest",
        rf_pipeline,
        X_train, y_train, X_test, y_test
    )
    print(rf_results)

    rf_importance = extract_rf_feature_importance(trained_rf)
    save_dataframe(rf_importance, TABLES_DIR / "rf_feature_importance.csv", index=False)
    print("\nTop RF features:")
    print(rf_importance.head(10))

    plot_roc_curve(
        trained_rf,
        X_test,
        y_test,
        PLOTS_DIR / "rf_roc.png",
        "Random Forest ROC Curve",
    )
    plot_pr_curve(
        trained_rf,
        X_test,
        y_test,
        PLOTS_DIR / "rf_pr.png",
        "Random Forest PR Curve",
    )

    print_section("STEP 8: Build decision recommendation table")

    # First-pass decision layer uses logistic predicted probabilities
    logit_test_probs = trained_logit.predict_proba(X_test)[:, 1]

    decision_df = build_decision_table(
        X_input=X_test.reset_index(drop=True),
        p0=logit_test_probs,
        hold_options=[0, 5, 10, 15],
        delay_cost_per_min=100.0,
        disruption_cost=3000.0,
    )

    save_dataframe(decision_df, TABLES_DIR / "decision_table_logit.csv", index=False)

    print("\nDecision table preview:")
    preview_cols = [
        "predicted_risk",
        "severity_proxy",
        "beta_proxy",
        "cost_hold_0",
        "cost_hold_5",
        "cost_hold_10",
        "cost_hold_15",
        "recommended_hold",
        "expected_benefit_vs_no_hold",
    ]
    print(decision_df[preview_cols].head(10))

    print("\nRecommended hold distribution:")
    print(decision_df["recommended_hold"].value_counts().sort_index())

    print("\nAverage expected benefit vs no-hold:")
    print(decision_df["expected_benefit_vs_no_hold"].mean())

    
    print_section("STEP 8.5: Policy Comparison")

    policy_eval_df = build_decision_table(
        X_input=X_test.reset_index(drop=True),
        p0=logit_test_probs,
        hold_options=[0, 5, 10, 15],
        delay_cost_per_min=100.0,
        disruption_cost=3000.0,
    )

    policy_summary = pd.DataFrame({
        "policy": ["No Hold", "Always Hold 5", "Model Recommended"],
        "avg_expected_cost": [
            policy_eval_df["cost_hold_0"].mean(),
            policy_eval_df["cost_hold_5"].mean(),
            policy_eval_df["recommended_cost"].mean(),
        ]
    })

    save_dataframe(policy_summary, TABLES_DIR / "policy_comparison_summary.csv", index=False)

    print("\nPolicy comparison summary:")
    print(policy_summary)
    



    print_section("STEP 9: Sensitivity Analysis")

    sensitivity_rows = []
    hold_dist_rows = []

    for disruption_cost in [1000, 2000, 3000, 5000]:

        decision_df = build_decision_table(
            X_input=X_test.reset_index(drop=True),
            p0=logit_test_probs,
            hold_options=[0, 5, 10, 15],
            delay_cost_per_min=100.0,
            disruption_cost=disruption_cost,
        )

        hold_counts = decision_df["recommended_hold"].value_counts().to_dict()

        row = {
            "disruption_cost": disruption_cost,
            "n_hold_0": hold_counts.get(0, 0),
            "n_hold_5": hold_counts.get(5, 0),
            "n_hold_10": hold_counts.get(10, 0),
            "n_hold_15": hold_counts.get(15, 0),
            "avg_benefit": decision_df["expected_benefit_vs_no_hold"].mean(),
            "avg_recommended_hold": decision_df["recommended_hold"].mean(),
        }
        sensitivity_rows.append(row)

        total_n = len(decision_df)
        hold_dist_rows.append({
            "disruption_cost": disruption_cost,
            "hold_0_pct": hold_counts.get(0, 0) / total_n,
            "hold_5_pct": hold_counts.get(5, 0) / total_n,
            "hold_10_pct": hold_counts.get(10, 0) / total_n,
            "hold_15_pct": hold_counts.get(15, 0) / total_n,
        })

        print(f"\n=== Disruption cost: {disruption_cost} ===")
        print("Hold distribution:")
        print(decision_df["recommended_hold"].value_counts().sort_index())
        print("Avg benefit:")
        print(decision_df["expected_benefit_vs_no_hold"].mean())

    sensitivity_df = pd.DataFrame(sensitivity_rows)
    hold_dist_df = pd.DataFrame(hold_dist_rows)

    save_dataframe(sensitivity_df, TABLES_DIR / "sensitivity_summary.csv", index=False)
    save_dataframe(hold_dist_df, TABLES_DIR / "hold_distribution_summary.csv", index=False)

    print("\nSensitivity summary:")
    print(sensitivity_df)

    print("\nHold distribution summary:")
    print(hold_dist_df)

    

    import matplotlib.pyplot as plt

    print_section("STEP 10: Plot Sensitivity Curve")

    # 读刚刚生成的 summary（或者直接用内存里的 sensitivity_df）
    costs = sensitivity_df["disruption_cost"]
    avg_hold = sensitivity_df["avg_recommended_hold"]

    plt.figure()
    plt.plot(costs, avg_hold, marker='o')

    plt.xlabel("Disruption Cost (USD)")
    plt.ylabel("Average Recommended Hold (minutes)")
    plt.title("Sensitivity of Hold Policy to Disruption Cost")

    plt.grid(True)

    plt.tight_layout()
    plt.savefig(PLOTS_DIR / "sensitivity_curve.png", dpi=300, bbox_inches="tight")
    plt.close()

    print_section("DONE")
    print("Outputs saved to outputs/models, outputs/tables, outputs/plots")


if __name__ == "__main__":
    main()