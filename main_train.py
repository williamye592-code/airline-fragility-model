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


def main():

    print_section("STEP 1: Load raw data")
    df = load_raw_data()
    print(f"Raw shape: {df.shape}")


    print_section("STEP 2: Preprocess raw data")
    df = preprocess_raw_data(df)
    print(f"Post-preprocess shape: {df.shape}")


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

    print_section("DONE")
    print("Outputs saved to outputs/models, outputs/tables, outputs/plots")


if __name__ == "__main__":
    main()