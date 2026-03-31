import joblib
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier

from src.utils.config import (
    CATEGORICAL_COLS,
    NUMERICAL_COLS,
    ENGINEERED_NUM_COLS,
    TEST_SIZE_FRACTION,
    RANDOM_STATE,
    MODELS_DIR,
)
from src.models.evaluate import evaluate_classifier, save_metrics


ALL_NUMERIC_COLS = NUMERICAL_COLS + ENGINEERED_NUM_COLS


def time_based_split(X: pd.DataFrame, y: pd.Series, test_size=TEST_SIZE_FRACTION):
    """
    Assume rows are already sorted by time.
    """
    n = len(X)
    split_idx = int(n * (1 - test_size))

    X_train = X.iloc[:split_idx].copy()
    X_test = X.iloc[split_idx:].copy()
    y_train = y.iloc[:split_idx].copy()
    y_test = y.iloc[split_idx:].copy()

    return X_train, X_test, y_train, y_test


def build_preprocessor():
    numeric_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )

    categorical_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore")),
        ]
    )

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_transformer, ALL_NUMERIC_COLS),
            ("cat", categorical_transformer, CATEGORICAL_COLS),
        ]
    )

    return preprocessor


def build_logistic_pipeline():
    pipeline = Pipeline(
        steps=[
            ("preprocessor", build_preprocessor()),
            (
                "classifier",
                LogisticRegression(
                    max_iter=1000,
                    class_weight="balanced",
                    random_state=RANDOM_STATE,
                ),
            ),
        ]
    )
    return pipeline


def build_rf_pipeline():
    pipeline = Pipeline(
        steps=[
            ("preprocessor", build_preprocessor()),
            (
                "classifier",
                RandomForestClassifier(
                    n_estimators=300,
                    min_samples_leaf=5,
                    class_weight="balanced",
                    random_state=RANDOM_STATE,
                    n_jobs=-1,
                ),
            ),
        ]
    )
    return pipeline


def train_and_evaluate_model(model_name, pipeline, X_train, y_train, X_test, y_test):
    pipeline.fit(X_train, y_train)

    results = evaluate_classifier(pipeline, X_train, y_train, X_test, y_test)

    model_path = MODELS_DIR / f"{model_name}.joblib"
    metrics_path = MODELS_DIR / f"{model_name}_metrics.json"

    joblib.dump(pipeline, model_path)
    save_metrics(results, metrics_path)

    return pipeline, results


def extract_logistic_feature_importance(trained_pipeline) -> pd.DataFrame:
    preprocessor = trained_pipeline.named_steps["preprocessor"]
    classifier = trained_pipeline.named_steps["classifier"]

    feature_names = preprocessor.get_feature_names_out()
    coefficients = classifier.coef_[0]

    importance_df = pd.DataFrame({
        "feature": feature_names,
        "coefficient": coefficients,
        "abs_coefficient": abs(coefficients),
    }).sort_values("abs_coefficient", ascending=False)

    return importance_df


def extract_rf_feature_importance(trained_pipeline) -> pd.DataFrame:
    preprocessor = trained_pipeline.named_steps["preprocessor"]
    classifier = trained_pipeline.named_steps["classifier"]

    feature_names = preprocessor.get_feature_names_out()
    importances = classifier.feature_importances_

    importance_df = pd.DataFrame({
        "feature": feature_names,
        "importance": importances,
    }).sort_values("importance", ascending=False)

    return importance_df