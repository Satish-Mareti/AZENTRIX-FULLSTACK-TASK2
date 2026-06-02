from __future__ import annotations

from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from imblearn.over_sampling import SMOTE, RandomOverSampler
from imblearn.pipeline import Pipeline as ImbPipeline
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import RandomizedSearchCV, StratifiedKFold, cross_validate, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


RANDOM_STATE = 42
TARGET_COLUMN = "Churn"
ID_COLUMN = "customerID"

RAW_NUMERIC_FEATURES = ["SeniorCitizen", "tenure", "MonthlyCharges", "TotalCharges"]
RAW_CATEGORICAL_FEATURES = [
    "gender",
    "Partner",
    "Dependents",
    "PhoneService",
    "MultipleLines",
    "InternetService",
    "OnlineSecurity",
    "OnlineBackup",
    "DeviceProtection",
    "TechSupport",
    "StreamingTV",
    "StreamingMovies",
    "Contract",
    "PaperlessBilling",
    "PaymentMethod",
]

ENGINEERED_NUMERIC_FEATURES = ["tenure_charge_ratio", "service_count", "is_month_to_month"]
NUMERIC_FEATURES = RAW_NUMERIC_FEATURES + ENGINEERED_NUMERIC_FEATURES
CATEGORICAL_FEATURES = RAW_CATEGORICAL_FEATURES

SERVICE_COUNT_COLUMNS = [
    "PhoneService",
    "MultipleLines",
    "OnlineSecurity",
    "OnlineBackup",
    "DeviceProtection",
    "TechSupport",
    "StreamingTV",
    "StreamingMovies",
]


def make_one_hot_encoder() -> OneHotEncoder:
    try:
        return OneHotEncoder(handle_unknown="ignore", sparse_output=False)
    except TypeError:  # pragma: no cover - backwards compatibility for older sklearn
        return OneHotEncoder(handle_unknown="ignore", sparse=False)


class TelcoFeatureEngineer(BaseEstimator, TransformerMixin):
    def fit(self, X: Any, y: Any = None) -> TelcoFeatureEngineer:
        return self

    def transform(self, X: Any) -> pd.DataFrame:
        frame = pd.DataFrame(X).copy()
        frame.columns = [str(column).strip() for column in frame.columns]

        if "TotalCharges" in frame.columns:
            frame["TotalCharges"] = pd.to_numeric(frame["TotalCharges"], errors="coerce")
        if "MonthlyCharges" in frame.columns:
            frame["MonthlyCharges"] = pd.to_numeric(frame["MonthlyCharges"], errors="coerce")
        if "tenure" in frame.columns:
            frame["tenure"] = pd.to_numeric(frame["tenure"], errors="coerce")
        if "SeniorCitizen" in frame.columns:
            frame["SeniorCitizen"] = pd.to_numeric(frame["SeniorCitizen"], errors="coerce")

        monthly = frame.get("MonthlyCharges", pd.Series(index=frame.index, dtype="float64"))
        total = frame.get("TotalCharges", pd.Series(index=frame.index, dtype="float64"))
        with np.errstate(divide="ignore", invalid="ignore"):
            ratio = np.divide(total, monthly)
        ratio = pd.Series(ratio, index=frame.index).replace([np.inf, -np.inf], np.nan).fillna(0.0)
        frame["tenure_charge_ratio"] = ratio

        service_flags = []
        for column in SERVICE_COUNT_COLUMNS:
            if column in frame.columns:
                service_flags.append(frame[column].astype(str).str.strip().str.lower().eq("yes").astype(int))
        if service_flags:
            frame["service_count"] = sum(service_flags)
        else:
            frame["service_count"] = 0

        if "Contract" in frame.columns:
            frame["is_month_to_month"] = frame["Contract"].astype(str).str.strip().eq("Month-to-month").astype(int)
        else:
            frame["is_month_to_month"] = 0

        return frame


def load_dataset(csv_path: str | Path) -> pd.DataFrame:
    frame = pd.read_csv(csv_path)
    frame.columns = [str(column).strip() for column in frame.columns]
    return frame


def clean_dataset(frame: pd.DataFrame) -> pd.DataFrame:
    cleaned = frame.copy()
    if ID_COLUMN in cleaned.columns:
        cleaned = cleaned.drop(columns=[ID_COLUMN])

    cleaned[TARGET_COLUMN] = cleaned[TARGET_COLUMN].astype(str).str.strip().map({"Yes": 1, "No": 0})
    cleaned = cleaned.dropna(subset=[TARGET_COLUMN])
    return cleaned


def build_preprocessor() -> ColumnTransformer:
    numeric_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )
    categorical_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("encoder", make_one_hot_encoder()),
        ]
    )
    return ColumnTransformer(
        transformers=[
            ("numeric", numeric_pipeline, NUMERIC_FEATURES),
            ("categorical", categorical_pipeline, CATEGORICAL_FEATURES),
        ],
        remainder="drop",
        sparse_threshold=0.0,
    )


def build_pipeline(estimator: BaseEstimator, sampler: str = "smote") -> ImbPipeline:
    steps = [
        ("feature_engineer", TelcoFeatureEngineer()),
        ("preprocessor", build_preprocessor()),
    ]

    sampler_lower = (sampler or "").lower()
    if sampler_lower == "smote":
        steps.append(("sampler", SMOTE(random_state=RANDOM_STATE)))
    elif sampler_lower == "oversample" or sampler_lower == "random":
        steps.append(("sampler", RandomOverSampler(random_state=RANDOM_STATE)))
    elif sampler_lower == "none":
        # no resampling step
        pass
    else:
        # SMOTENC or unknown sampler: fall back to RandomOverSampler and warn
        steps.append(("sampler", RandomOverSampler(random_state=RANDOM_STATE)))

    steps.append(("classifier", estimator))
    return ImbPipeline(steps=steps)


def build_model_candidates(random_state: int = RANDOM_STATE, sampler: str = "smote") -> dict[str, dict[str, Any]]:
    return {
        "logistic_regression": {
            "pipeline": build_pipeline(
                LogisticRegression(max_iter=2000, class_weight="balanced", solver="liblinear", random_state=random_state),
                sampler=sampler,
            ),
            "param_distributions": {
                "classifier__C": [0.1, 0.3, 1.0, 3.0, 10.0],
                "classifier__penalty": ["l1", "l2"],
            },
        },
        "random_forest": {
            "pipeline": build_pipeline(
                RandomForestClassifier(class_weight="balanced_subsample", random_state=random_state),
                sampler=sampler,
            ),
            "param_distributions": {
                "classifier__n_estimators": [200, 300, 500],
                "classifier__max_depth": [None, 8, 12, 16],
                "classifier__min_samples_leaf": [1, 2, 4],
                "classifier__min_samples_split": [2, 5, 10],
            },
        },
        "gradient_boosting": {
            "pipeline": build_pipeline(GradientBoostingClassifier(random_state=random_state), sampler=sampler),
            "param_distributions": {
                "classifier__n_estimators": [75, 100, 150],
                "classifier__learning_rate": [0.03, 0.05, 0.1],
                "classifier__max_depth": [2, 3, 4],
                "classifier__min_samples_leaf": [1, 2, 4],
            },
        },
    }


def create_eda_summary(frame: pd.DataFrame) -> str:
    total_rows = len(frame)
    churn_rate = frame[TARGET_COLUMN].mean()
    churn_counts = frame[TARGET_COLUMN].value_counts().sort_index()

    lines = [
        "# Telco Churn EDA Summary",
        "",
        f"Rows: {total_rows}",
        f"Features used for modeling: {len(frame.columns) - 1}",
        f"Target churn rate: {churn_rate:.2%}",
        "",
        "## Target distribution",
        f"- Not churned: {int(churn_counts.get(0, 0))}",
        f"- Churned: {int(churn_counts.get(1, 0))}",
        "",
        "## Missing values",
    ]

    missing = frame.isna().sum().sort_values(ascending=False)
    missing = missing[missing > 0]
    if missing.empty:
        lines.append("- No missing values detected after cleaning.")
    else:
        for column, count in missing.items():
            lines.append(f"- {column}: {int(count)}")

    lines.extend([
        "",
        "## Numeric snapshot",
        frame[RAW_NUMERIC_FEATURES].describe().to_string(),
    ])

    return "\n".join(lines)


def split_train_test(
    frame: pd.DataFrame,
    test_size: float = 0.2,
    random_state: int = RANDOM_STATE,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    features = frame.drop(columns=[TARGET_COLUMN])
    target = frame[TARGET_COLUMN].astype(int)
    return train_test_split(
        features,
        target,
        test_size=test_size,
        random_state=random_state,
        stratify=target,
    )


def compare_models(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    random_state: int = RANDOM_STATE,
    sampler: str = "smote",
) -> pd.DataFrame:
    scoring = {"roc_auc": "roc_auc", "f1": "f1", "recall": "recall", "accuracy": "accuracy"}
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=random_state)
    rows: list[dict[str, Any]] = []

    for model_name, spec in build_model_candidates(random_state, sampler=sampler).items():
        pipeline = spec["pipeline"]
        try:
            scores = cross_validate(
                pipeline,
                X_train,
                y_train,
                cv=cv,
                scoring=scoring,
                n_jobs=-1,
                error_score="raise",
            )
        except Exception:
            # SMOTE (or other sampler) can fail on certain data shapes/types.
            # Fall back to simple RandomOverSampler and retry the evaluation.
            ros = RandomOverSampler(random_state=random_state)
            # replace sampler step named 'smote' if present, else append
            steps = [(name, step) for name, step in pipeline.steps]
            replaced = False
            for i, (name, step) in enumerate(steps):
                if name.lower().find("smote") != -1:
                    steps[i] = (name, ros)
                    replaced = True
                    break
            if not replaced:
                steps.insert(-1, ("oversampler", ros))
            retry_pipeline = ImbPipeline(steps=steps)
            scores = cross_validate(
                retry_pipeline,
                X_train,
                y_train,
                cv=cv,
                scoring=scoring,
                n_jobs=-1,
                error_score="raise",
            )
        rows.append(
            {
                "model": model_name,
                "mean_accuracy": float(np.mean(scores["test_accuracy"])),
                "mean_roc_auc": float(np.mean(scores["test_roc_auc"])),
                "mean_f1": float(np.mean(scores["test_f1"])),
                "mean_recall": float(np.mean(scores["test_recall"])),
                "std_f1": float(np.std(scores["test_f1"])),
            }
        )

    comparison = pd.DataFrame(rows).sort_values(by=["mean_f1", "mean_roc_auc"], ascending=False).reset_index(drop=True)
    return comparison


def tune_best_model(
    best_model_name: str,
    X_train: pd.DataFrame,
    y_train: pd.Series,
    random_state: int = RANDOM_STATE,
    sampler: str = "smote",
) -> RandomizedSearchCV:
    candidates = build_model_candidates(random_state, sampler=sampler)
    if best_model_name not in candidates:
        raise ValueError(f"Unknown model '{best_model_name}'.")

    spec = candidates[best_model_name]
    search = RandomizedSearchCV(
        estimator=spec["pipeline"],
        param_distributions=spec["param_distributions"],
        n_iter=min(10, sum(len(values) for values in spec["param_distributions"].values())),
        scoring="f1",
        cv=StratifiedKFold(n_splits=5, shuffle=True, random_state=random_state),
        random_state=random_state,
        n_jobs=-1,
        refit=True,
    )
    try:
        search.fit(X_train, y_train)
    except Exception:
        # If sampler in the pipeline fails (e.g., SMOTE on incompatible data),
        # replace with RandomOverSampler and retry the randomized search.
        ros = RandomOverSampler(random_state=random_state)
        pipeline = spec["pipeline"]
        steps = [(name, step) for name, step in pipeline.steps]
        replaced = False
        for i, (name, step) in enumerate(steps):
            if name.lower().find("smote") != -1:
                steps[i] = (name, ros)
                replaced = True
                break
        if not replaced:
            steps.insert(-1, ("oversampler", ros))
        retry_pipeline = ImbPipeline(steps=steps)
        retry_search = RandomizedSearchCV(
            estimator=retry_pipeline,
            param_distributions=spec["param_distributions"],
            n_iter=min(10, sum(len(values) for values in spec["param_distributions"].values())),
            scoring="f1",
            cv=StratifiedKFold(n_splits=5, shuffle=True, random_state=random_state),
            random_state=random_state,
            n_jobs=-1,
            refit=True,
        )
        retry_search.fit(X_train, y_train)
        return retry_search
    return search


def evaluate_model(model: Any, X_test: pd.DataFrame, y_test: pd.Series) -> dict[str, Any]:
    predictions = model.predict(X_test)
    probabilities = model.predict_proba(X_test)[:, 1]
    return {
        "accuracy": float(accuracy_score(y_test, predictions)),
        "precision": float(precision_score(y_test, predictions, zero_division=0)),
        "recall": float(recall_score(y_test, predictions, zero_division=0)),
        "f1": float(f1_score(y_test, predictions, zero_division=0)),
        "roc_auc": float(roc_auc_score(y_test, probabilities)),
        "classification_report": classification_report(y_test, predictions, zero_division=0),
    }


def train_end_to_end(
    csv_path: str | Path,
    model_path: str | Path,
    eda_path: str | Path | None = None,
    comparison_path: str | Path | None = None,
    random_state: int = RANDOM_STATE,
    sampler: str = "smote",
) -> dict[str, Any]:
    raw_frame = load_dataset(csv_path)
    cleaned_frame = clean_dataset(raw_frame)

    if eda_path is not None:
        Path(eda_path).parent.mkdir(parents=True, exist_ok=True)
        Path(eda_path).write_text(create_eda_summary(cleaned_frame), encoding="utf-8")

    X_train, X_test, y_train, y_test = split_train_test(cleaned_frame, random_state=random_state)
    comparison = compare_models(X_train, y_train, random_state=random_state, sampler=sampler)

    if comparison_path is not None:
        Path(comparison_path).parent.mkdir(parents=True, exist_ok=True)
        comparison.to_csv(comparison_path, index=False)

    best_model_name = str(comparison.iloc[0]["model"])
    search = tune_best_model(best_model_name, X_train, y_train, random_state=random_state, sampler=sampler)
    tuned_model = search.best_estimator_
    test_metrics = evaluate_model(tuned_model, X_test, y_test)

    artifact = {
        "model": tuned_model,
        "best_model_name": best_model_name,
        "best_params": search.best_params_,
        "comparison": comparison,
        "test_metrics": test_metrics,
        "numeric_features": NUMERIC_FEATURES,
        "categorical_features": CATEGORICAL_FEATURES,
        "feature_columns": list(X_train.columns),
        "target_column": TARGET_COLUMN,
        "random_state": random_state,
    }

    model_path = Path(model_path)
    model_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(artifact, model_path)

    return artifact


def load_model_bundle(model_path: str | Path) -> dict[str, Any]:
    return joblib.load(model_path)


def predict_churn(model_bundle: dict[str, Any], record: dict[str, Any]) -> dict[str, Any]:
    model = model_bundle["model"]
    frame = pd.DataFrame([record])
    probability = float(model.predict_proba(frame)[:, 1][0])
    prediction = int(model.predict(frame)[0])
    return {
        "prediction": prediction,
        "label": "Churn" if prediction == 1 else "No churn",
        "probability": probability,
    }


def parse_input_json(raw_value: str) -> dict[str, Any]:
    import json

    candidate = raw_value.strip()
    if not candidate:
        raise ValueError("Input JSON cannot be empty.")
    data = json.loads(candidate)
    if not isinstance(data, dict):
        raise ValueError("Input JSON must decode to an object with feature names.")
    return data
