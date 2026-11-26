"""Reusable preprocessing for the IBM Telco customer churn dataset."""

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, OneHotEncoder, StandardScaler

NUMERIC_FEATURES = ["tenure", "MonthlyCharges", "TotalCharges"]
CATEGORICAL_FEATURES = [
    "gender",
    "SeniorCitizen",
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
FEATURES = NUMERIC_FEATURES + CATEGORICAL_FEATURES


def prepare_xy(data: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    """Separate the label and identifier before any learned transformations."""
    if not data.columns.is_unique:
        raise ValueError("Назви стовпців повинні бути унікальними.")
    required = set(FEATURES) | {"customerID", "Churn"}
    missing = required.difference(data.columns)
    extra = set(data.columns).difference(required)
    if missing or extra:
        raise ValueError(f"Неправильна схема: відсутні {sorted(missing)}, зайві {sorted(extra)}")
    if data["Churn"].isna().any() or not data["Churn"].isin(["Yes", "No"]).all():
        raise ValueError("Churn повинен містити тільки значення Yes або No.")
    return data.loc[:, FEATURES].copy(), data["Churn"].map({"No": 0, "Yes": 1})


def normalize_features(data: pd.DataFrame) -> pd.DataFrame:
    """Validate schema and normalize raw input without learning from data."""
    if not isinstance(data, pd.DataFrame):
        raise TypeError("Очікується pandas DataFrame з іменованими ознаками.")
    if not data.columns.is_unique:
        raise ValueError("Назви стовпців повинні бути унікальними.")
    missing = set(FEATURES).difference(data.columns)
    extra = set(data.columns).difference(FEATURES)
    if missing or extra:
        raise ValueError(f"Неправильна схема: відсутні {sorted(missing)}, зайві {sorted(extra)}")

    normalized = data.loc[:, FEATURES].copy()
    for column in NUMERIC_FEATURES:
        values = normalized[column].replace(r"^\s*$", np.nan, regex=True)
        normalized[column] = pd.to_numeric(values, errors="raise").astype(float)
    for column in CATEGORICAL_FEATURES:
        values = normalized[column].astype("string").str.strip()
        normalized[column] = values.mask(values.eq(""), pd.NA).astype(object).where(
            values.notna() & values.ne(""), np.nan
        )
    return normalized


def build_preprocessor() -> Pipeline:
    """Create an unfitted transformer for use inside a model pipeline."""
    numerical = Pipeline(
        [
            ("imputer", SimpleImputer(strategy="median", keep_empty_features=True)),
            ("scaler", StandardScaler()),
        ]
    )
    categorical = Pipeline(
        [
            ("imputer", SimpleImputer(strategy="constant", fill_value="__missing__")),
            ("encoder", OneHotEncoder(handle_unknown="ignore", sparse_output=True)),
        ]
    )
    return Pipeline(
        [
            ("normalize", FunctionTransformer(normalize_features, feature_names_out="one-to-one")),
            (
                "columns",
                ColumnTransformer(
                    [
                        ("numeric", numerical, NUMERIC_FEATURES),
                        ("categorical", categorical, CATEGORICAL_FEATURES),
                    ],
                    remainder="drop",
                ),
            ),
        ]
    )
