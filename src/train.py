"""Compare candidate churn models on training data with cross-validation."""

import pandas as pd
from catboost import CatBoostClassifier
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import make_scorer, precision_score, recall_score, f1_score
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.tree import DecisionTreeClassifier

from src.preprocessing import build_preprocessor

SEED = 42
SCORING = {
    "precision": make_scorer(precision_score, zero_division=0),
    "recall": make_scorer(recall_score, zero_division=0),
    "f1": make_scorer(f1_score, zero_division=0),
    "roc_auc": "roc_auc",
    "pr_auc": "average_precision",
}


def candidate_models():
    """Return unfitted estimators and small, explicit search grids."""
    return {
        "Baseline": (DummyClassifier(strategy="most_frequent"), {}),
        "Logistic Regression": (
            LogisticRegression(max_iter=2000, random_state=SEED),
            {"model__C": [0.1, 1.0], "model__class_weight": [None, "balanced"]},
        ),
        "Decision Tree": (
            DecisionTreeClassifier(random_state=SEED),
            {"model__max_depth": [3, 6], "model__min_samples_leaf": [10, 30]},
        ),
        "Random Forest": (
            RandomForestClassifier(n_estimators=100, random_state=SEED, n_jobs=1),
            {"model__max_depth": [6, None], "model__min_samples_leaf": [5, 15]},
        ),
        "CatBoost": (
            CatBoostClassifier(
                iterations=150, random_seed=SEED, verbose=False,
                thread_count=2, allow_writing_files=False,
            ),
            {"model__depth": [4, 6], "model__learning_rate": [0.05, 0.15]},
        ),
    }


def compare_models(X: pd.DataFrame, y: pd.Series) -> pd.DataFrame:
    """Tune on mean F1 and report the same folds' scores, without holdout use."""
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
    rows = []
    for name, (estimator, grid) in candidate_models().items():
        pipeline = Pipeline([("preprocess", build_preprocessor()), ("model", estimator)])
        search = GridSearchCV(
            pipeline, grid, scoring=SCORING, refit=False, cv=cv, n_jobs=1,
            error_score="raise", return_train_score=False,
        )
        search.fit(X, y)
        results = pd.DataFrame(search.cv_results_)
        best_index = results["mean_test_f1"].idxmax()
        row = {"model": name, "best_params": results.loc[best_index, "params"]}
        for metric in SCORING:
            row[f"{metric}_mean"] = results.loc[best_index, f"mean_test_{metric}"]
            row[f"{metric}_std"] = results.loc[best_index, f"std_test_{metric}"]
        rows.append(row)
    return pd.DataFrame(rows).sort_values("f1_mean", ascending=False).reset_index(drop=True)
