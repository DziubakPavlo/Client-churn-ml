"""Reproducible, stratified holdout split for the pinned Telco dataset."""

from hashlib import sha256
from io import BytesIO
from pathlib import Path

import pandas as pd

DATASET_SHA256 = "16320c9c1ec72448db59aa0a26a0b95401046bef5d02fd3aeb906448e3055e91"
SPLIT_SEED = 42


def load_dataset(path: Path) -> pd.DataFrame:
    data = path.read_bytes()
    actual_hash = sha256(data).hexdigest()
    if actual_hash != DATASET_SHA256:
        raise ValueError(
            f"Невідповідний SHA-256 датасету: {actual_hash}. "
            "Завантажте зафіксований CSV згідно з README."
        )
    return pd.read_csv(BytesIO(data), dtype={"customerID": "string"})


def split_dataset(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    required = {"customerID", "Churn"}
    missing = required.difference(df.columns)
    if missing:
        raise ValueError(f"Відсутні стовпці: {sorted(missing)}")
    ids = df["customerID"]
    if ids.isna().any() or ids.astype("string").str.strip().eq("").any():
        raise ValueError("customerID не може бути порожнім.")
    if ids.duplicated().any():
        raise ValueError("customerID повинен бути унікальним.")
    if df["Churn"].isna().any() or set(df["Churn"]) != {"Yes", "No"}:
        raise ValueError("Churn повинен містити тільки обидва класи Yes і No.")

    test_ids: set[str] = set()
    for label in ("No", "Yes"):
        class_ids = df.loc[df["Churn"] == label, "customerID"].astype(str)
        if len(class_ids) < 3:
            raise ValueError(f"Недостатньо записів класу {label} для розділення.")
        ranked_ids = sorted(
            class_ids,
            key=lambda customer_id: (
                sha256(f"{SPLIT_SEED}:{customer_id}".encode("utf-8")).digest(),
                customer_id,
            ),
        )
        test_count = (len(ranked_ids) + 2) // 5
        test_ids.update(ranked_ids[:test_count])

    test_mask = ids.isin(test_ids)
    return df.loc[~test_mask].copy(), df.loc[test_mask].copy()
