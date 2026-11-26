import unittest

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.model_selection import StratifiedKFold

from src.preprocessing import (
    CATEGORICAL_FEATURES,
    FEATURES,
    build_preprocessor,
    prepare_xy,
)


def example_data():
    rows = []
    for i in range(8):
        row = {name: "Yes" for name in CATEGORICAL_FEATURES}
        row.update(
            customerID=f"id-{i}",
            Churn="Yes" if i % 2 else "No",
            tenure=i + 1,
            MonthlyCharges=20.0 + i,
            TotalCharges=" " if i == 0 else str((i + 1) * 20),
            SeniorCitizen=i % 2,
            Contract="Month-to-month" if i % 2 else "Two year",
        )
        rows.append(row)
    return pd.DataFrame(rows)


class PreprocessingTests(unittest.TestCase):
    def setUp(self):
        self.data = example_data()
        self.x, self.y = prepare_xy(self.data)

    def test_prepare_xy_excludes_identifier_and_encodes_target(self):
        self.assertEqual(list(self.x.columns), FEATURES)
        self.assertEqual(self.y.tolist(), [0, 1] * 4)
        with self.assertRaises(ValueError):
            prepare_xy(self.data.assign(Churn="unknown"))
        with self.assertRaises(ValueError):
            prepare_xy(self.data.assign(unexpected=1))
        with self.assertRaises(ValueError):
            prepare_xy(pd.concat([self.data, self.data[["Churn"]]], axis=1))

    def test_training_median_unknown_categories_and_missing_values(self):
        pipeline = build_preprocessor()
        pipeline.fit(self.x)
        numerical = pipeline.named_steps["columns"].named_transformers_["numeric"]
        self.assertEqual(numerical.named_steps["imputer"].statistics_[2], 100.0)
        new = self.x.iloc[:2].copy()
        new.loc[new.index[0], "Contract"] = "unknown contract"
        new.loc[new.index[0], "TotalCharges"] = " "
        new.loc[new.index[1], "TechSupport"] = None
        transformed = pipeline.transform(new)
        array = transformed.toarray() if hasattr(transformed, "toarray") else transformed
        self.assertEqual(array.shape[0], 2)
        self.assertTrue(np.isfinite(array).all())
        self.assertEqual(array.shape[1], len(pipeline.get_feature_names_out()))

    def test_invalid_columns_and_numeric_values_raise(self):
        for invalid in (
            self.x.drop(columns="Contract"),
            self.x.assign(Churn="Yes"),
            self.x.assign(TotalCharges="not-a-number"),
        ):
            with self.subTest(columns=list(invalid.columns)):
                with self.assertRaises(ValueError):
                    build_preprocessor().fit(invalid)

    def test_each_fold_learns_its_own_statistics(self):
        folds = StratifiedKFold(n_splits=2, shuffle=True, random_state=42)
        for train_indices, validation_indices in folds.split(self.x, self.y):
            training = self.x.iloc[train_indices]
            validation = self.x.iloc[validation_indices]
            pipeline = clone(build_preprocessor())
            pipeline.fit(training)
            numerical = pipeline.named_steps["columns"].named_transformers_["numeric"]
            values = pd.to_numeric(training["TotalCharges"].replace(r"^\s*$", np.nan, regex=True))
            self.assertEqual(numerical.named_steps["imputer"].statistics_[2], values.median())
            self.assertEqual(pipeline.transform(validation).shape[0], len(validation))


if __name__ == "__main__":
    unittest.main()
