import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

import pandas as pd

from src.data_split import load_dataset, split_dataset


class SplitTests(unittest.TestCase):
    def setUp(self):
        self.data = pd.DataFrame(
            {
                "customerID": [f"customer-{i}" for i in range(20)],
                "Churn": ["No"] * 15 + ["Yes"] * 5,
            }
        )

    def test_stratified_disjoint_and_order_independent(self):
        train, test = split_dataset(self.data)
        shuffled_train, shuffled_test = split_dataset(
            self.data.sample(frac=1, random_state=1)
        )
        self.assertEqual(len(train), 16)
        self.assertEqual(test["Churn"].value_counts().to_dict(), {"No": 3, "Yes": 1})
        self.assertEqual(
            set(test["customerID"]),
            {"customer-0", "customer-2", "customer-12", "customer-16"},
        )
        self.assertFalse(set(train["customerID"]) & set(test["customerID"]))
        self.assertEqual(
            set(train["customerID"]) | set(test["customerID"]),
            set(self.data["customerID"]),
        )
        self.assertEqual(set(train["customerID"]), set(shuffled_train["customerID"]))
        self.assertEqual(set(test["customerID"]), set(shuffled_test["customerID"]))

    def test_invalid_inputs_are_rejected(self):
        for invalid in (
            self.data.drop(columns="Churn"),
            self.data.assign(customerID="duplicate"),
            self.data.assign(Churn="unknown"),
            self.data.assign(customerID=[""] + list(self.data["customerID"][1:])),
        ):
            with self.subTest(columns=list(invalid.columns)):
                with self.assertRaises(ValueError):
                    split_dataset(invalid)

    def test_unexpected_dataset_bytes_are_rejected(self):
        with TemporaryDirectory() as directory:
            path = Path(directory) / "data.csv"
            path.write_text("customerID,Churn\n1,No\n", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "SHA-256"):
                load_dataset(path)


if __name__ == "__main__":
    unittest.main()
