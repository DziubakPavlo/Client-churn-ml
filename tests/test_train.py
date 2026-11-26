import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd
from sklearn.dummy import DummyClassifier
from sklearn.linear_model import LogisticRegression

from src.train import SCORING, compare_models
from test_preprocessing import example_data
from src.preprocessing import prepare_xy


class TrainingTests(unittest.TestCase):
    def test_cv_results_and_grid_selection(self):
        data = pd.concat([example_data()] * 3, ignore_index=True)
        data.loc[data.index % 8 == 1, "Churn"] = "No"
        X, y = prepare_xy(data)
        candidates = {
            "Baseline": (DummyClassifier(strategy="most_frequent"), {}),
            "Logistic": (
                LogisticRegression(max_iter=500),
                {"model__C": [0.1, 1.0]},
            ),
        }
        with patch("src.train.candidate_models", return_value=candidates):
            results = compare_models(X, y)
        self.assertEqual(set(results["model"]), set(candidates))
        self.assertEqual(len(results), 2)
        for metric in SCORING:
            self.assertTrue(np.isfinite(results[f"{metric}_mean"]).all())
            self.assertTrue(np.isfinite(results[f"{metric}_std"]).all())
        baseline = results.set_index("model").loc["Baseline"]
        self.assertEqual(baseline["f1_mean"], 0)
        self.assertEqual(baseline["best_params"], {})
        self.assertIn(results.set_index("model").loc["Logistic", "best_params"]["model__C"], [0.1, 1.0])


if __name__ == "__main__":
    unittest.main()
