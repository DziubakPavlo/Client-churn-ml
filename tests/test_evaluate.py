import unittest

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score

from src.evaluate import build_selected_model, evaluate_predictions
from src.preprocessing import prepare_xy
from test_preprocessing import example_data


class EvaluationTests(unittest.TestCase):
    def test_metrics_and_confusion_matrix(self):
        labels = [0, 0, 1, 1]
        scores = [0.1, 0.6, 0.4, 0.9]
        metrics, matrix = evaluate_predictions(labels, scores)
        np.testing.assert_array_equal(matrix, [[1, 1], [1, 1]])
        self.assertAlmostEqual(metrics["precision"], 0.5)
        self.assertAlmostEqual(metrics["recall"], 0.5)
        self.assertAlmostEqual(metrics["f1"], 0.5)
        self.assertEqual(metrics["roc_auc"], roc_auc_score(labels, scores))
        self.assertEqual(metrics["average_precision"], average_precision_score(labels, scores))

    def test_threshold_and_invalid_inputs(self):
        _, matrix = evaluate_predictions([0, 1], [0.5, 0.5])
        np.testing.assert_array_equal(matrix, [[0, 1], [0, 1]])
        for labels, scores, threshold in [
            ([0, 1], [0.1], 0.5),
            ([0, 0], [0.1, 0.2], 0.5),
            ([0, 1], [0.1, np.nan], 0.5),
            ([0, 1], [0.1, 1.1], 0.5),
            ([0, 1], [0.1, 0.2], -0.1),
        ]:
            with self.subTest(labels=labels, scores=scores, threshold=threshold):
                with self.assertRaises(ValueError):
                    evaluate_predictions(labels, scores, threshold)

    def test_pipeline_fit_and_predict(self):
        data = pd.concat([example_data()] * 2, ignore_index=True)
        X, y = prepare_xy(data)
        model = build_selected_model()
        model.fit(X.iloc[:8], y.iloc[:8])
        scores = model.predict_proba(X.iloc[8:])[:, 1]
        metrics, matrix = evaluate_predictions(y.iloc[8:], scores)
        self.assertEqual(matrix.sum(), 8)
        self.assertTrue(all(np.isfinite(value) for value in metrics.values()))


if __name__ == "__main__":
    unittest.main()
