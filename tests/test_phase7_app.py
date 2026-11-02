"""Headless Phase 7 app and model-compatibility tests."""

from __future__ import annotations

import math
import unittest
from pathlib import Path

from streamlit.testing.v1 import AppTest

from app import DEFAULTS, build_feature_row, load_model_bundle


class Phase7AppTests(unittest.TestCase):
    def test_saved_models_accept_app_feature_row(self) -> None:
        classification_model, regression_model, feature_columns = load_model_bundle()
        features = build_feature_row(DEFAULTS, feature_columns)
        self.assertEqual(list(features.columns), feature_columns)
        self.assertEqual(features.shape, (1, len(feature_columns)))
        predicted_label = int(classification_model.predict(features)[0])
        projected_value = float(regression_model.predict(features)[0])
        self.assertIn(predicted_label, (0, 1))
        self.assertTrue(math.isfinite(projected_value))
        self.assertGreater(projected_value, 0)

    def test_streamlit_app_renders_and_predicts(self) -> None:
        app_test = AppTest.from_file(str(Path(__file__).resolve().parents[1] / "app.py"), default_timeout=30).run()
        self.assertFalse(app_test.exception)
        self.assertTrue(app_test.title)
        self.assertTrue(app_test.form_submit_button)

        app_test.form_submit_button[0].click().run()
        self.assertFalse(app_test.exception)
        self.assertGreaterEqual(len(app_test.metric), 3)
        self.assertTrue(app_test.info)

    def test_distance_inputs_cover_training_ranges(self) -> None:
        app_test = AppTest.from_file(
            str(Path(__file__).resolve().parents[1] / "app.py"),
            default_timeout=30,
        ).run()
        inputs = {widget.label: widget for widget in app_test.number_input}

        self.assertEqual(inputs["Distance to school (km)"].min, 0.0)
        self.assertEqual(inputs["Distance to school (km)"].max, 8.0)
        self.assertEqual(inputs["Distance to school (km)"].step, 0.1)
        self.assertEqual(inputs["Distance to hospital (km)"].min, 0.0)
        self.assertEqual(inputs["Distance to hospital (km)"].max, 9.6)
        self.assertEqual(inputs["Distance to hospital (km)"].step, 0.1)


if __name__ == "__main__":
    unittest.main()
