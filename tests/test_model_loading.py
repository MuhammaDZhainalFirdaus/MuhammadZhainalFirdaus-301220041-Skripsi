"""Unit and Integration Tests for Android Malware Scanner Model Verification.

Verifies:
1. Presence of all 6 research artifacts in models/
2. Exact structure and count of feature_names.csv (24,583 features) and selected_features.csv (4,916 features)
3. Loading of xgboost_rfe.json via XGBoost (4,916 input features)
4. Loading of rfe_selector.joblib via Joblib (24,583 input features, 4,916 selected)
5. Dimension matching and indexing alignment between RFE support mask and selected_features.csv
6. Direct column indexing optimization produces output 100% bitwise identical to RFE transform
7. End-to-end technical inference test using dummy data shape (1, 24583) and (N, 24583)

IMPORTANT: Dummy data testing verifies technical dataflow and numerical properties only,
and DOES NOT evaluate malware detection accuracy.
"""

from __future__ import annotations

import os
from pathlib import Path
import sys
import unittest

import numpy as np
import pandas as pd

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.model_loader import (
    DEFAULT_MODELS_DIR,
    AndroidMalwarePipeline,
    load_feature_names,
    load_metadata,
    load_metrics,
    load_rfe_selector,
    load_selected_features,
    load_xgboost_model,
)


class TestModelArtifacts(unittest.TestCase):
    """Test verification of model and feature artifact files."""

    def setUp(self) -> None:
        self.models_dir = DEFAULT_MODELS_DIR
        self.required_files = [
            "xgboost_rfe.json",
            "rfe_selector.joblib",
            "feature_names.csv",
            "selected_features.csv",
            "model_metadata.json",
            "metrics_rfe.json",
        ]

    def test_artifacts_existence(self) -> None:
        """Verify all 6 research files exist and are non-empty."""
        for filename in self.required_files:
            file_path = self.models_dir / filename
            self.assertTrue(
                file_path.is_file(),
                f"Missing required model artifact: {file_path}",
            )
            self.assertGreater(
                file_path.stat().st_size,
                0,
                f"Artifact file is empty: {file_path}",
            )

    def test_feature_names_csv_structure_and_count(self) -> None:
        """Verify feature_names.csv has columns ['feature_index', 'feature_name'] and 24,583 rows."""
        fn_df = load_feature_names(self.models_dir / "feature_names.csv")
        self.assertEqual(
            list(fn_df.columns),
            ["feature_index", "feature_name"],
            "Column names in feature_names.csv do not match expected schema",
        )
        self.assertEqual(
            len(fn_df),
            24583,
            f"Expected exactly 24,583 features, found {len(fn_df)}",
        )
        # Check feature index continuity: 0 to 24582
        self.assertTrue(
            np.array_equal(fn_df["feature_index"].values, np.arange(24583)),
            "feature_index in feature_names.csv is not continuous 0..24582",
        )

        # Verify prefix breakdown matches metadata: 166 Permission, 24,417 APICall
        perms = fn_df[fn_df["feature_name"].str.startswith("Permission::")]
        apis = fn_df[fn_df["feature_name"].str.startswith("APICall::")]
        self.assertEqual(len(perms), 166, f"Expected 166 permissions, found {len(perms)}")
        self.assertEqual(len(apis), 24417, f"Expected 24417 API calls, found {len(apis)}")
        self.assertEqual(len(perms) + len(apis), 24583)

    def test_selected_features_csv_structure_and_count(self) -> None:
        """Verify selected_features.csv has columns ['selected_index', 'feature_index', 'feature_name'] and 4,916 rows."""
        sf_df = load_selected_features(self.models_dir / "selected_features.csv")
        self.assertEqual(
            list(sf_df.columns),
            ["selected_index", "feature_index", "feature_name"],
            "Column names in selected_features.csv do not match expected schema",
        )
        self.assertEqual(
            len(sf_df),
            4916,
            f"Expected exactly 4,916 selected features, found {len(sf_df)}",
        )
        # Check selected index continuity: 0 to 4915
        self.assertTrue(
            np.array_equal(sf_df["selected_index"].values, np.arange(4916)),
            "selected_index in selected_features.csv is not continuous 0..4915",
        )

    def test_metadata_and_metrics_consistency(self) -> None:
        """Verify model_metadata.json and metrics_rfe.json consistency."""
        metadata = load_metadata(self.models_dir / "model_metadata.json")
        metrics = load_metrics(self.models_dir / "metrics_rfe.json")

        self.assertEqual(metadata["original_features"], 24583)
        self.assertEqual(metadata["selected_features"], 4916)
        self.assertEqual(metadata["permission_features_original"], 166)
        self.assertEqual(metadata["api_call_features_original"], 24417)

        self.assertEqual(metrics["features_before_rfe"], 24583)
        self.assertEqual(metrics["features_after_rfe"], 4916)


class TestModelLoadingAndInference(unittest.TestCase):
    """Test loading of XGBoost model, RFE selector, and dummy dataflow."""

    @classmethod
    def setUpClass(cls) -> None:
        cls.models_dir = DEFAULT_MODELS_DIR
        # Fast direct indexing pipeline (app production mode)
        cls.pipeline_direct = AndroidMalwarePipeline(cls.models_dir, use_rfe_joblib=False)
        # Verification pipeline with explicit RFE joblib loader
        cls.pipeline_rfe = AndroidMalwarePipeline(cls.models_dir, use_rfe_joblib=True)

    def test_xgboost_model_loading_and_dimensions(self) -> None:
        """Verify xgboost_rfe.json loads with input dimension 4,916."""
        model = self.pipeline_direct.model
        self.assertEqual(
            model.n_features_in_,
            4916,
            f"XGBoost expected 4,916 input features, got {model.n_features_in_}",
        )
        self.assertTrue(
            np.array_equal(model.classes_, np.array([0, 1])),
            f"XGBoost classes expected [0, 1], got {model.classes_}",
        )

    def test_rfe_selector_loading_and_dimensions(self) -> None:
        """Verify rfe_selector.joblib loads with input 24,583 and output 4,916."""
        selector = self.pipeline_rfe.selector
        self.assertEqual(
            selector.n_features_in_,
            24583,
            f"RFE selector expected 24,583 input features, got {selector.n_features_in_}",
        )
        self.assertEqual(
            selector.n_features_,
            4916,
            f"RFE selector output expected 4,916 features, got {selector.n_features_}",
        )
        self.assertEqual(
            selector.support_.shape,
            (24583,),
            f"RFE support_ shape expected (24583,), got {selector.support_.shape}",
        )
        self.assertEqual(
            selector.support_.sum(),
            4916,
            f"RFE support_ active features count expected 4,916, got {selector.support_.sum()}",
        )

    def test_selector_and_selected_features_csv_alignment(self) -> None:
        """Verify that RFE support_ mask indices perfectly match selected_features.csv feature_index."""
        selector = self.pipeline_rfe.selector
        sf_df = self.pipeline_direct.selected_features_df
        fn_df = self.pipeline_direct.feature_names_df

        rfe_indices = np.where(selector.support_)[0]
        csv_indices = sf_df["feature_index"].values

        self.assertTrue(
            np.array_equal(rfe_indices, csv_indices),
            "RFE support_ mask indices do NOT match selected_features.csv feature_index!",
        )

        rfe_names = fn_df.iloc[rfe_indices]["feature_name"].values
        csv_names = sf_df["feature_name"].values
        self.assertTrue(
            np.array_equal(rfe_names, csv_names),
            "Feature names at selected indices do NOT match selected_features.csv feature_name!",
        )

    def test_direct_indexing_identity_with_rfe_transform(self) -> None:
        """Verify that direct column indexing via selected_features.csv produces 100% identical matrices to RFE transform."""
        np.random.seed(42)
        random_batch = np.random.randn(5, 24583).astype(np.float32)

        trans_direct = self.pipeline_direct.transform(random_batch)
        trans_rfe = self.pipeline_rfe.transform(random_batch)

        self.assertEqual(trans_direct.shape, (5, 4916))
        self.assertEqual(trans_rfe.shape, (5, 4916))
        self.assertTrue(
            np.array_equal(trans_direct, trans_rfe),
            "Direct indexing transformed matrix is not bitwise identical to RFE transform!",
        )

        proba_direct = self.pipeline_direct.predict_proba(random_batch)
        proba_rfe = self.pipeline_rfe.predict_proba(random_batch)
        self.assertTrue(
            np.allclose(proba_direct, proba_rfe),
            "Predicted probabilities differ between direct indexing and RFE transform!",
        )

        preds_direct, labels_direct = self.pipeline_direct.predict(random_batch)
        preds_rfe, labels_rfe = self.pipeline_rfe.predict(random_batch)
        self.assertTrue(
            np.array_equal(preds_direct, preds_rfe),
            "Predicted classes differ between direct indexing and RFE transform!",
        )
        self.assertTrue(
            np.array_equal(labels_direct, labels_rfe),
            "Predicted labels differ between direct indexing and RFE transform!",
        )

    def test_dummy_data_flow_single_sample(self) -> None:
        """Verify data dummy (1, 24583) through direct indexing and XGBoost model.

        NOTE: Dummy data only validates technical dataflow and numerical properties
        (finite probabilities, sum close to 1.0). It DOES NOT validate detection accuracy.
        """
        # Create zero dummy array
        dummy_zeros = np.zeros((1, 24583), dtype=np.float32)

        # 1. Test transformation
        selected_features = self.pipeline_direct.transform(dummy_zeros)
        self.assertEqual(
            selected_features.shape,
            (1, 4916),
            f"Transformed dummy shape should be (1, 4916), got {selected_features.shape}",
        )

        # 2. Test predict_proba
        proba = self.pipeline_direct.predict_proba(dummy_zeros)
        self.assertEqual(
            proba.shape,
            (1, 2),
            f"Probability shape should be (1, 2), got {proba.shape}",
        )

        # 3. Check values are finite
        self.assertTrue(
            np.all(np.isfinite(proba)),
            f"Probabilities contain NaN or Inf values: {proba}",
        )

        # 4. Check probabilities are within [0, 1]
        self.assertTrue(
            np.all((proba >= 0.0) & (proba <= 1.0)),
            f"Probabilities out of bounds [0, 1]: {proba}",
        )

        # 5. Check sum of probabilities equals ~1.0
        proba_sum = float(np.sum(proba))
        self.assertAlmostEqual(
            proba_sum,
            1.0,
            places=5,
            msg=f"Sum of probabilities {proba_sum} is not close to 1.0",
        )

        # 6. Test predict() helper
        preds, labels = self.pipeline_direct.predict(dummy_zeros)
        self.assertEqual(len(preds), 1)
        self.assertIn(preds[0], [0, 1])
        self.assertIn(labels[0], ["Benign", "Malware"])

    def test_dummy_data_flow_batch_samples(self) -> None:
        """Verify batch dummy data (5, 24583) with sparse active features."""
        np.random.seed(42)
        dummy_batch = np.zeros((5, 24583), dtype=np.float32)
        for i in range(5):
            active_indices = np.random.choice(24583, size=100, replace=False)
            dummy_batch[i, active_indices] = 1.0

        proba = self.pipeline_direct.predict_proba(dummy_batch)
        self.assertEqual(proba.shape, (5, 2))
        self.assertTrue(np.all(np.isfinite(proba)))

        # Check all rows sum to 1.0
        row_sums = np.sum(proba, axis=1)
        for i, s in enumerate(row_sums):
            self.assertAlmostEqual(
                float(s),
                1.0,
                places=5,
                msg=f"Batch sample {i} probability sum {s} != 1.0",
            )

        preds, labels = self.pipeline_direct.predict(dummy_batch)
        self.assertEqual(len(preds), 5)
        for p, label in zip(preds, labels):
            self.assertIn(p, [0, 1])
            self.assertIn(label, ["Benign", "Malware"])


if __name__ == "__main__":
    unittest.main(verbosity=2)
