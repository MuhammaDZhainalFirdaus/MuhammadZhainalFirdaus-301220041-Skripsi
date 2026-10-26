"""Integration test for end-to-end static feature extraction and inference on a real APK.

Uses a real APK sample found in Downloads (if present):
C:\\Users\\HONGKONG\\Downloads\\WhatsApp-Plus-v3.0-AntiBan-www.ReXdl.com.apk
"""

from __future__ import annotations

from pathlib import Path
import sys
import unittest

import numpy as np

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.feature_extractor import APKFeatureExtractor
from src.model_loader import AndroidMalwarePipeline


class TestRealAPKExtraction(unittest.TestCase):
    """End-to-end integration test with a real APK file."""

    @classmethod
    def setUpClass(cls) -> None:
        cls.sample_apk = Path(
            r"C:\Users\HONGKONG\Downloads\WhatsApp-Plus-v3.0-AntiBan-www.ReXdl.com.apk"
        )
        cls.extractor = APKFeatureExtractor()
        cls.pipeline = AndroidMalwarePipeline()

    def test_real_apk_static_extraction_and_inference(self) -> None:
        """Run full extraction and inference pipeline on real APK."""
        if not self.sample_apk.is_file():
            self.skipTest(f"Sample APK not found at: {self.sample_apk}")

        # 1. Extraction
        vector, report = self.extractor.extract_features_from_apk(self.sample_apk)

        self.assertEqual(vector.shape, (1, 24583))
        self.assertGreater(report["total_extracted_permissions"], 0)
        self.assertGreater(report["matched_research_permissions"], 0)
        self.assertGreater(report["total_extracted_api_calls"], 0)
        self.assertGreater(report["matched_research_api_calls"], 0)
        self.assertGreater(report["total_active_features"], 0)

        # 2. RFE Direct Indexing Selection
        selected = self.pipeline.transform(vector)
        self.assertEqual(selected.shape, (1, 4916))

        # 3. XGBoost Inference
        proba = self.pipeline.predict_proba(vector)
        self.assertEqual(proba.shape, (1, 2))
        self.assertTrue(np.all(np.isfinite(proba)))
        self.assertAlmostEqual(float(np.sum(proba)), 1.0, places=5)

        preds, labels = self.pipeline.predict(vector)
        self.assertIn(preds[0], [0, 1])
        self.assertIn(labels[0], ["Benign", "Malware"])


if __name__ == "__main__":
    unittest.main(verbosity=2)
