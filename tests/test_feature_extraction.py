"""Unit tests for APK feature extraction schema, archive validation, and upload handling.

Verifies:
1. APKFeatureExtractor correctly maps 24,583 features (166 permissions, 24,417 API calls)
2. Normalization logic from AndroidManifest permission strings to research format
3. Normalization logic from Dalvik bytecode method descriptors to research format
4. Validation and error handling for APK files (missing file, missing AndroidManifest, corrupt archive)
5. Support for APK Bundle / APKM containers (extracting base.apk automatically)
6. Streamlit-style uploaded file simulation (verifying seek(0), byte integrity, and SHA-256 invariance with scan_* prefix)
"""

from __future__ import annotations

import hashlib
import io
from pathlib import Path
import sys
import tempfile
import unittest
import zipfile

import numpy as np

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.feature_extractor import APKFeatureExtractor, inspect_archive


class TestFeatureExtractorMapping(unittest.TestCase):
    """Test schema normalization and indexing for APKFeatureExtractor."""

    @classmethod
    def setUpClass(cls) -> None:
        cls.extractor = APKFeatureExtractor()

    def test_feature_counts(self) -> None:
        """Verify total feature count and distribution between permissions and API calls."""
        self.assertEqual(self.extractor.num_total_features, 24583)
        self.assertEqual(len(self.extractor.permission_features), 166)
        self.assertEqual(len(self.extractor.apicall_features), 24417)

    def test_permission_normalization_and_indices(self) -> None:
        """Verify permissions are resolved by stripping package prefix to short uppercase name."""
        test_cases = [
            ("android.permission.WAKE_LOCK", "Permission::WAKE_LOCK", 0),
            ("android.permission.WRITE_EXTERNAL_STORAGE", "Permission::WRITE_EXTERNAL_STORAGE", 1),
            ("android.permission.ACCESS_NETWORK_STATE", "Permission::ACCESS_NETWORK_STATE", 2),
            ("android.permission.WRITE_SETTINGS", "Permission::WRITE_SETTINGS", 3),
            ("android.permission.INTERNET", "Permission::INTERNET", 4),
        ]
        for raw_perm, expected_key, expected_idx in test_cases:
            short_perm = raw_perm.split(".")[-1].strip()
            feature_key = f"Permission::{short_perm}"
            self.assertEqual(feature_key, expected_key)
            self.assertIn(feature_key, self.extractor.feature_to_idx)
            self.assertEqual(
                self.extractor.feature_to_idx[feature_key],
                expected_idx,
                f"Index mismatch for {feature_key}",
            )

    def test_apicall_normalization_and_indices(self) -> None:
        """Verify Dalvik method calls are resolved by stripping trailing semicolon and appending ()."""
        test_cases = [
            ("Landroid/content/Intent;", "toUri", "APICall::Landroid/content/Intent.toUri()", 5),
            ("Landroid/view/View;", "setTag", "APICall::Landroid/view/View.setTag()", 6),
            ("Landroid/util/Xml;", "newSerializer", "APICall::Landroid/util/Xml.newSerializer()", 7),
            (
                "Landroid/content/pm/PackageManager;",
                "queryIntentServices",
                "APICall::Landroid/content/pm/PackageManager.queryIntentServices()",
                8,
            ),
        ]
        for raw_cls, raw_method, expected_sig, expected_idx in test_cases:
            cls_name = raw_cls.rstrip(";")
            sig = f"APICall::{cls_name}.{raw_method}()"
            self.assertEqual(sig, expected_sig)
            self.assertIn(sig, self.extractor.feature_to_idx)
            self.assertEqual(
                self.extractor.feature_to_idx[sig],
                expected_idx,
                f"Index mismatch for {sig}",
            )

    def test_file_not_found_handling(self) -> None:
        """Verify FileNotFoundError is raised for non-existent APK path."""
        with self.assertRaises(FileNotFoundError):
            self.extractor.extract_features_from_apk("non_existent_app.apk")

    def test_inspect_archive_diagnostics(self) -> None:
        """Verify inspect_archive extracts correct metadata across all calling conventions."""
        with tempfile.NamedTemporaryFile(suffix=".apk", delete=False) as tmp:
            tmp_path = Path(tmp.name)
            with zipfile.ZipFile(tmp_path, "w") as z:
                z.writestr("AndroidManifest.xml", b"<dummy_xml>")
                z.writestr("classes.dex", b"dummy_dex")
                z.writestr("classes2.dex", b"dummy_dex_2")

        try:
            # 1. Test call via extractor instance method
            diag_instance = self.extractor.inspect_archive(tmp_path)
            # 2. Test call via class staticmethod
            diag_class = APKFeatureExtractor.inspect_archive(tmp_path)
            # 3. Test call via standalone module function
            diag_func = inspect_archive(tmp_path)
            # 4. Test call via app.py convention with hasattr fallback
            diag_app = (
                self.extractor.inspect_archive(tmp_path)
                if hasattr(self.extractor, "inspect_archive")
                else inspect_archive(tmp_path)
            )

            # All calling conventions must return identical diagnostic dictionaries
            self.assertEqual(diag_instance, diag_class)
            self.assertEqual(diag_instance, diag_func)
            self.assertEqual(diag_instance, diag_app)

            self.assertTrue(diag_instance["is_valid_zip"])
            self.assertTrue(diag_instance["has_manifest"])
            self.assertEqual(diag_instance["manifest_entry"], "AndroidManifest.xml")
            self.assertFalse(diag_instance["has_base_apk"])
            self.assertEqual(len(diag_instance["dex_entries"]), 2)
            self.assertIn("classes.dex", diag_instance["dex_entries"])
            self.assertIn("classes2.dex", diag_instance["dex_entries"])
        finally:
            if tmp_path.exists():
                tmp_path.unlink()

        # Test bundle diagnostics with inspect_archive
        with tempfile.NamedTemporaryFile(suffix=".apkm", delete=False) as tmp_bundle:
            bundle_path = Path(tmp_bundle.name)
            with zipfile.ZipFile(bundle_path, "w") as bz:
                bz.writestr("base.apk", b"dummy_base_bytes")
                bz.writestr("split_config.arm64_v8a.apk", b"dummy_split_bytes")

        try:
            diag_bundle = self.extractor.inspect_archive(bundle_path)
            self.assertTrue(diag_bundle["is_valid_zip"])
            self.assertFalse(diag_bundle["has_manifest"])
            self.assertTrue(diag_bundle["has_base_apk"])
            self.assertEqual(diag_bundle["base_apk_entry"], "base.apk")
            self.assertIn("base.apk", diag_bundle["relevant_entries"])
        finally:
            if bundle_path.exists():
                bundle_path.unlink()

    def test_split_apk_missing_manifest_and_base_handling(self) -> None:
        """Verify ValueError is raised if APK archive does not contain AndroidManifest.xml nor base.apk."""
        with tempfile.NamedTemporaryFile(suffix=".apk", delete=False) as tmp:
            tmp_path = Path(tmp.name)
            with zipfile.ZipFile(tmp_path, "w") as z:
                z.writestr("resources.arsc", b"dummy_resources")

        try:
            with self.assertRaises(ValueError) as ctx:
                self.extractor.resolve_apk_target(tmp_path)
            self.assertIn("tidak memuat 'AndroidManifest.xml'", str(ctx.exception))
            self.assertIn("tidak memuat 'base.apk'", str(ctx.exception))
        finally:
            if tmp_path.exists():
                tmp_path.unlink()

    def test_apk_bundle_apkm_resolution(self) -> None:
        """Verify that an APK Bundle containing base.apk is automatically detected and unpacked."""
        # Create an inner valid base.apk
        inner_buf = io.BytesIO()
        with zipfile.ZipFile(inner_buf, "w") as bz:
            bz.writestr("AndroidManifest.xml", b"<dummy_manifest>")
            bz.writestr("classes.dex", b"dummy_dex_bytes")
        inner_bytes = inner_buf.getvalue()

        # Create outer bundle archive (like an APKMirror / APKM bundle)
        with tempfile.NamedTemporaryFile(suffix=".apk", delete=False) as tmp:
            tmp_path = Path(tmp.name)
            with zipfile.ZipFile(tmp_path, "w") as z:
                z.writestr("info.json", b"{}")
                z.writestr("base.apk", inner_bytes)
                z.writestr("split_config.arm64.apk", b"dummy_split")

        try:
            target_apk, temp_extracted, diag = self.extractor.resolve_apk_target(tmp_path)
            self.assertTrue(diag["has_base_apk"])
            self.assertEqual(diag["base_apk_entry"], "base.apk")
            self.assertIsNotNone(temp_extracted)
            self.assertTrue(target_apk.is_file())

            # Verify target apk has AndroidManifest.xml and classes.dex
            with zipfile.ZipFile(target_apk, "r") as tz:
                self.assertIn("AndroidManifest.xml", tz.namelist())
                self.assertIn("classes.dex", tz.namelist())

            # Cleanup
            if temp_extracted is not None and temp_extracted.exists():
                temp_extracted.unlink()
        finally:
            if tmp_path.exists():
                tmp_path.unlink()

    def test_corrupt_apk_handling(self) -> None:
        """Verify ValueError is raised if APK file is corrupted or not a valid ZIP."""
        with tempfile.NamedTemporaryFile(suffix=".apk", delete=False) as tmp:
            tmp_path = Path(tmp.name)
            tmp.write(b"NOT_A_VALID_ZIP_HEADER_CORRUPTED_DATA")

        try:
            with self.assertRaises(ValueError) as ctx:
                self.extractor.resolve_apk_target(tmp_path)
            self.assertIn("rusak atau bukan format", str(ctx.exception))
        finally:
            if tmp_path.exists():
                tmp_path.unlink()

    def test_uploaded_file_seek_and_byte_integrity(self) -> None:
        """Simulate Streamlit file_uploader object and verify byte invariance and SHA-256 match."""
        # Create dummy sample APK bytes
        apk_buf = io.BytesIO()
        with zipfile.ZipFile(apk_buf, "w") as z:
            z.writestr("AndroidManifest.xml", b"<valid_manifest_test>")
            z.writestr("classes.dex", b"valid_dex_code")
        raw_bytes = apk_buf.getvalue()
        original_hash = hashlib.sha256(raw_bytes).hexdigest()

        # Simulate Streamlit UploadedFile with pointer moved (e.g. read before)
        simulated_uploader = io.BytesIO(raw_bytes)
        simulated_uploader.read(50)  # Move pointer away from 0

        # Application pattern: ensure seek(0) is called before saving
        simulated_uploader.seek(0)
        read_bytes = simulated_uploader.read()

        self.assertEqual(len(read_bytes), len(raw_bytes))
        self.assertEqual(hashlib.sha256(read_bytes).hexdigest(), original_hash)

        # Write to temporary file with scan_<timestamp>_ prefix
        temp_dir = Path(tempfile.gettempdir())
        temp_file = temp_dir / "scan_1792767597_test_app.apk"
        try:
            with open(temp_file, "wb") as f:
                f.write(read_bytes)
                f.flush()

            # Verify on-disk file
            with open(temp_file, "rb") as f:
                written_bytes = f.read()

            self.assertEqual(written_bytes, raw_bytes)
            self.assertEqual(hashlib.sha256(written_bytes).hexdigest(), original_hash)

            # Verify extractor validates it cleanly
            diag = self.extractor.validate_apk_file(temp_file)
            self.assertTrue(diag["has_manifest"])
            self.assertEqual(diag["dex_entries"], ["classes.dex"])
        finally:
            if temp_file.exists():
                temp_file.unlink()

    def test_inspect_archive_attribute_fallback(self) -> None:
        """Verify fallback logic when an object without inspect_archive attribute is encountered."""
        class StaleDummyExtractor:
            pass

        stale_extractor = StaleDummyExtractor()
        with tempfile.NamedTemporaryFile(suffix=".apk", delete=False) as tmp:
            tmp_path = Path(tmp.name)
            with zipfile.ZipFile(tmp_path, "w") as z:
                z.writestr("AndroidManifest.xml", b"<dummy_xml>")

        try:
            # Replicate app.py pattern:
            diag = (
                stale_extractor.inspect_archive(tmp_path)
                if hasattr(stale_extractor, "inspect_archive")
                else inspect_archive(tmp_path)
            )
            self.assertTrue(diag["has_manifest"])
            self.assertTrue(diag["is_valid_zip"])
        finally:
            if tmp_path.exists():
                tmp_path.unlink()


if __name__ == "__main__":
    unittest.main(verbosity=2)
