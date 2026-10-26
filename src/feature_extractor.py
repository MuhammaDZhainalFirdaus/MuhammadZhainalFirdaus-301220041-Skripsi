"""APK Feature Extractor Module for Android Malware Detection.

Extracts static features from Android APK binaries using Androguard 4.1.4,
matching the exact schema of the MH-100K research dataset:
- Permission features: `Permission::<SHORT_PERMISSION_NAME>` (e.g. `Permission::WAKE_LOCK`)
- API Call features: `APICall::<CLASS_NAME>.<METHOD_NAME>()` (e.g. `APICall::Landroid/content/Intent.toUri()`)

Supports both:
1. Standalone standard APK files (containing `AndroidManifest.xml` and `classes.dex` at root).
2. APK bundle archives / APKM / XAPK (containing `base.apk` which is automatically resolved).
"""

from __future__ import annotations

import logging
from pathlib import Path
import tempfile
import time
from typing import Any, Dict, List, Optional, Set, Tuple, Union
import zipfile

from loguru import logger as loguru_logger
import numpy as np
import pandas as pd

# Disable verbose debug logging from androguard loguru handlers
loguru_logger.disable("androguard")

from .model_loader import DEFAULT_MODELS_DIR, load_feature_names

logger = logging.getLogger(__name__)


def inspect_archive(file_path: Union[str, Path]) -> Dict[str, Any]:
    """Inspect the ZIP structure of an APK or bundle and return safe diagnostics."""
    apk_file = Path(file_path)
    diag: Dict[str, Any] = {
        "file_name": apk_file.name,
        "file_size_bytes": 0,
        "is_valid_zip": False,
        "has_manifest": False,
        "manifest_entry": None,
        "dex_entries": [],
        "has_base_apk": False,
        "base_apk_entry": None,
        "relevant_entries": [],
    }

    if not apk_file.is_file():
        return diag

    diag["file_size_bytes"] = apk_file.stat().st_size
    if diag["file_size_bytes"] == 0:
        return diag

    try:
        with zipfile.ZipFile(apk_file, "r") as z:
            diag["is_valid_zip"] = True
            namelist = z.namelist()

            for name in namelist:
                clean = name.lstrip("./\\")
                # Check manifest at root
                if clean.lower() == "androidmanifest.xml":
                    diag["has_manifest"] = True
                    diag["manifest_entry"] = name
                    if name not in diag["relevant_entries"]:
                        diag["relevant_entries"].append(name)
                # Check Dalvik executable
                elif clean.lower().endswith(".dex"):
                    diag["dex_entries"].append(name)
                    if len(diag["relevant_entries"]) < 10 and name not in diag["relevant_entries"]:
                        diag["relevant_entries"].append(name)
                # Check for base.apk inside bundle (APKM / XAPK / App Bundle)
                elif clean.lower() in ["base.apk", "base-master.apk"]:
                    diag["has_base_apk"] = True
                    diag["base_apk_entry"] = name
                    if name not in diag["relevant_entries"]:
                        diag["relevant_entries"].append(name)
                # Check for split APKs
                elif clean.lower().endswith(".apk"):
                    if len(diag["relevant_entries"]) < 10 and name not in diag["relevant_entries"]:
                        diag["relevant_entries"].append(name)
    except (zipfile.BadZipFile, OSError):
        diag["is_valid_zip"] = False

    return diag


class APKFeatureExtractor:
    """Extracts static features from Android APKs matching the 24,583 MH-100K research feature space."""

    inspect_archive = staticmethod(inspect_archive)

    def __init__(
        self,
        feature_names_path: Union[str, Path] = DEFAULT_MODELS_DIR / "feature_names.csv",
    ) -> None:
        self.feature_names_path = Path(feature_names_path)
        self.feature_names_df = load_feature_names(self.feature_names_path)

        # Ensure inspect_archive is explicitly available on instance
        self.inspect_archive = inspect_archive

        # Build reverse index: feature_name -> feature_index (0..24582)
        self.feature_to_idx: Dict[str, int] = {
            name: idx for idx, name in zip(
                self.feature_names_df["feature_index"],
                self.feature_names_df["feature_name"],
            )
        }
        self.num_total_features = len(self.feature_to_idx)
        assert self.num_total_features == 24583, f"Expected 24583 features, got {self.num_total_features}"

        # Segment features by type for quick analysis
        self.permission_features: Set[str] = {
            f for f in self.feature_to_idx if f.startswith("Permission::")
        }
        self.apicall_features: Set[str] = {
            f for f in self.feature_to_idx if f.startswith("APICall::")
        }

        logger.info(
            "APKFeatureExtractor initialized with %d total features (%d permissions, %d API calls)",
            self.num_total_features,
            len(self.permission_features),
            len(self.apicall_features),
        )

    def resolve_apk_target(
        self, apk_path: Union[str, Path]
    ) -> Tuple[Path, Optional[Path], Dict[str, Any]]:
        """Validate and resolve the target APK to analyze.

        If `apk_path` is a standalone APK, it is validated directly.
        If `apk_path` is an APK bundle (APKM/XAPK/Split APK) containing `base.apk`,
        `base.apk` is safely extracted to a temporary file for analysis.

        Returns:
            (target_apk_path, temp_extracted_path, diagnostics)
        """
        apk_file = Path(apk_path)
        if not apk_file.is_file():
            raise FileNotFoundError(f"File APK tidak ditemukan: {apk_file}")

        diag = self.inspect_archive(apk_file)

        if diag["file_size_bytes"] == 0:
            raise ValueError(f"File APK kosong (0 bytes): '{apk_file.name}'.")

        if not diag["is_valid_zip"]:
            raise ValueError(
                f"File '{apk_file.name}' ({diag['file_size_bytes']:,} bytes) rusak "
                "atau bukan format arsip ZIP/APK yang valid."
            )

        # Case 1: Standalone APK with AndroidManifest.xml at root
        if diag["has_manifest"]:
            if not diag["dex_entries"]:
                raise ValueError(
                    f"File '{apk_file.name}' memuat 'AndroidManifest.xml' namun tidak memuat berkas bytecode DEX (*.dex). "
                    "APK tidak dapat dianalisis pemanggilan metodenya."
                )
            return apk_file, None, diag

        # Case 2: APK Bundle (APKM / XAPK / Split Bundle) containing base.apk
        if diag["has_base_apk"] and diag["base_apk_entry"]:
            logger.info(
                "Detected APK Bundle container (%s). Unpacking '%s' for static analysis...",
                apk_file.name,
                diag["base_apk_entry"],
            )
            # Unpack base.apk to a secure temporary file
            with zipfile.ZipFile(apk_file, "r") as z:
                base_bytes = z.read(diag["base_apk_entry"])

            temp_base = tempfile.NamedTemporaryFile(
                prefix="unpacked_base_", suffix=".apk", delete=False
            )
            try:
                temp_base.write(base_bytes)
                temp_base.flush()
            finally:
                temp_base.close()

            temp_base_path = Path(temp_base.name)

            # Validate the extracted base.apk
            base_diag = self.inspect_archive(temp_base_path)
            if not base_diag["has_manifest"] or not base_diag["dex_entries"]:
                if temp_base_path.exists():
                    temp_base_path.unlink()
                raise ValueError(
                    f"Bundel '{apk_file.name}' memuat '{diag['base_apk_entry']}', namun berkas tersebut "
                    "tidak memuat 'AndroidManifest.xml' atau berkas DEX yang valid."
                )

            diag["unpacked_base_apk_size"] = temp_base_path.stat().st_size
            diag["base_diag"] = base_diag
            return temp_base_path, temp_base_path, diag

        # Case 3: Neither AndroidManifest.xml nor base.apk found
        safe_entries = diag["relevant_entries"] or ["(tidak ada entri .apk, .xml, atau .dex)"]
        raise ValueError(
            f"File '{apk_file.name}' ({diag['file_size_bytes']:,} bytes) tidak memuat 'AndroidManifest.xml' "
            f"dan tidak memuat 'base.apk'. Entri terdeteksi: {safe_entries}. "
            "Pastikan berkas yang diunggah adalah APK Android standar (*.apk) atau paket APKM/XAPK yang memuat base.apk."
        )

    def validate_apk_file(self, apk_file: Union[str, Path]) -> Dict[str, Any]:
        """Validate APK archive integrity and return diagnostics."""
        target, temp_extracted, diag = self.resolve_apk_target(apk_file)
        if temp_extracted is not None and temp_extracted.exists():
            temp_extracted.unlink()
        return diag

    def extract_features_from_apk(
        self, apk_path: Union[str, Path]
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        """Extract permissions and Dalvik API calls from an APK file or bundle.

        Args:
            apk_path: Path to the .apk or bundle file

        Returns:
            feature_vector: np.ndarray of shape (1, 24583) with binary flags (0.0 or 1.0)
            extraction_report: Dictionary containing extraction statistics and matched features
        """
        from androguard.misc import AnalyzeAPK

        original_file = Path(apk_path)
        target_apk, temp_extracted, diag = self.resolve_apk_target(original_file)

        start_time = time.time()
        logger.info("Starting Androguard 4.1.4 analysis on %s", target_apk.name)

        try:
            # Run static analysis through Androguard
            apk_obj, dex_list, dx = AnalyzeAPK(str(target_apk))

            extracted_permissions: List[str] = []
            matched_permissions: List[str] = []

            extracted_api_calls: Set[str] = set()
            matched_api_calls: List[str] = []

            # 1. Extract Permissions from AndroidManifest.xml
            raw_permissions = apk_obj.get_permissions() or []
            for raw_perm in raw_permissions:
                extracted_permissions.append(raw_perm)
                # In research schema: android.permission.INTERNET -> Permission::INTERNET
                short_perm = raw_perm.split(".")[-1].strip()
                feature_key = f"Permission::{short_perm}"
                if feature_key in self.feature_to_idx:
                    matched_permissions.append(feature_key)

            # 2. Extract API Calls from Dalvik Executable bytecode
            if dx is not None:
                for method in dx.get_methods():
                    xrefs_from = method.get_xref_from()
                    if len(xrefs_from) > 0 or method.is_external():
                        cls_name = method.class_name.rstrip(";") if method.class_name else ""
                        m_name = method.name or ""
                        api_signature = f"APICall::{cls_name}.{m_name}()"

                        extracted_api_calls.add(api_signature)
                        if api_signature in self.feature_to_idx:
                            matched_api_calls.append(api_signature)

            # 3. Construct 24,583-dimensional binary feature vector
            vector = np.zeros(self.num_total_features, dtype=np.float32)

            for feat in matched_permissions:
                idx = self.feature_to_idx[feat]
                vector[idx] = 1.0

            for feat in matched_api_calls:
                idx = self.feature_to_idx[feat]
                vector[idx] = 1.0

            elapsed_time = time.time() - start_time

            report: Dict[str, Any] = {
                "apk_filename": original_file.name,
                "apk_size_bytes": original_file.stat().st_size,
                "is_bundle": diag["has_base_apk"],
                "base_apk_entry": diag.get("base_apk_entry"),
                "package_name": apk_obj.get_package() or "Tidak diketahui",
                "app_name": apk_obj.get_app_name() or "Tidak diketahui",
                "target_sdk_version": apk_obj.get_target_sdk_version() or "Tidak ditentukan",
                "min_sdk_version": apk_obj.get_min_sdk_version() or "Tidak ditentukan",
                "total_extracted_permissions": len(extracted_permissions),
                "matched_research_permissions": len(set(matched_permissions)),
                "total_extracted_api_calls": len(extracted_api_calls),
                "matched_research_api_calls": len(set(matched_api_calls)),
                "total_active_features": int(np.sum(vector)),
                "extraction_time_seconds": round(elapsed_time, 2),
                "matched_permissions_list": sorted(list(set(matched_permissions))),
                "matched_api_calls_sample": sorted(list(set(matched_api_calls)))[:50],
                "diagnostics": diag,
            }

            return vector.reshape(1, -1), report
        finally:
            # Clean up unpacked base.apk if created
            if temp_extracted is not None and temp_extracted.exists():
                try:
                    temp_extracted.unlink()
                except Exception as exc:
                    logger.warning("Could not unlink temporary base APK %s: %s", temp_extracted, exc)


__all__ = ["APKFeatureExtractor", "inspect_archive"]
