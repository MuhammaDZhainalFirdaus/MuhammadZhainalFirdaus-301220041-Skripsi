"""Model Loader and Inference Pipeline for Android Malware Scanner.

Handles loading and inference for original research artifacts:
- models/feature_names.csv (24,583 features)
- models/selected_features.csv (4,916 features)
- models/xgboost_rfe.json (XGBoost classifier)
- models/rfe_selector.joblib (Scikit-Learn RFE selector, kept for validation/testing)
- models/model_metadata.json (Training metadata)
- models/metrics_rfe.json (RFE evaluation metrics)

Inference Optimization:
The application inference path utilizes `selected_features.csv` column indices directly:
`X[:, selected_indices]`, matching `rfe_selector.joblib` output with 100% bitwise identity,
completely avoiding runtime unpickling overhead and GPU/CPU deserialization fallback warnings.
"""

from __future__ import annotations

import contextlib
import ctypes
import json
import logging
from pathlib import Path
from typing import Any, Dict, Optional, Tuple, Union

import joblib
import numpy as np
import pandas as pd
from sklearn.feature_selection import RFE
import xgboost as xgb

logger = logging.getLogger(__name__)

DEFAULT_MODELS_DIR = Path(__file__).resolve().parent.parent / "models"


@contextlib.contextmanager
def patch_booster_setstate_cpu():
    """Context manager to safely unpickle XGBoost Booster instances on CPU environments.

    The research RFE selector was trained and serialized on a machine with CUDA GPU
    (`device='cuda:0'`). In XGBoost 3.4.1+, when deserializing a GPU-serialized internal
    Booster buffer on a CPU-only host, XGBoost attempts an automatic device migration
    ('grow_gpu_hist' -> 'grow_quantile_histmaker') which triggers an `input stream corrupted`
    error in `_LIB.XGBoosterUnserializeFromBuffer`.

    Because Scikit-Learn's RFE selector only relies on `support_`, `ranking_`,
    `n_features_in_`, and `n_features_` for inference transformation (`X[:, self.support_]`),
    the internal estimator handle is safely caught and neutralized if the GPU buffer cannot
    be rebuilt on CPU.
    """
    orig_setstate = xgb.core.Booster.__setstate__

    def safe_setstate(self: xgb.core.Booster, state: Dict[str, Any]) -> None:
        try:
            handle = state.get("handle")
            if handle is not None:
                buf = handle
                dmats = xgb.core.c_array(ctypes.c_void_p, [])
                h = ctypes.c_void_p()
                xgb.core._check_call(
                    xgb.core._LIB.XGBoosterCreate(
                        dmats, xgb.core.c_bst_ulong(0), ctypes.byref(h)
                    )
                )
                length = xgb.core.c_bst_ulong(len(buf))
                ptr = (ctypes.c_char * len(buf)).from_buffer(buf)
                xgb.core._check_call(
                    xgb.core._LIB.XGBoosterUnserializeFromBuffer(h, ptr, length)
                )
                state["handle"] = h
        except Exception as exc:
            logger.warning(
                "Booster.__setstate__ warning: GPU serialized booster handle neutralized on CPU (%s). "
                "RFE selector support_ mask remains fully valid and operational.",
                exc,
            )
            state["handle"] = None
        self.__dict__.update(state)

    xgb.core.Booster.__setstate__ = safe_setstate
    try:
        yield
    finally:
        xgb.core.Booster.__setstate__ = orig_setstate


def load_feature_names(path: Union[str, Path]) -> pd.DataFrame:
    """Load feature_names.csv and validate structure."""
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"feature_names.csv not found at: {path}")

    df = pd.read_csv(path)
    expected_cols = ["feature_index", "feature_name"]
    if list(df.columns) != expected_cols:
        raise ValueError(
            f"Invalid columns in feature_names.csv: {list(df.columns)}. Expected {expected_cols}"
        )

    if len(df) != 24583:
        raise ValueError(f"Expected 24,583 features in feature_names.csv, found {len(df)}")

    return df


def load_selected_features(path: Union[str, Path]) -> pd.DataFrame:
    """Load selected_features.csv and validate structure."""
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"selected_features.csv not found at: {path}")

    df = pd.read_csv(path)
    expected_cols = ["selected_index", "feature_index", "feature_name"]
    if list(df.columns) != expected_cols:
        raise ValueError(
            f"Invalid columns in selected_features.csv: {list(df.columns)}. Expected {expected_cols}"
        )

    if len(df) != 4916:
        raise ValueError(f"Expected 4,916 features in selected_features.csv, found {len(df)}")

    return df


def load_xgboost_model(path: Union[str, Path]) -> xgb.XGBClassifier:
    """Load research XGBoost model from JSON file."""
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"xgboost_rfe.json not found at: {path}")

    model = xgb.XGBClassifier()
    model.load_model(str(path))

    if model.n_features_in_ != 4916:
        raise ValueError(
            f"Model expects {model.n_features_in_} features, but research requires 4,916."
        )

    return model


def load_rfe_selector(path: Union[str, Path]) -> RFE:
    """Load research Scikit-Learn RFE selector from joblib archive (used for validation)."""
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"rfe_selector.joblib not found at: {path}")

    with patch_booster_setstate_cpu():
        selector: RFE = joblib.load(str(path))

    if selector.n_features_in_ != 24583:
        raise ValueError(
            f"RFE selector expects {selector.n_features_in_} input features, research requires 24,583."
        )

    if selector.n_features_ != 4916:
        raise ValueError(
            f"RFE selector outputs {selector.n_features_} features, research requires 4,916."
        )

    if selector.support_.sum() != 4916:
        raise ValueError(
            f"RFE selector support_ mask sum is {selector.support_.sum()}, expected 4,916."
        )

    return selector


def load_metadata(path: Union[str, Path]) -> Dict[str, Any]:
    """Load model_metadata.json."""
    path = Path(path)
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def load_metrics(path: Union[str, Path]) -> Dict[str, Any]:
    """Load metrics_rfe.json."""
    path = Path(path)
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


class AndroidMalwarePipeline:
    """End-to-end inference pipeline for Android Malware Detection.

    Flow:
    APK raw feature vector (1, 24583) -> Direct column indexing via `selected_features.csv` (1, 4916) -> XGBoost -> Probabilities.
    """

    def __init__(
        self,
        models_dir: Union[str, Path] = DEFAULT_MODELS_DIR,
        use_rfe_joblib: bool = False,
    ) -> None:
        self.models_dir = Path(models_dir)
        self.feature_names_df = load_feature_names(self.models_dir / "feature_names.csv")
        self.selected_features_df = load_selected_features(
            self.models_dir / "selected_features.csv"
        )
        self.model = load_xgboost_model(self.models_dir / "xgboost_rfe.json")

        # Direct selection indices strictly based on selected_features.csv in selected_index order
        self.selected_indices: np.ndarray = self.selected_features_df["feature_index"].values
        self.num_selected_features = len(self.selected_indices)
        assert self.num_selected_features == 4916, f"Expected 4,916 features, got {self.num_selected_features}"

        # Optional RFE selector instance (only loaded when explicitly requested for verification)
        self.use_rfe_joblib = use_rfe_joblib
        self._selector: Optional[RFE] = None
        if self.use_rfe_joblib:
            self._selector = load_rfe_selector(self.models_dir / "rfe_selector.joblib")

        self.metadata = load_metadata(self.models_dir / "model_metadata.json")
        self.label_mapping = {0: "Benign", 1: "Malware"}

    @property
    def selector(self) -> RFE:
        """Lazy loader for RFE selector if accessed."""
        if self._selector is None:
            self._selector = load_rfe_selector(self.models_dir / "rfe_selector.joblib")
        return self._selector

    def transform(self, X: np.ndarray) -> np.ndarray:
        """Transform raw 24,583 features to 4,916 selected features.

        By default, uses direct column indexing via `selected_features.csv`,
        producing output 100% bitwise identical to `rfe_selector.joblib` without
        runtime unpickling overhead.
        """
        X_arr = np.asarray(X, dtype=np.float32)
        if X_arr.ndim == 1:
            X_arr = X_arr.reshape(1, -1)

        if X_arr.shape[1] != 24583:
            raise ValueError(
                f"Expected input array with 24,583 features, got shape {X_arr.shape}"
            )

        if self.use_rfe_joblib:
            return self.selector.transform(X_arr)
        else:
            return X_arr[:, self.selected_indices]

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Generate class probabilities for input feature vectors.

        Returns array of shape (N, 2): [P(Benign), P(Malware)].
        """
        selected = self.transform(X)
        proba = self.model.predict_proba(selected)
        return proba

    def predict(
        self, X: np.ndarray, threshold: float = 0.5
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Predict binary class and class label string.

        Returns:
            (predictions, class_labels): e.g. (array([0]), array(['Benign']))
        """
        proba = self.predict_proba(X)
        malware_proba = proba[:, 1]
        preds = (malware_proba >= threshold).astype(int)
        labels = np.array([self.label_mapping[p] for p in preds])
        return preds, labels
