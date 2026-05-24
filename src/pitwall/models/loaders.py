"""Cached loaders for the saved degradation + undercut models.

`train_and_validate.py` dumps these to `data/processed/*.joblib`. The app
imports these helpers so every page can load the same artifacts without
re-fitting.
"""

from __future__ import annotations

from pathlib import Path

import joblib

from pitwall.config import PROCESSED_DIR
from pitwall.models.degradation_curve import DegradationModel
from pitwall.models.undercut_classifier import UndercutClassifier

DEG_MODEL_PATH = PROCESSED_DIR / "degradation_model.joblib"
UNDERCUT_MODEL_PATH = PROCESSED_DIR / "undercut_classifier.joblib"


def load_degradation_model(path: Path | str = DEG_MODEL_PATH) -> DegradationModel | None:
    """Return the trained DegradationModel, or None if it hasn't been built yet."""
    p = Path(path)
    if not p.exists():
        return None
    try:
        return joblib.load(p)
    except Exception:
        return None


def load_undercut_classifier(path: Path | str = UNDERCUT_MODEL_PATH) -> UndercutClassifier | None:
    """Return the trained UndercutClassifier, or None if it hasn't been built yet."""
    p = Path(path)
    if not p.exists():
        return None
    try:
        return joblib.load(p)
    except Exception:
        return None
