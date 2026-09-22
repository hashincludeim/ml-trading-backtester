"""Save and load fitted pipelines with joblib."""

from __future__ import annotations

import logging
from pathlib import Path

import joblib
from sklearn.pipeline import Pipeline

logger = logging.getLogger(__name__)


def save_model(pipeline: Pipeline, path: Path) -> Path:
    """Persist a fitted pipeline, creating parent directories as needed.

    Returns:
        The path written.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(pipeline, path)
    logger.info("Saved model to %s", path)
    return path


def load_model(path: Path) -> Pipeline:
    """Load a pipeline saved by :func:`save_model`.

    Raises:
        TypeError: If the file does not contain an sklearn ``Pipeline``.
    """
    obj = joblib.load(Path(path))
    if not isinstance(obj, Pipeline):
        raise TypeError(f"{path} does not contain a sklearn Pipeline")
    return obj
