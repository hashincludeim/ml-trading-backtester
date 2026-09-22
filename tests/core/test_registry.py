from __future__ import annotations

import pytest
from sklearn.decomposition import PCA
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from stockml.config import ModelConfig
from stockml.models.registry import MODEL_REGISTRY, build_pipeline, get_spec, model_label


@pytest.mark.parametrize("name", list(MODEL_REGISTRY))
def test_every_model_builds_scaled_pipeline(name: str) -> None:
    pipe = build_pipeline(name, ModelConfig())
    assert isinstance(pipe, Pipeline)
    assert isinstance(pipe.steps[0][1], StandardScaler)
    assert pipe.steps[-1][0] == "model"
    params = pipe.named_steps["model"].get_params()
    if "random_state" in params:
        assert params["random_state"] == ModelConfig().random_state


def test_pca_is_optional() -> None:
    assert "pca" not in build_pipeline("logistic_regression").named_steps
    pipe = build_pipeline("logistic_regression", ModelConfig(pca_components=3))
    assert isinstance(pipe.named_steps["pca"], PCA)


def test_gradient_boosting_is_shallow() -> None:
    depth = build_pipeline("gradient_boosting").named_steps["model"].max_depth
    assert 2 <= depth <= 5
    assert all(2 <= d <= 5 for d in get_spec("gradient_boosting").param_grid["model__max_depth"])


def test_unknown_model_raises() -> None:
    with pytest.raises(KeyError, match="Unknown model"):
        get_spec("linear_regression")
    assert model_label("mystery") == "mystery"
