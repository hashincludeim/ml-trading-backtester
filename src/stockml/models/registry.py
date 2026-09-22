"""Registry of classification pipelines.

Each entry builds a fresh, unfitted sklearn ``Pipeline`` of the form
``scaler -> [PCA] -> estimator`` so that all preprocessing is fitted inside each training fold.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

from sklearn.base import BaseEstimator
from sklearn.decomposition import PCA
from sklearn.ensemble import (
    BaggingClassifier,
    ExtraTreesClassifier,
    GradientBoostingClassifier,
    RandomForestClassifier,
)
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC, LinearSVC

from stockml.config import ModelConfig

EstimatorFactory = Callable[[ModelConfig], BaseEstimator]


@dataclass(frozen=True)
class ModelSpec:
    """Metadata and factory for one registered model.

    Attributes:
        name: Registry key (snake_case).
        label: Human-readable name for charts and tables.
        make_estimator: Builds the final estimator from a ``ModelConfig``.
        param_grid: Small grid (keys prefixed with ``model__``) searched with time-series CV.
        pca_components: PCA components always applied for this model (overrides config).
    """

    name: str
    label: str
    make_estimator: EstimatorFactory
    param_grid: dict[str, list[Any]] = field(default_factory=dict)
    pca_components: int | None = None


def _logistic(cfg: ModelConfig) -> BaseEstimator:
    return LogisticRegression(max_iter=2000, random_state=cfg.random_state)


def _svm_rbf(cfg: ModelConfig) -> BaseEstimator:
    return SVC(kernel="rbf", C=1.0, gamma="scale", random_state=cfg.random_state)


def _linear_svc(cfg: ModelConfig) -> BaseEstimator:
    return LinearSVC(C=0.1, max_iter=10_000, random_state=cfg.random_state)


def _random_forest(cfg: ModelConfig) -> BaseEstimator:
    return RandomForestClassifier(
        n_estimators=300,
        max_depth=6,
        min_samples_leaf=20,
        random_state=cfg.random_state,
        n_jobs=cfg.n_jobs,
    )


def _extra_trees(cfg: ModelConfig) -> BaseEstimator:
    return ExtraTreesClassifier(
        n_estimators=300,
        max_depth=6,
        min_samples_leaf=20,
        random_state=cfg.random_state,
        n_jobs=cfg.n_jobs,
    )


def _bagging_knn(cfg: ModelConfig) -> BaseEstimator:
    return BaggingClassifier(
        KNeighborsClassifier(n_neighbors=25),
        n_estimators=20,
        max_samples=0.5,
        random_state=cfg.random_state,
        n_jobs=cfg.n_jobs,
    )


def _gradient_boosting(cfg: ModelConfig) -> BaseEstimator:
    return GradientBoostingClassifier(
        n_estimators=200,
        max_depth=3,
        learning_rate=0.03,
        subsample=0.7,
        random_state=cfg.random_state,
    )


MODEL_REGISTRY: dict[str, ModelSpec] = {
    spec.name: spec
    for spec in (
        ModelSpec(
            "logistic_regression",
            "Logistic Regression",
            _logistic,
            param_grid={"model__C": [0.01, 0.1, 1.0]},
        ),
        ModelSpec("svm_rbf", "SVM (RBF)", _svm_rbf, param_grid={"model__C": [0.3, 1.0, 3.0]}),
        ModelSpec("linear_svc", "Linear SVC", _linear_svc),
        ModelSpec(
            "random_forest",
            "Random Forest",
            _random_forest,
            param_grid={"model__max_depth": [3, 6]},
        ),
        ModelSpec("extra_trees", "Extra Trees", _extra_trees),
        ModelSpec("bagging_knn", "Bagging (KNN)", _bagging_knn, pca_components=5),
        ModelSpec(
            "gradient_boosting",
            "Gradient Boosting",
            _gradient_boosting,
            param_grid={"model__max_depth": [2, 3, 5]},
        ),
    )
}


def get_spec(name: str) -> ModelSpec:
    """Look up a registered model.

    Raises:
        KeyError: With the list of valid names if ``name`` is unknown.
    """
    try:
        return MODEL_REGISTRY[name]
    except KeyError as exc:
        raise KeyError(f"Unknown model {name!r}; choose from {sorted(MODEL_REGISTRY)}") from exc


def model_label(name: str) -> str:
    """Human-readable label for a registry name (falls back to the name itself)."""
    spec = MODEL_REGISTRY.get(name)
    return spec.label if spec else name


def build_pipeline(name: str, config: ModelConfig | None = None) -> Pipeline:
    """Build a fresh, unfitted ``scaler -> [PCA] -> model`` pipeline.

    Args:
        name: Registry key.
        config: Model settings; ``config.pca_components`` adds PCA to every model unless the
            spec fixes its own component count.

    Returns:
        An unfitted sklearn ``Pipeline`` whose final step is named ``model``.
    """
    cfg = config or ModelConfig()
    spec = get_spec(name)
    steps: list[tuple[str, Any]] = [("scaler", StandardScaler())]
    n_components = spec.pca_components or cfg.pca_components
    if n_components:
        steps.append(("pca", PCA(n_components=n_components, random_state=cfg.random_state)))
    steps.append(("model", spec.make_estimator(cfg)))
    return Pipeline(steps)
