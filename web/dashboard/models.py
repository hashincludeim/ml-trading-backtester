"""Metadata only. Prices live in Parquet under DATA_DIR; fitted models are joblib files."""

from __future__ import annotations

from django.db import models


class Ticker(models.Model):
    symbol = models.CharField(max_length=20, unique=True)
    first_date = models.DateField(null=True, blank=True)
    last_date = models.DateField(null=True, blank=True)
    n_rows = models.PositiveIntegerField(default=0)
    prices_path = models.CharField(max_length=500, blank=True)
    updated_at = models.DateTimeField(auto_now=True)

    class Meta:
        ordering = ["symbol"]

    def __str__(self) -> str:
        return self.symbol


class TrainingRun(models.Model):
    ticker = models.ForeignKey(Ticker, on_delete=models.CASCADE, related_name="runs")
    created_at = models.DateTimeField(auto_now_add=True)
    config_hash = models.CharField(max_length=32)
    target = models.CharField(max_length=20, default="direction", db_index=True)
    config = models.JSONField(default=dict)
    train_start = models.DateField()
    train_end = models.DateField()
    test_start = models.DateField()
    test_end = models.DateField()
    n_train = models.PositiveIntegerField()
    n_test = models.PositiveIntegerField()
    baseline = models.JSONField(default=dict, help_text="Always-positive classifier metrics")
    heuristic = models.JSONField(
        default=dict, blank=True, help_text="No-model rule metrics (volatility target only)"
    )
    predictions_path = models.CharField(max_length=500)

    class Meta:
        ordering = ["-created_at"]
        get_latest_by = "created_at"

    def __str__(self) -> str:
        return (
            f"{self.ticker} {self.target} @ {self.created_at:%Y-%m-%d %H:%M} ({self.config_hash})"
        )


class ModelResult(models.Model):
    run = models.ForeignKey(TrainingRun, on_delete=models.CASCADE, related_name="results")
    model_name = models.CharField(max_length=50)
    best_params = models.JSONField(default=dict)
    cv_scores = models.JSONField(default=dict, help_text="{metric: [fold scores]}")
    cv_accuracy_mean = models.FloatField()
    cv_accuracy_std = models.FloatField()
    cv_roc_auc_mean = models.FloatField()
    accuracy = models.FloatField()
    precision = models.FloatField()
    recall = models.FloatField()
    f1 = models.FloatField()
    roc_auc = models.FloatField()
    confusion = models.JSONField(default=list)
    roc = models.JSONField(default=dict, help_text="{fpr, tpr, auc}")
    importance = models.JSONField(default=dict, help_text="{feature: [mean, std]}")
    walk_forward = models.JSONField(
        default=dict,
        blank=True,
        help_text="Walk-forward test {metrics, confusion}; empty for runs trained without it",
    )
    model_path = models.CharField(max_length=500)

    class Meta:
        ordering = ["run", "id"]
        constraints = [
            models.UniqueConstraint(fields=["run", "model_name"], name="unique_model_per_run")
        ]

    def __str__(self) -> str:
        return f"{self.model_name} ({self.run})"
