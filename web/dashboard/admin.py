from django.contrib import admin

from dashboard.models import ModelResult, Ticker, TrainingRun


@admin.register(Ticker)
class TickerAdmin(admin.ModelAdmin):  # type: ignore[type-arg]
    list_display = ("symbol", "first_date", "last_date", "n_rows", "updated_at")


class ModelResultInline(admin.TabularInline):  # type: ignore[type-arg]
    model = ModelResult
    fields = ("model_name", "cv_accuracy_mean", "accuracy", "roc_auc", "f1")
    readonly_fields = fields
    extra = 0


@admin.register(TrainingRun)
class TrainingRunAdmin(admin.ModelAdmin):  # type: ignore[type-arg]
    list_display = ("ticker", "created_at", "config_hash", "test_start", "test_end", "n_test")
    list_filter = ("ticker",)
    inlines = [ModelResultInline]
