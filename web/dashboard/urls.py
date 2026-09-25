from django.urls import path

from dashboard import views

app_name = "dashboard"

urlpatterns = [
    path("", views.OverviewView.as_view(), name="overview"),
    path("indicators/", views.IndicatorsView.as_view(), name="indicators"),
    path("exploration/", views.ExplorationView.as_view(), name="exploration"),
    path("models/", views.ModelsView.as_view(), name="models"),
    path("backtest/", views.BacktestView.as_view(), name="backtest"),
    path("multi-ticker/", views.MultiTickerView.as_view(), name="multi_ticker"),
    path("about/", views.AboutView.as_view(), name="about"),
]
