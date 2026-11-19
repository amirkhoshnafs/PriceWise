
"""HTTP tests for PriceWise using lightweight test resources."""

from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest
from fastapi.testclient import TestClient

from pricewise.api.main import (
    PricingResources,
    create_app,
)
from pricewise.pricing.policy import PricingNotSupported


class FakeEngine:
    def __init__(self):
        self.calls = []

    def evaluate(
        self,
        sales,
        *,
        upc,
        store,
        forecast_week,
        promo_context,
    ):
        self.calls.append(
            (upc, store, forecast_week, promo_context)
        )

        if promo_context != "UNMARKED":
            raise PricingNotSupported(
                "Unsupported promotion"
            )

        # A deterministic scenario grid.
        prices = np.arange(302, 360) / 100
        units = 36.8 * (prices / 3.35) ** -1.864

        cost = 2.8251
        margin = prices - cost
        baseline = (3.35 - cost) * 36.8

        scenarios = pd.DataFrame({
            "price": prices,
            "price_change_pct": (
                100 * (prices / 3.35 - 1)
            ),
            "predicted_units": units,
            "revenue": prices * units,
            "margin_per_unit": margin,
            "gross_margin_pct": 100 * margin / prices,
            "gross_profit": margin * units,
            "profit_delta_vs_reference": (
                margin * units - baseline
            ),
        })

        return SimpleNamespace(
            upc=upc,
            store=store,
            forecast_week=forecast_week,
            reference_price=3.35,
            reference_units=36.8,
            unit_cost_proxy=cost,
            cost_proxy_min=2.8224,
            cost_proxy_max=2.8274,
            lower_bound=3.015,
            upper_bound=3.59,
            baseline_gross_profit=baseline,
            scenarios=scenarios,
        )


@pytest.fixture
def api_client():
    """Create a complete HTTP app with test-only resources."""

    sales = pd.DataFrame({
        "UPC": ["1600066610"] * 4,
        "STORE": [101] * 4,
        "WEEK": [396, 397, 398, 399],
    })

    engine = FakeEngine()

    app = create_app(
        resources_loader=lambda: PricingResources(
            sales=sales,
            engine=engine,
        )
    )

    # The context manager triggers FastAPI lifespan.
    with TestClient(app) as client:
        yield client, engine


def test_health_returns_ok(api_client):
    client, _ = api_client

    response = client.get("/health")

    assert response.status_code == 200
    assert response.json()["status"] == "ok"
    assert response.json()["history_rows"] == 4


def test_pricing_response_contains_strategies_and_curve(
    api_client,
):
    client, engine = api_client

    response = client.post(
        "/api/v1/pricing/scenarios",
        json={
            "upc": "1600066610",
            "store": 101,
        },
    )

    assert response.status_code == 200

    data = response.json()

    assert data["last_observed_week"] == 399
    assert data["forecast_week"] == 400
    assert data["historical_demo"] is True

    assert data["reference"]["price"] == 3.35

    assert data["reference"][
        "predicted_units"
    ] == pytest.approx(36.8)

    assert len(data["scenarios"]) == 58

    assert set(data["strategies"]) == {
        "Conservative",
        "Balanced",
        "Aggressive",
    }

    assert data["strategies"][
        "Conservative"
    ]["price"] == pytest.approx(3.32)

    assert engine.calls == [
        ("1600066610", 101, 400, "UNMARKED")
    ]


def test_unknown_product_history_returns_404(api_client):
    client, _ = api_client

    response = client.post(
        "/api/v1/pricing/scenarios",
        json={
            "upc": "3800000120",
            "store": 101,
        },
    )

    assert response.status_code == 404

    assert response.json()["detail"]["code"] == (
        "HISTORY_NOT_FOUND"
    )


def test_unsupported_promotion_returns_409(api_client):
    client, _ = api_client

    response = client.post(
        "/api/v1/pricing/scenarios",
        json={
            "upc": "1600066610",
            "store": 101,
            "promo_context": "s",
        },
    )

    assert response.status_code == 409

    assert response.json()["detail"]["code"] == (
        "PRICING_NOT_SUPPORTED"
    )


def test_invalid_request_returns_422(api_client):
    client, _ = api_client

    response = client.post(
        "/api/v1/pricing/scenarios",
        json={
            "upc": "not-a-upc",
            "store": 0,
        },
    )

    assert response.status_code == 422
