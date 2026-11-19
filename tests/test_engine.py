
"""Tests for PriceWise scenario evaluation."""

import json

import numpy as np
import pandas as pd
import pytest

from pricewise.pricing.engine import PricingEngine
from pricewise.pricing.policy import (
    PricingNotSupported,
    PricingPolicy,
)


class FakePredictor:
    def __init__(self, units=40):
        self.units = units
        self.calls = []

    def predict(self, features):
        self.calls.append(features.copy())

        assert len(features) == 1

        assert features.iloc[0]["CANDIDATE_PRICE"] == (
            pytest.approx(
                features.iloc[0]["LAG_1_PRICE"]
            )
        )

        assert features.iloc[0]["PRICE_CHANGE_PCT"] == (
            pytest.approx(0)
        )

        return np.array([self.units], dtype=float)


@pytest.fixture
def engine(tmp_path):
    path = tmp_path / "policy.json"

    path.write_text(
        json.dumps({
            "pricing_policy": {
                "supported_promotion": "UNMARKED",
                "max_abs_price_change_pct": 10,
                "min_training_rows": 30,
                "min_price_levels": 5,
            },
            "product_store_support": [{
                "upc": "1600066610",
                "store": 101,
                "p10": 2.90,
                "p90": 3.59,
                "training_rows": 120,
                "price_levels": 30,
            }],
            "elasticities": {
                "1600066610": -2.0,
            },
        }),
        encoding="utf-8",
    )

    return PricingEngine(
        policy=PricingPolicy(path),
        predictor=FakePredictor(),
    )


@pytest.fixture
def history():
    return pd.DataFrame({
        "UPC": ["1600066610"] * 4,
        "STORE": [101] * 4,
        "WEEK": [396, 397, 398, 399],
        "MOVE": [25, 45, 35, 30],
        "UNIT_PRICE": [3.25, 3.35, 3.35, 3.35],
        "UNIT_AAC_PROXY": [2, 2, 2, 4],
    })


def test_scenarios_use_one_reference_call_and_median_cost(
    engine, history,
):
    output = engine.evaluate(
        history,
        upc="1600066610",
        store=101,
        forecast_week=400,
    )

    assert len(engine.predictor.calls) == 1
    assert output.reference_price == 3.35
    assert output.reference_units == 40
    assert output.unit_cost_proxy == 2
    assert (output.cost_proxy_min, output.cost_proxy_max) == (
        2, 4
    )

    assert len(output.scenarios) == 58

    assert output.scenarios["price"].iloc[0] == (
        pytest.approx(3.02)
    )

    assert output.scenarios["price"].iloc[-1] == (
        pytest.approx(3.59)
    )

    baseline = output.scenarios.loc[
        np.isclose(output.scenarios["price"], 3.35)
    ].iloc[0]

    assert baseline["predicted_units"] == pytest.approx(40)
    assert baseline["revenue"] == pytest.approx(134)
    assert baseline["gross_profit"] == pytest.approx(54)

    assert baseline["profit_delta_vs_reference"] == (
        pytest.approx(0)
    )

    # Negative elasticity -> nonincreasing predicted demand
    assert np.all(
        np.diff(output.scenarios["predicted_units"]) <= 0
    )


def test_future_row_does_not_leak_cost_or_sales(
    engine, history,
):
    first = engine.evaluate(
        history,
        upc="1600066610",
        store=101,
        forecast_week=400,
    )

    added = pd.concat([
        history,
        pd.DataFrame([{
            "UPC": "1600066610",
            "STORE": 101,
            "WEEK": 400,
            "MOVE": 99999,
            "UNIT_PRICE": 0.01,
            "UNIT_AAC_PROXY": 99999,
        }]),
    ], ignore_index=True)

    second = engine.evaluate(
        added,
        upc="1600066610",
        store=101,
        forecast_week=400,
    )

    pd.testing.assert_frame_equal(
        first.scenarios, second.scenarios
    )

    assert first.unit_cost_proxy == second.unit_cost_proxy


def test_missing_history_is_rejected(engine, history):
    missing = history.loc[history["WEEK"] != 398]

    with pytest.raises(
        PricingNotSupported,
        match="four consecutive",
    ):
        engine.evaluate(
            missing,
            upc="1600066610",
            store=101,
            forecast_week=400,
        )


def test_invalid_cost_is_rejected(engine, history):
    history.loc[
        history["WEEK"] == 399, "UNIT_AAC_PROXY"
    ] = float("nan")

    with pytest.raises(
        PricingNotSupported,
        match="acquisition-cost",
    ):
        engine.evaluate(
            history,
            upc="1600066610",
            store=101,
            forecast_week=400,
        )


def test_unsupported_promotion_is_rejected(engine, history):
    with pytest.raises(
        PricingNotSupported,
        match="Promotion",
    ):
        engine.evaluate(
            history,
            upc="1600066610",
            store=101,
            forecast_week=400,
            promo_context="S",
        )


def test_invalid_elasticity_metadata_is_rejected(tmp_path):
    path = tmp_path / "policy.json"

    path.write_text(
        json.dumps({
            "pricing_policy": {
                "supported_promotion": "UNMARKED",
                "max_abs_price_change_pct": 10,
                "min_training_rows": 30,
                "min_price_levels": 5,
            },
            "product_store_support": [],
            "elasticities": {
                "1600066610": 0,
            },
        }),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="negative"):
        PricingPolicy(path)
