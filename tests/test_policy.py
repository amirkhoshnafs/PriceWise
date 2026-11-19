
import json

import pytest

from pricewise.pricing.policy import (
    PricingPolicy,
    PricingNotSupported,
)


@pytest.fixture
def policy(tmp_path):

    path = tmp_path / "pricewise_policy.json"

    path.write_text(
        json.dumps({
            "pricing_policy": {
                "supported_promotion": "UNMARKED",
                "max_abs_price_change_pct": 10,
                "min_training_rows": 30,
                "min_price_levels": 5,
            },
            "product_store_support": [
                {
                    "upc": "1600066610",
                    "store": 101,
                    "p10": 2.90,
                    "p90": 3.59,
                    "training_rows": 200,
                    "price_levels": 40,
                },
                {
                    "upc": "3800000120",
                    "store": 101,
                    "p10": 1.90,
                    "p90": 2.84,
                    "training_rows": 200,
                    "price_levels": 30,
                },
                {
                    "upc": "1600066610",
                    "store": 102,
                    "p10": 2.90,
                    "p90": 3.59,
                    "training_rows": 20,
                    "price_levels": 40,
                },
                {
                    "upc": "1600066610",
                    "store": 103,
                    "p10": 3.005,
                    "p90": 3.005,
                    "training_rows": 200,
                    "price_levels": 8,
                },
            ],
        }),
        encoding="utf-8",
    )

    return PricingPolicy(path)


def test_price_grid_intersects_history_and_ten_percent(
    policy,
):
    grid = policy.candidates(
        upc="1600066610",
        store=101,
        reference_price=3.35,
    )

    assert grid.lower_bound == pytest.approx(3.015)
    assert grid.upper_bound == pytest.approx(3.59)

    assert grid.prices[0] == pytest.approx(3.02)
    assert grid.prices[-1] == pytest.approx(3.59)

    assert len(grid.prices) == 58
    assert 3.35 in grid.prices


def test_rejects_unknown_store(policy):
    with pytest.raises(
        PricingNotSupported,
        match="No training support",
    ):
        policy.candidates(
            upc="1600066610",
            store=999,
            reference_price=3.35,
        )


def test_rejects_promoted_context(policy):
    with pytest.raises(
        PricingNotSupported,
        match="Promotion",
    ):
        policy.candidates(
            upc="1600066610",
            store=101,
            reference_price=3.35,
            promo_context="S",
        )


def test_rejects_reference_outside_training_support(
    policy,
):
    with pytest.raises(
        PricingNotSupported,
        match="outside training support",
    ):
        policy.candidates(
            upc="3800000120",
            store=101,
            reference_price=2.85,
        )


def test_rejects_insufficient_training_rows(policy):
    with pytest.raises(
        PricingNotSupported,
        match="training rows",
    ):
        policy.candidates(
            upc="1600066610",
            store=102,
            reference_price=3.35,
        )


def test_rejects_no_eligible_cent_price(policy):
    with pytest.raises(
        PricingNotSupported,
        match="whole-cent",
    ):
        policy.candidates(
            upc="1600066610",
            store=103,
            reference_price=3.005,
        )


def test_rejects_invalid_reference_price(policy):
    with pytest.raises(
        ValueError,
        match="positive",
    ):
        policy.candidates(
            upc="1600066610",
            store=101,
            reference_price=0,
        )
