
"""Tests for the three PriceWise pricing strategies."""

from dataclasses import replace

import numpy as np
import pandas as pd
import pytest

from pricewise.pricing.engine import ScenarioEvaluation
from pricewise.pricing.strategies import select_strategies


@pytest.fixture
def evaluation():
    """Deterministic scenario grid resembling Store 101."""

    reference_price = 3.35
    reference_units = 36.8
    cost = 2.8251
    epsilon = -1.864

    prices = np.arange(302, 360) / 100

    units = (
        reference_units
        * (prices / reference_price) ** epsilon
    )

    revenue = prices * units
    margin = prices - cost
    profit = margin * units

    baseline_profit = (
        (reference_price - cost) * reference_units
    )

    scenarios = pd.DataFrame({
        "price": prices,
        "predicted_units": units,
        "revenue": revenue,
        "margin_per_unit": margin,
        "gross_margin_pct": 100 * margin / prices,
        "gross_profit": profit,
    })

    return ScenarioEvaluation(
        upc="1600066610",
        store=101,
        forecast_week=400,
        reference_price=reference_price,
        reference_units=reference_units,
        unit_cost_proxy=cost,
        cost_proxy_min=2.8224,
        cost_proxy_max=2.8274,
        lower_bound=3.015,
        upper_bound=3.59,
        baseline_gross_profit=baseline_profit,
        scenarios=scenarios,
    )


def test_three_strategies_select_expected_prices(evaluation):
    decisions = select_strategies(evaluation)

    assert all(
        decision.status == "scenario"
        for decision in decisions.values()
    )

    assert decisions["Conservative"].price == pytest.approx(
        3.32
    )
    assert decisions["Balanced"].price == pytest.approx(
        3.54
    )
    assert decisions["Aggressive"].price == pytest.approx(
        3.59
    )


def test_each_strategy_obeys_business_constraints(evaluation):
    decisions = select_strategies(evaluation)

    conservative = decisions["Conservative"]
    balanced = decisions["Balanced"]
    aggressive = decisions["Aggressive"]

    baseline_units = evaluation.reference_units
    baseline_profit = evaluation.baseline_gross_profit

    assert conservative.predicted_units >= baseline_units
    assert conservative.gross_profit >= 0.95 * baseline_profit

    assert balanced.predicted_units >= 0.90 * baseline_units
    assert balanced.gross_profit >= baseline_profit

    assert aggressive.predicted_units >= 0.85 * baseline_units
    assert aggressive.gross_profit >= baseline_profit

    for decision in decisions.values():
        assert (
            evaluation.lower_bound
            <= decision.price
            <= evaluation.upper_bound
        )
        assert decision.price > evaluation.cost_proxy_max


def test_unstable_cost_withholds_all(evaluation):
    unstable = replace(
        evaluation,
        cost_proxy_min=2.50,
    )

    decisions = select_strategies(unstable)

    assert all(
        decision.status == "withheld"
        for decision in decisions.values()
    )

    assert all(
        "unstable" in decision.explanation
        for decision in decisions.values()
    )


def test_nonpositive_baseline_profit_withholds_all(evaluation):
    invalid = replace(
        evaluation,
        baseline_gross_profit=0.0,
    )

    decisions = select_strategies(invalid)

    assert all(
        decision.status == "withheld"
        for decision in decisions.values()
    )


def test_strategies_may_select_same_price(evaluation):
    # All three strategies should be allowed
    # to choose the same price if it is the
    # only eligible option.
    only_reference = evaluation.scenarios.loc[
        np.isclose(
            evaluation.scenarios["price"],
            evaluation.reference_price,
        )
    ].copy()

    restricted = replace(
        evaluation,
        scenarios=only_reference,
        lower_bound=3.35,
        upper_bound=3.35,
    )

    decisions = select_strategies(restricted)

    assert all(
        decision.status == "scenario"
        and decision.price == pytest.approx(3.35)
        for decision in decisions.values()
    )


def test_no_qualifying_candidate_is_withheld(evaluation):
    only_low_price = evaluation.scenarios.loc[
        np.isclose(
            evaluation.scenarios["price"], 3.02
        )
    ].copy()

    restricted = replace(
        evaluation,
        scenarios=only_low_price,
    )

    decisions = select_strategies(restricted)

    assert all(
        decision.status == "withheld"
        for decision in decisions.values()
    )


def test_duplicate_candidate_prices_are_rejected(evaluation):
    duplicated = pd.concat(
        [
            evaluation.scenarios,
            evaluation.scenarios.iloc[[0]],
        ],
        ignore_index=True,
    )

    invalid = replace(
        evaluation,
        scenarios=duplicated,
    )

    with pytest.raises(ValueError, match="Duplicate"):
        select_strategies(invalid)


def test_nonfinite_scenario_profit_is_rejected(evaluation):
    scenarios = evaluation.scenarios.copy()

    scenarios.loc[
        scenarios.index[0], "gross_profit"
    ] = np.nan

    invalid = replace(
        evaluation,
        scenarios=scenarios,
    )

    with pytest.raises(ValueError, match="Nonfinite"):
        select_strategies(invalid)
