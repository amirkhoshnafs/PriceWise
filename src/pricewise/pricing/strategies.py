
"""PriceWise: transparent pricing scenario strategies."""

from dataclasses import dataclass

import numpy as np
import pandas as pd

from pricewise.pricing.engine import ScenarioEvaluation


STRATEGY_NAMES = (
    "Conservative",
    "Balanced",
    "Aggressive",
)

# Initial business-policy thresholds.
MAX_COST_SPREAD_PCT = 5.0

CONSERVATIVE_PROFIT_FLOOR = 0.95
CONSERVATIVE_VOLUME_FLOOR = 1.00

BALANCED_PROFIT_FLOOR = 1.00
BALANCED_VOLUME_FLOOR = 0.90

AGGRESSIVE_PROFIT_FLOOR = 1.00
AGGRESSIVE_VOLUME_FLOOR = 0.85

TOLERANCE = 1e-8


@dataclass(frozen=True)
class StrategyDecision:
    """One selected scenario, or a reason for withholding it."""

    strategy: str
    status: str
    explanation: str

    price: float | None = None
    predicted_units: float | None = None
    revenue: float | None = None
    margin_per_unit: float | None = None
    gross_margin_pct: float | None = None
    gross_profit: float | None = None

    price_change_pct: float | None = None
    units_change_pct: float | None = None
    profit_change_pct: float | None = None


def _withheld(
    name: str,
    reason: str,
) -> StrategyDecision:
    return StrategyDecision(
        strategy=name,
        status="withheld",
        explanation=reason,
    )


def _withhold_all(
    reason: str,
) -> dict[str, StrategyDecision]:
    return {
        name: _withheld(name, reason)
        for name in STRATEGY_NAMES
    }


def _choose(
    *,
    name: str,
    candidates: pd.DataFrame,
    explanation: str,
    evaluation: ScenarioEvaluation,
) -> StrategyDecision:
    """Convert the first eligible row into a decision."""

    if candidates.empty:
        return _withheld(
            name,
            "No candidate satisfies this strategy's "
            "demand and profitability requirements.",
        )

    row = candidates.iloc[0]

    return StrategyDecision(
        strategy=name,
        status="scenario",
        explanation=explanation,
        price=float(row["price"]),
        predicted_units=float(row["predicted_units"]),
        revenue=float(row["revenue"]),
        margin_per_unit=float(row["margin_per_unit"]),
        gross_margin_pct=float(row["gross_margin_pct"]),
        gross_profit=float(row["gross_profit"]),
        price_change_pct=100 * (
            float(row["price"])
            / evaluation.reference_price
            - 1
        ),
        units_change_pct=100 * (
            float(row["predicted_units"])
            / evaluation.reference_units
            - 1
        ),
        profit_change_pct=100 * (
            float(row["gross_profit"])
            / evaluation.baseline_gross_profit
            - 1
        ),
    )


def select_strategies(
    evaluation: ScenarioEvaluation,
) -> dict[str, StrategyDecision]:
    """
    Select three explainable pricing scenarios.

    Uses only previously evaluated, policy-supported
    candidate prices. Never calls the ML model again.
    """

    # --------------------------------------------------
    # A. Check cost reliability and reference forecast
    # --------------------------------------------------

    cost = float(evaluation.unit_cost_proxy)
    cost_min = float(evaluation.cost_proxy_min)
    cost_max = float(evaluation.cost_proxy_max)

    reference_price = float(evaluation.reference_price)
    baseline_units = float(evaluation.reference_units)
    baseline_profit = float(
        evaluation.baseline_gross_profit
    )

    values = [
        cost, cost_min, cost_max,
        reference_price, baseline_units, baseline_profit,
        evaluation.lower_bound, evaluation.upper_bound,
    ]

    if not np.isfinite(values).all():
        raise ValueError(
            "Nonfinite scenario evaluation metadata"
        )

    if (
        cost <= 0
        or cost_min <= 0
        or cost_max < cost_min
        or not cost_min <= cost <= cost_max
    ):
        return _withhold_all(
            "Historical unit-cost proxy is invalid."
        )

    cost_spread_pct = (
        100 * (cost_max - cost_min) / cost
    )

    if cost_spread_pct > MAX_COST_SPREAD_PCT:
        return _withhold_all(
            "Recent acquisition-cost estimates are "
            "too unstable for a pricing comparison."
        )

    if baseline_units <= 0 or reference_price <= 0:
        return _withhold_all(
            "Reference demand or price is invalid."
        )

    if baseline_profit <= 0:
        return _withhold_all(
            "Reference gross profit is not positive."
        )

    # --------------------------------------------------
    # B. Validate the already-evaluated price scenarios
    # --------------------------------------------------

    scenarios = evaluation.scenarios.copy()

    required = {
        "price", "predicted_units", "revenue",
        "margin_per_unit", "gross_margin_pct",
        "gross_profit",
    }

    missing = required - set(scenarios.columns)

    if missing:
        raise ValueError(
            f"Missing scenario columns: {sorted(missing)}"
        )

    if scenarios.empty:
        return _withhold_all(
            "No candidate prices were generated."
        )

    if scenarios["price"].duplicated().any():
        raise ValueError("Duplicate candidate prices")

    numbers = scenarios[list(required)].to_numpy(
        dtype=float
    )

    if not np.isfinite(numbers).all():
        raise ValueError(
            "Nonfinite values in price scenarios"
        )

    if (
        scenarios["price"].lt(
            evaluation.lower_bound - TOLERANCE
        ).any()
        or scenarios["price"].gt(
            evaluation.upper_bound + TOLERANCE
        ).any()
    ):
        raise ValueError(
            "Candidate price outside policy bounds"
        )

    # --------------------------------------------------
    # C. Common profitability protections
    # --------------------------------------------------

    # Price must exceed even the highest recent
    # acquisition-cost proxy, not only its median.
    viable = scenarios.loc[
        (scenarios["price"] > cost_max)
        & (scenarios["margin_per_unit"] > 0)
        & (scenarios["gross_profit"] >= 0)
        & (scenarios["predicted_units"] > 0)
    ].copy()

    if viable.empty:
        return _withhold_all(
            "No candidate has an acceptable positive "
            "unit margin and predicted demand."
        )

    # --------------------------------------------------
    # D. Conservative: favor volume, protect profit
    # --------------------------------------------------

    conservative = viable.loc[
        (
            viable["gross_profit"] + TOLERANCE
            >= CONSERVATIVE_PROFIT_FLOOR
            * baseline_profit
        )
        & (
            viable["predicted_units"] + TOLERANCE
            >= CONSERVATIVE_VOLUME_FLOOR
            * baseline_units
        )
    ].sort_values(
        ["price", "gross_profit"],
        ascending=[True, False],
    )

    # --------------------------------------------------
    # E. Balanced: maximize profit, protect volume
    # --------------------------------------------------

    balanced = viable.loc[
        (
            viable["gross_profit"] + TOLERANCE
            >= BALANCED_PROFIT_FLOOR
            * baseline_profit
        )
        & (
            viable["predicted_units"] + TOLERANCE
            >= BALANCED_VOLUME_FLOOR
            * baseline_units
        )
    ].sort_values(
        ["gross_profit", "price"],
        ascending=[False, True],
    )

    # --------------------------------------------------
    # F. Aggressive: favor higher prices
    # --------------------------------------------------

    aggressive = viable.loc[
        (
            viable["gross_profit"] + TOLERANCE
            >= AGGRESSIVE_PROFIT_FLOOR
            * baseline_profit
        )
        & (
            viable["predicted_units"] + TOLERANCE
            >= AGGRESSIVE_VOLUME_FLOOR
            * baseline_units
        )
    ].sort_values(
        ["price", "gross_profit"],
        ascending=[False, False],
    )

    # --------------------------------------------------
    # G. Return selections or explicit withholding
    # --------------------------------------------------

    return {
        "Conservative": _choose(
            name="Conservative",
            candidates=conservative,
            explanation=(
                "Lowest qualifying price; preserves "
                "reference volume and at least 95% "
                "of modeled reference gross profit."
            ),
            evaluation=evaluation,
        ),
        "Balanced": _choose(
            name="Balanced",
            candidates=balanced,
            explanation=(
                "Highest modeled gross profit with "
                "at least 90% of reference demand "
                "and no modeled profit decline."
            ),
            evaluation=evaluation,
        ),
        "Aggressive": _choose(
            name="Aggressive",
            candidates=aggressive,
            explanation=(
                "Highest qualifying price with "
                "at least 85% of reference demand "
                "and no modeled profit decline."
            ),
            evaluation=evaluation,
        ),
    }
