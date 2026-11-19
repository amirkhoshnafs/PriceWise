
"""Evaluate bounded PriceWise demand and gross-profit scenarios."""

from dataclasses import dataclass

import numpy as np
import pandas as pd

from pricewise.data.features import build_future_feature_row
from pricewise.pricing.policy import (
    PricingNotSupported,
    PricingPolicy,
)


@dataclass(frozen=True)
class ScenarioEvaluation:
    """Reference forecast, historical cost, and candidate outcomes."""

    upc: str
    store: int
    forecast_week: int
    reference_price: float
    reference_units: float

    unit_cost_proxy: float
    cost_proxy_min: float
    cost_proxy_max: float

    lower_bound: float
    upper_bound: float
    baseline_gross_profit: float

    scenarios: pd.DataFrame


class PricingEngine:
    """Connect pricing policy, history, and ONNX reference demand."""

    def __init__(self, *, policy: PricingPolicy, predictor):
        self.policy = policy
        self.predictor = predictor

    def evaluate(
        self,
        sales: pd.DataFrame,
        *,
        upc: str,
        store: int,
        forecast_week: int,
        promo_context: str = "UNMARKED",
    ) -> ScenarioEvaluation:
        """Evaluate supported prices without future-week actual sales."""

        # --------------------------------------------------
        # A. Validate the historical input
        # --------------------------------------------------

        if not isinstance(sales, pd.DataFrame):
            raise TypeError("sales must be a DataFrame")

        required = {
            "UPC", "STORE", "WEEK", "MOVE",
            "UNIT_PRICE", "UNIT_AAC_PROXY",
        }

        missing = sorted(required - set(sales.columns))

        if missing:
            raise ValueError(
                f"Missing historical columns: {missing}"
            )

        if (
            isinstance(forecast_week, (bool, np.bool_))
            or not isinstance(
                forecast_week, (int, np.integer)
            )
            or forecast_week <= 4
        ):
            raise ValueError(
                "forecast_week must be an integer greater than 4"
            )

        # --------------------------------------------------
        # B. Select the exact four previous calendar weeks
        # --------------------------------------------------

        expected_weeks = list(
            range(forecast_week - 4, forecast_week)
        )

        mask = (
            sales["UPC"]
            .astype("string")
            .str.strip()
            .eq(upc)
            .fillna(False)

            & pd.to_numeric(
                sales["STORE"], errors="coerce"
            ).eq(store)

            & pd.to_numeric(
                sales["WEEK"], errors="coerce"
            ).isin(expected_weeks)
        )

        history = sales.loc[
            mask,
            ["WEEK", "UNIT_PRICE", "UNIT_AAC_PROXY"],
        ].copy()

        history["WEEK"] = pd.to_numeric(
            history["WEEK"],
            errors="coerce",
        )

        history = history.sort_values("WEEK")

        if history["WEEK"].tolist() != expected_weeks:
            raise PricingNotSupported(
                "Need exactly four consecutive historical "
                f"weeks {expected_weeks} for UPC {upc}, "
                f"store {store}"
            )

        reference_price = float(
            history["UNIT_PRICE"].iloc[-1]
        )

        # --------------------------------------------------
        # C. Apply the frozen pricing policy
        # --------------------------------------------------

        grid = self.policy.candidates(
            upc=upc,
            store=store,
            reference_price=reference_price,
            promo_context=promo_context,
        )

        epsilon = self.policy.elasticities.get(upc)

        if (
            epsilon is None
            or not np.isfinite(epsilon)
            or epsilon >= 0
        ):
            raise PricingNotSupported(
                f"No validated negative elasticity for UPC {upc}"
            )

        # --------------------------------------------------
        # D. Estimate acquisition cost from past observations
        # --------------------------------------------------

        costs = pd.to_numeric(
            history["UNIT_AAC_PROXY"],
            errors="coerce",
        ).to_numpy(dtype=float)

        if (
            not np.isfinite(costs).all()
            or (costs <= 0).any()
        ):
            raise PricingNotSupported(
                "Historical acquisition-cost proxies "
                "are unavailable or invalid"
            )

        # Fixed across all candidate selling prices.
        unit_cost = float(np.median(costs))

        # --------------------------------------------------
        # E. Build the reference-price feature row
        # --------------------------------------------------

        features = build_future_feature_row(
            sales,
            upc=upc,
            store=store,
            forecast_week=forecast_week,
            candidate_price=reference_price,
            promo_context=promo_context,
        )

        # --------------------------------------------------
        # F. Run ONNX once at the reference price
        # --------------------------------------------------

        reference_output = np.asarray(
            self.predictor.predict(features),
            dtype=float,
        ).reshape(-1)

        if (
            len(reference_output) != 1
            or not np.isfinite(reference_output).all()
            or (reference_output < 0).any()
        ):
            raise ValueError(
                "Invalid ONNX reference demand forecast"
            )

        reference_units = float(reference_output[0])

        # --------------------------------------------------
        # G. Evaluate every approved candidate price
        # --------------------------------------------------

        prices = np.asarray(
            grid.prices,
            dtype=float,
        )

        predicted_units = (
            reference_units
            * np.power(
                prices / reference_price,
                epsilon,
            )
        )

        if not np.isfinite(predicted_units).all():
            raise ValueError(
                "Nonfinite scenario demand"
            )

        revenue = prices * predicted_units

        margin_per_unit = prices - unit_cost

        gross_profit = (
            margin_per_unit * predicted_units
        )

        baseline_profit = (
            (reference_price - unit_cost)
            * reference_units
        )

        # --------------------------------------------------
        # H. Return an explainable scenario table
        # --------------------------------------------------

        scenarios = pd.DataFrame({
            "price": prices,
            "price_change_pct": (
                100 * (prices / reference_price - 1)
            ),
            "predicted_units": predicted_units,
            "revenue": revenue,
            "margin_per_unit": margin_per_unit,
            "gross_margin_pct": (
                100 * margin_per_unit / prices
            ),
            "gross_profit": gross_profit,
            "profit_delta_vs_reference": (
                gross_profit - baseline_profit
            ),
        })

        return ScenarioEvaluation(
            upc=upc,
            store=store,
            forecast_week=forecast_week,
            reference_price=reference_price,
            reference_units=reference_units,
            unit_cost_proxy=unit_cost,
            cost_proxy_min=float(costs.min()),
            cost_proxy_max=float(costs.max()),
            lower_bound=grid.lower_bound,
            upper_bound=grid.upper_bound,
            baseline_gross_profit=baseline_profit,
            scenarios=scenarios,
        )
