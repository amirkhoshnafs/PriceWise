
"""PriceWise REST API: HTTP delivery for existing pricing services."""

from contextlib import asynccontextmanager
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Annotated, Callable

import pandas as pd
from fastapi import (
    Depends,
    FastAPI,
    HTTPException,
    Request,
)

from pricewise.api.schemas import (
    HealthResponse,
    PricingRequest,
    PricingResponse,
    ReferenceSummary,
    ScenarioResult,
    StrategyResult,
)
from pricewise.ml.inference import OnnxDemandPredictor
from pricewise.pricing.engine import PricingEngine
from pricewise.pricing.policy import (
    PricingNotSupported,
    PricingPolicy,
)
from pricewise.pricing.strategies import select_strategies


ROOT = Path(__file__).resolve().parents[3]

PRODUCT_NAMES = {
    "1600066610": "Cheerios 15 oz",
    "3800000120": "Corn Flakes 18 oz",
}


@dataclass(frozen=True)
class PricingResources:
    sales: pd.DataFrame
    engine: PricingEngine


def load_resources() -> PricingResources:
    """Load historical data and inference components once."""

    sales = pd.read_parquet(
        ROOT / "data/processed/cereal_focus.parquet",
        columns=[
            "UPC",
            "STORE",
            "WEEK",
            "MOVE",
            "UNIT_PRICE",
            "UNIT_AAC_PROXY",
        ],
    )

    if (
        sales.empty
        or sales.duplicated(
            ["UPC", "STORE", "WEEK"]
        ).any()
    ):
        raise RuntimeError(
            "Historical sales are empty or have duplicate keys"
        )

    engine = PricingEngine(
        policy=PricingPolicy(
            ROOT / "models/pricewise_policy.json"
        ),
        predictor=OnnxDemandPredictor(
            ROOT / "models/pricewise_rf.onnx"
        ),
    )

    return PricingResources(
        sales=sales,
        engine=engine,
    )


def get_resources(request: Request) -> PricingResources:
    """Provide the initialized resources to an endpoint."""

    return request.app.state.pricewise


def create_app(
    resources_loader: Callable[
        [], PricingResources
    ] | None = None,
) -> FastAPI:
    """Build the application with injectable startup resources."""

    loader = (
        resources_loader
        if resources_loader is not None
        else load_resources
    )

    @asynccontextmanager
    async def lifespan(api: FastAPI):
        # Startup: load the data and model once.
        api.state.pricewise = loader()

        yield

        # Shutdown: no external connections to close yet.

    api = FastAPI(
        title="PriceWise API",
        version="0.1.0",
        lifespan=lifespan,
    )

    # ----------------------------------------------
    # GET /health
    # ----------------------------------------------

    @api.get(
        "/health",
        response_model=HealthResponse,
    )
    def health(
        resources: Annotated[
            PricingResources,
            Depends(get_resources),
        ],
    ):
        return HealthResponse(
            status="ok",
            model="ONNX Runtime",
            history_rows=len(resources.sales),
        )

    # ----------------------------------------------
    # POST /api/v1/pricing/scenarios
    # ----------------------------------------------

    @api.post(
        "/api/v1/pricing/scenarios",
        response_model=PricingResponse,
    )
    def pricing_scenarios(
        body: PricingRequest,
        resources: Annotated[
            PricingResources,
            Depends(get_resources),
        ],
    ):
        sales = resources.sales

        # A. Find history for the requested product/store.
        mask = (
            sales["UPC"]
            .astype("string")
            .eq(body.upc)
            .fillna(False)
            & sales["STORE"].eq(body.store)
        )

        product_history = sales.loc[mask]

        if product_history.empty:
            raise HTTPException(
                status_code=404,
                detail={
                    "code": "HISTORY_NOT_FOUND",
                    "message": (
                        "No history for this product-store."
                    ),
                },
            )

        # B. Always forecast one week after the
        # latest available historical observation.
        last_week = int(
            product_history["WEEK"].max()
        )

        try:
            # C. Policy + historical features + ONNX
            # + demand/revenue/profit simulation.
            simulation = resources.engine.evaluate(
                sales,
                upc=body.upc,
                store=body.store,
                forecast_week=last_week + 1,
                promo_context=body.promo_context,
            )

        except PricingNotSupported as exc:
            raise HTTPException(
                status_code=409,
                detail={
                    "code": "PRICING_NOT_SUPPORTED",
                    "message": str(exc),
                },
            ) from exc

        # D. Apply our three existing business strategies.
        decisions = select_strategies(
            simulation
        )

        # E. Serialize results using the public schema.
        return PricingResponse(
            upc=simulation.upc,
            product=PRODUCT_NAMES.get(
                simulation.upc,
                simulation.upc,
            ),
            store=simulation.store,
            last_observed_week=last_week,
            forecast_week=simulation.forecast_week,
            historical_demo=True,
            promo_context=body.promo_context,

            reference=ReferenceSummary(
                price=simulation.reference_price,
                predicted_units=simulation.reference_units,
                unit_cost_proxy=simulation.unit_cost_proxy,
                cost_proxy_min=simulation.cost_proxy_min,
                cost_proxy_max=simulation.cost_proxy_max,
                modeled_gross_profit=(
                    simulation.baseline_gross_profit
                ),
            ),

            strategies={
                name: StrategyResult.model_validate(
                    asdict(decision)
                )
                for name, decision in decisions.items()
            },

            scenarios=[
                ScenarioResult.model_validate(row)
                for row in (
                    simulation.scenarios.to_dict(
                        orient="records"
                    )
                )
            ],

            disclaimer=(
                "Historical-data scenario, not a "
                "current-market forecast. Modeled price "
                "response is observational, not a proven "
                "causal effect. No autonomous price changes."
            ),
        )

    return api


# Uvicorn imports this ASGI application.
app = create_app()
