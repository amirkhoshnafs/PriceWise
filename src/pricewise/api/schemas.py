
"""Public HTTP request and response contracts for PriceWise."""

from typing import Literal

from pydantic import BaseModel, Field, field_validator, ConfigDict


class PricingRequest(BaseModel):

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "upc": "1600066610",
                "store": 101,
                "promo_context": "UNMARKED",
            }
        }
    )
    upc: str = Field(pattern=r"^\d+$")
    store: int = Field(gt=0)
    promo_context: str = Field(
        default="UNMARKED",
        min_length=1,
        max_length=20,
    )

    @field_validator("promo_context")
    @classmethod
    def normalize_promotion(cls, value: str) -> str:
        value = value.strip().upper()

        if not value:
            raise ValueError(
                "Promotion context cannot be blank"
            )

        return value


class HealthResponse(BaseModel):
    status: Literal["ok"]
    model: str
    history_rows: int


class ReferenceSummary(BaseModel):
    price: float
    predicted_units: float
    unit_cost_proxy: float
    cost_proxy_min: float
    cost_proxy_max: float
    modeled_gross_profit: float


class StrategyResult(BaseModel):
    strategy: str
    status: Literal["scenario", "withheld"]
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


class ScenarioResult(BaseModel):
    price: float
    price_change_pct: float
    predicted_units: float
    revenue: float
    margin_per_unit: float
    gross_margin_pct: float
    gross_profit: float
    profit_delta_vs_reference: float


class PricingResponse(BaseModel):
    upc: str
    product: str
    store: int

    last_observed_week: int
    forecast_week: int
    historical_demo: bool
    promo_context: str

    reference: ReferenceSummary
    strategies: dict[str, StrategyResult]
    scenarios: list[ScenarioResult]

    disclaimer: str
