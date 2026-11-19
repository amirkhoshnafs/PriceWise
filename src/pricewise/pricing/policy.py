
"""Frozen training-support rules for PriceWise candidate prices."""

from dataclasses import dataclass
from decimal import (
    Decimal,
    InvalidOperation,
    ROUND_CEILING,
    ROUND_FLOOR,
)
import json
from pathlib import Path


def _decimal(value: object, label: str) -> Decimal:
    """Convert a finite numeric value to Decimal."""

    if isinstance(value, bool):
        raise ValueError(f"{label} must be a finite number")

    try:
        number = Decimal(str(value))
    except (InvalidOperation, TypeError, ValueError) as exc:
        raise ValueError(
            f"{label} must be a finite number"
        ) from exc

    if not number.is_finite():
        raise ValueError(f"{label} must be a finite number")

    return number


class PricingNotSupported(ValueError):
    """Requested price scenario is outside the frozen policy."""


@dataclass(frozen=True)
class PriceCandidates:
    upc: str
    store: int
    reference_price: float
    lower_bound: float
    upper_bound: float
    prices: tuple[float, ...]


class PricingPolicy:
    """Read frozen JSON rules and generate supported price grids."""

    def __init__(self, path: str | Path):

        config = json.loads(
            Path(path).read_text(encoding="utf-8")
        )

        rules = config["pricing_policy"]

        self.supported_promo = str(
            rules["supported_promotion"]
        ).strip().upper()

        self.max_change_pct = _decimal(
            rules["max_abs_price_change_pct"],
            "max_abs_price_change_pct",
        )

        self.min_training_rows = int(
            rules["min_training_rows"]
        )

        self.min_price_levels = int(
            rules["min_price_levels"]
        )

        if (
            not self.supported_promo
            or not (
                Decimal("0")
                < self.max_change_pct
                < Decimal("100")
            )
            or self.min_training_rows < 1
            or self.min_price_levels < 1
        ):
            raise ValueError(
                "Invalid pricing policy settings"
            )

        # Key = (UPC, STORE)
        # Value = training-period support information.
        self.support = {}

        for entry in config["product_store_support"]:

            key = (
                str(entry["upc"]).strip(),
                int(entry["store"]),
            )

            if key in self.support:
                raise ValueError(
                    f"Duplicate support record for {key}"
                )

            p10 = _decimal(entry["p10"], "p10")
            p90 = _decimal(entry["p90"], "p90")

            if p10 <= 0 or p90 < p10:
                raise ValueError(
                    f"Invalid support prices for {key}"
                )

            self.support[key] = {
                "p10": p10,
                "p90": p90,
                "training_rows": int(
                    entry["training_rows"]
                ),
                "price_levels": int(
                    entry["price_levels"]
                ),
            }

        # Product elasticities frozen during model training.
        self.elasticities = {}

        for product_upc, raw_value in config.get(
            "elasticities", {}
        ).items():

            epsilon = _decimal(
                raw_value,
                f"elasticity for {product_upc}",
            )

            if epsilon >= 0:
                raise ValueError(
                    "Elasticity must be negative "
                    f"for UPC {product_upc}"
                )

            self.elasticities[
                str(product_upc).strip()
            ] = float(epsilon)

    def candidates(
        self,
        *,
        upc: str,
        store: int,
        reference_price: float,
        promo_context: str = "UNMARKED",
    ) -> PriceCandidates:
        """Generate one-cent candidates within frozen guardrails."""

        # A. Validate request information.
        if (
            not isinstance(upc, str)
            or not upc.strip().isdigit()
        ):
            raise ValueError(
                "UPC must be a numeric identifier string"
            )

        if (
            isinstance(store, bool)
            or not isinstance(store, int)
            or store <= 0
        ):
            raise ValueError(
                "Store must be a positive integer"
            )

        ref = _decimal(
            reference_price,
            "Reference price",
        )

        if ref <= 0:
            raise ValueError(
                "Reference price must be positive"
            )

        if (
            not isinstance(promo_context, str)
            or not promo_context.strip()
        ):
            raise ValueError(
                "Promotion context must be a nonempty string"
            )

        promo = promo_context.strip().upper()

        # B. Check supported promotion context.
        if promo != self.supported_promo:
            raise PricingNotSupported(
                f"Promotion {promo!r} is unsupported; "
                f"expected {self.supported_promo!r}"
            )

        # C. Look up training-period support.
        key = (upc.strip(), store)
        support = self.support.get(key)

        if support is None:
            raise PricingNotSupported(
                f"No training support for UPC {key[0]} "
                f"at store {store}"
            )

        # D. Require adequate training evidence.
        if (
            support["training_rows"]
            < self.min_training_rows
        ):
            raise PricingNotSupported(
                "Insufficient training rows for this product-store"
            )

        if (
            support["price_levels"]
            < self.min_price_levels
        ):
            raise PricingNotSupported(
                "Insufficient historical price variation"
            )

        # E. Validate the reference price itself.
        p10 = support["p10"]
        p90 = support["p90"]

        if not p10 <= ref <= p90:
            raise PricingNotSupported(
                f"Reference price ${ref} is outside "
                f"training support ${p10}–${p90}"
            )

        # F. Intersect historical and change limits.
        fraction = (
            self.max_change_pct / Decimal("100")
        )

        low = max(
            p10,
            ref * (Decimal("1") - fraction),
        )

        high = min(
            p90,
            ref * (Decimal("1") + fraction),
        )

        # G. Convert to whole-cent boundaries.
        start = int(
            (low * 100).to_integral_value(
                rounding=ROUND_CEILING
            )
        )

        end = int(
            (high * 100).to_integral_value(
                rounding=ROUND_FLOOR
            )
        )

        if start > end:
            raise PricingNotSupported(
                "No whole-cent prices fall inside supported interval"
            )

        # H. Generate all eligible prices.
        prices = tuple(
            cents / 100
            for cents in range(start, end + 1)
        )

        return PriceCandidates(
            upc=key[0],
            store=store,
            reference_price=float(ref),
            lower_bound=float(low),
            upper_bound=float(high),
            prices=prices,
        )
