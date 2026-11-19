
"""PriceWise: one-week-ahead demand feature engineering."""

import numpy as np
import pandas as pd


# Features shared by training and future inference.
FEATURE_COLUMNS = [
    "UPC",
    "STORE",
    "CANDIDATE_PRICE",
    "LAG_1_PRICE",
    "LAG_1_UNITS",
    "AVG_UNITS_PREV_4W",
    "PRICE_CHANGE_PCT",
    "WEEK_SIN",
    "WEEK_COS",
    "PROMO_CONTEXT",
]

TARGET_COLUMN = "MOVE"


def build_feature_table(
    sales: pd.DataFrame,
) -> pd.DataFrame:
    """
    Build training observations using prior calendar weeks.

    Expects cleaned, positive-price and positive-sales
    records at the product-store-week level.
    """

    # 1. Validate input columns.
    required = {
        "UPC", "STORE", "WEEK",
        "MOVE", "UNIT_PRICE", "SALE",
    }

    missing = sorted(required - set(sales.columns))

    if missing:
        raise ValueError(
            f"Missing required columns: {missing}"
        )

    data = sales.copy()

    data["UPC"] = (
        data["UPC"].astype("string").str.strip()
    )

    essential = [
        "UPC", "STORE", "WEEK",
        "MOVE", "UNIT_PRICE",
    ]

    if data[essential].isna().any().any():
        raise ValueError(
            "Missing product, store, week, price, or sales"
        )

    if (
        (data["UNIT_PRICE"] <= 0).any()
        or (data["MOVE"] <= 0).any()
    ):
        raise ValueError(
            "Expected positive-price, positive-sales data"
        )

    keys = ["UPC", "STORE", "WEEK"]

    if data.duplicated(keys).any():
        raise ValueError(
            "Duplicate product-store-week keys"
        )

    # 2. Sort chronologically within each product/store.
    data = data.sort_values(keys).copy()

    data["PROMO_CONTEXT"] = (
        data["SALE"]
        .astype("string")
        .fillna("UNMARKED")
        .str.strip()
        .str.upper()
        .replace("", "UNMARKED")
    )

    groups = data.groupby(
        ["UPC", "STORE"],
        sort=False,
    )

    # 3. Historical features.
    data["PREV_WEEK"] = (
        groups["WEEK"].shift(1)
    )

    data["WEEK_4_BACK"] = (
        groups["WEEK"].shift(4)
    )

    data["LAG_1_UNITS"] = (
        groups["MOVE"].shift(1)
    )

    data["LAG_1_PRICE"] = (
        groups["UNIT_PRICE"].shift(1)
    )

    data["AVG_UNITS_PREV_4W"] = (
        groups["MOVE"].transform(
            lambda values: (
                values.shift(1)
                .rolling(4, min_periods=4)
                .mean()
            )
        )
    )

    # 4. Observed price as training candidate.
    data["CANDIDATE_PRICE"] = (
        data["UNIT_PRICE"]
    )

    data["PRICE_CHANGE_PCT"] = 100 * (
        data["CANDIDATE_PRICE"]
        / data["LAG_1_PRICE"]
        - 1
    )

    # 5. Approximate annual seasonality.
    cycle = (data["WEEK"] - 1) % 52

    data["WEEK_SIN"] = np.sin(
        2 * np.pi * cycle / 52
    )

    data["WEEK_COS"] = np.cos(
        2 * np.pi * cycle / 52
    )

    # 6. Require four consecutive previous calendar weeks.
    has_four_consecutive_weeks = (
        data["WEEK"] - data["WEEK_4_BACK"]
    ).eq(4)

    return data.loc[
        has_four_consecutive_weeks
    ].copy()



def build_future_feature_row(
    sales: pd.DataFrame,
    *,
    upc: str,
    store: int,
    forecast_week: int,
    candidate_price: float,
    promo_context: str = "UNMARKED",
) -> pd.DataFrame:
    """
    Build one future-week ONNX feature row.

    Uses exactly the four calendar weeks preceding
    forecast_week. Never reads the forecast target.
    """

    # A. Validate the function inputs
    required = {
        "UPC", "STORE", "WEEK", "MOVE", "UNIT_PRICE"
    }

    missing = sorted(required - set(sales.columns))

    if missing:
        raise ValueError(
            f"Missing historical columns: {missing}"
        )

    if (
        not isinstance(upc, str)
        or not upc.strip().isdigit()
    ):
        raise ValueError(
            "UPC must be a numeric identifier string"
        )

    upc = upc.strip()

    if (
        isinstance(store, (bool, np.bool_))
        or not isinstance(store, (int, np.integer))
        or store <= 0
    ):
        raise ValueError(
            "Store must be a positive integer"
        )

    if (
        isinstance(forecast_week, (bool, np.bool_))
        or not isinstance(
            forecast_week, (int, np.integer)
        )
        or forecast_week <= 4
    ):
        raise ValueError(
            "Forecast week must be an integer greater than 4"
        )

    try:
        price = float(candidate_price)
    except (TypeError, ValueError) as exc:
        raise ValueError(
            "Candidate price must be numeric"
        ) from exc

    if (
        isinstance(candidate_price, (bool, np.bool_))
        or not np.isfinite(price)
        or price <= 0
    ):
        raise ValueError(
            "Candidate price must be finite and positive"
        )

    if (
        not isinstance(promo_context, str)
        or not promo_context.strip()
    ):
        raise ValueError(
            "Promotion context must be a nonempty string"
        )

    promo = promo_context.strip().upper()

    # B. Select the requested product and store
    upcs = (
        sales["UPC"].astype("string").str.strip()
    )

    stores = pd.to_numeric(
        sales["STORE"], errors="coerce"
    )

    weeks = pd.to_numeric(
        sales["WEEK"], errors="coerce"
    )

    # C. Require the exact four previous weeks
    expected_weeks = list(
        range(forecast_week - 4, forecast_week)
    )

    mask = (
        upcs.eq(upc).fillna(False)
        & stores.eq(store)
        & weeks.isin(expected_weeks)
    )

    history = sales.loc[
        mask,
        ["WEEK", "MOVE", "UNIT_PRICE"],
    ].copy()

    history["WEEK"] = pd.to_numeric(
        history["WEEK"], errors="coerce"
    )

    history = history.sort_values("WEEK")

    if history["WEEK"].tolist() != expected_weeks:
        raise ValueError(
            "Need exactly four consecutive historical "
            f"weeks {expected_weeks} for UPC {upc}, "
            f"store {store}"
        )

    # D. Validate historical numeric values
    numeric = history[
        ["MOVE", "UNIT_PRICE"]
    ].apply(
        pd.to_numeric,
        errors="coerce",
    )

    if not np.isfinite(
        numeric.to_numpy(dtype=float)
    ).all():
        raise ValueError(
            "Historical units and prices must be finite"
        )

    if (
        (numeric <= 0).any().any()
        or (numeric["MOVE"] % 1 != 0).any()
    ):
        raise ValueError(
            "Historical units must be positive integers "
            "and prices positive"
        )

    # E. Historical features
    last_price = float(
        numeric["UNIT_PRICE"].iloc[-1]
    )

    last_units = float(
        numeric["MOVE"].iloc[-1]
    )

    average_units = float(
        numeric["MOVE"].mean()
    )

    # F. Seasonal features for the forecast week
    position = (forecast_week - 1) % 52

    # G. Construct the feature row
    row = {
        "UPC": upc,
        "STORE": str(store),
        "CANDIDATE_PRICE": price,
        "LAG_1_PRICE": last_price,
        "LAG_1_UNITS": last_units,
        "AVG_UNITS_PREV_4W": average_units,
        "PRICE_CHANGE_PCT": 100 * (
            price / last_price - 1
        ),
        "WEEK_SIN": float(
            np.sin(2 * np.pi * position / 52)
        ),
        "WEEK_COS": float(
            np.cos(2 * np.pi * position / 52)
        ),
        "PROMO_CONTEXT": promo,
    }

    return pd.DataFrame(
        [row],
        columns=FEATURE_COLUMNS,
    )
