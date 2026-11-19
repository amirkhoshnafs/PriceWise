
import pandas as pd
import pytest
from pricewise.data.features import (
    FEATURE_COLUMNS,
    build_feature_table,
    build_future_feature_row,
)

from pricewise.data.features import (
    FEATURE_COLUMNS,
    build_feature_table,
)


def sample(
    weeks,
    units,
    prices=None,
    *,
    store=101,
):
    if prices is None:
        prices = [3.0] * len(weeks)

    return pd.DataFrame({
        "UPC": ["00123"] * len(weeks),
        "STORE": [store] * len(weeks),
        "WEEK": weeks,
        "MOVE": units,
        "UNIT_PRICE": prices,
        "SALE": [pd.NA] * len(weeks),
    })


def test_lags_rolling_mean_and_price_change():

    sales = sample(
        [1, 2, 3, 4, 5],
        [10, 20, 30, 40, 50],
        [2, 2, 2, 2, 3],
    )

    result = build_feature_table(sales)

    assert list(result["WEEK"]) == [5]

    row = result.iloc[0]

    assert row["LAG_1_UNITS"] == 40
    assert row["AVG_UNITS_PREV_4W"] == 25
    assert row["LAG_1_PRICE"] == 2

    assert row["PRICE_CHANGE_PCT"] == pytest.approx(50)
    assert row["PROMO_CONTEXT"] == "UNMARKED"

    assert set(FEATURE_COLUMNS).issubset(result.columns)


def test_missing_weeks_are_not_treated_as_consecutive():

    sales = sample(
        [1, 2, 3, 4, 6, 7, 8, 9, 10],
        [10, 20, 30, 40, 60, 70, 80, 90, 100],
    )

    result = build_feature_table(sales)

    assert list(result["WEEK"]) == [10]


def test_store_histories_are_independent():

    frame = pd.concat([
        sample(
            [1, 2, 3, 4, 5],
            [10, 20, 30, 40, 50],
            store=101,
        ),
        sample(
            [1, 2, 3, 4, 5],
            [100, 200, 300, 400, 500],
            store=102,
        ),
    ], ignore_index=True)

    result = (
        build_feature_table(frame)
        .set_index("STORE")
    )

    assert result.loc[101, "LAG_1_UNITS"] == 40
    assert result.loc[102, "LAG_1_UNITS"] == 400


def test_current_target_does_not_leak_into_features():

    original = sample(
        [1, 2, 3, 4, 5],
        [10, 20, 30, 40, 50],
    )

    revised = original.copy()

    revised.loc[
        revised["WEEK"] == 5, "MOVE"
    ] = 9999

    old = build_feature_table(original).iloc[0]
    new = build_feature_table(revised).iloc[0]

    historical = [
        "LAG_1_UNITS",
        "AVG_UNITS_PREV_4W",
        "LAG_1_PRICE",
        "PRICE_CHANGE_PCT",
    ]

    for feature in historical:
        assert old[feature] == new[feature]

    assert old["MOVE"] != new["MOVE"]


def test_duplicate_keys_raise_error():

    sales = sample(
        [1, 1],
        [10, 20],
    )

    with pytest.raises(ValueError, match="Duplicate"):
        build_feature_table(sales)


def test_future_matches_training_features():

    sales = sample(
        [1, 2, 3, 4, 5],
        [10, 20, 30, 40, 50],
        [2, 2, 2, 2, 3],
    )

    sales.loc[
        sales["WEEK"] == 5, "SALE"
    ] = "S"

    # Training-side features for week 5
    expected = build_feature_table(sales).iloc[0]

    # Production-side features for the same week
    actual = build_future_feature_row(
        sales,
        upc="00123",
        store=101,
        forecast_week=5,
        candidate_price=3.0,
        promo_context="S",
    ).iloc[0]

    for column in FEATURE_COLUMNS:
        if column in {"UPC", "STORE", "PROMO_CONTEXT"}:
            assert str(actual[column]) == str(expected[column])
        else:
            assert actual[column] == pytest.approx(
                float(expected[column])
            )


def test_future_never_reads_target_sales():

    sales = sample(
        [1, 2, 3, 4, 5],
        [10, 20, 30, 40, 50],
    )

    first = build_future_feature_row(
        sales,
        upc="00123",
        store=101,
        forecast_week=5,
        candidate_price=3.0,
    )

    # Change the future target deliberately.
    modified = sales.copy()
    modified.loc[
        modified["WEEK"] == 5, "MOVE"
    ] = 9999

    second = build_future_feature_row(
        modified,
        upc="00123",
        store=101,
        forecast_week=5,
        candidate_price=3.0,
    )

    pd.testing.assert_frame_equal(first, second)


def test_future_rejects_missing_calendar_week():

    sales = sample(
        [1, 2, 4, 5],
        [10, 20, 40, 50],
    )

    with pytest.raises(
        ValueError,
        match="four consecutive historical weeks",
    ):
        build_future_feature_row(
            sales,
            upc="00123",
            store=101,
            forecast_week=6,
            candidate_price=3.0,
        )


def test_future_store_histories_do_not_mix():

    sales = pd.concat([
        sample(
            [1, 2, 3, 4],
            [10, 20, 30, 40],
            store=101,
        ),
        sample(
            [1, 2, 3, 4],
            [100, 200, 300, 400],
            store=102,
        ),
    ], ignore_index=True)

    result = build_future_feature_row(
        sales,
        upc="00123",
        store=101,
        forecast_week=5,
        candidate_price=3.0,
    )

    assert result.loc[0, "LAG_1_UNITS"] == 40
    assert result.loc[0, "AVG_UNITS_PREV_4W"] == 25


def test_future_rejects_invalid_prices():

    sales = sample(
        [1, 2, 3, 4],
        [10, 20, 30, 40],
    )

    with pytest.raises(
        ValueError,
        match="Candidate price",
    ):
        build_future_feature_row(
            sales,
            upc="00123",
            store=101,
            forecast_week=5,
            candidate_price=0,
        )

    # A historical price must also be usable.
    sales.loc[
        sales["WEEK"] == 4, "UNIT_PRICE"
    ] = 0

    with pytest.raises(ValueError, match="prices positive"):
        build_future_feature_row(
            sales,
            upc="00123",
            store=101,
            forecast_week=5,
            candidate_price=3.0,
        )
