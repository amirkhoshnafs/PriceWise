
"""Tests for PriceWise ONNX inference."""

import numpy as np
import pandas as pd
import pytest

from pricewise.ml.inference import OnnxDemandPredictor


FEATURE_NAMES = [
    "UPC", "STORE", "CANDIDATE_PRICE",
    "LAG_1_PRICE", "LAG_1_UNITS",
    "AVG_UNITS_PREV_4W", "PRICE_CHANGE_PCT",
    "WEEK_SIN", "WEEK_COS", "PROMO_CONTEXT",
]

CATEGORIES = {"UPC", "STORE", "PROMO_CONTEXT"}


class FakeInput:
    def __init__(self, name):
        self.name = name
        self.type = (
            "tensor(string)"
            if name in CATEGORIES
            else "tensor(float)"
        )


class FakeSession:
    def __init__(self, response=42.0):
        self.response = response
        self.last_feeds = None

    def get_inputs(self):
        return [
            FakeInput(name)
            for name in FEATURE_NAMES
        ]

    def run(self, output_names, feeds):
        self.last_feeds = feeds

        rows = len(feeds["UPC"])

        return [
            np.full(
                (rows, 1),
                self.response,
                dtype=np.float32,
            )
        ]


def valid_features():
    return pd.DataFrame([{
        "UPC": "1600066610",
        "STORE": 101,
        "CANDIDATE_PRICE": 3.20,
        "LAG_1_PRICE": 3.20,
        "LAG_1_UNITS": 45.0,
        "AVG_UNITS_PREV_4W": 48.0,
        "PRICE_CHANGE_PCT": 0.0,
        "WEEK_SIN": 0.0,
        "WEEK_COS": 1.0,
        "PROMO_CONTEXT": "unmarked",
    }])


def test_prediction_and_tensor_types():
    session = FakeSession(response=42.5)

    model = OnnxDemandPredictor(
        "unused.onnx",
        session=session,
    )

    predictions = model.predict(valid_features())

    assert predictions.tolist() == [42.5]

    assert (
        session.last_feeds["UPC"].dtype.kind == "U"
    )

    assert (
        session.last_feeds["STORE"][0, 0] == "101"
    )

    assert (
        session.last_feeds["CANDIDATE_PRICE"].dtype
        == np.float32
    )

    assert (
        session.last_feeds["PROMO_CONTEXT"][0, 0]
        == "UNMARKED"
    )


def test_negative_predictions_are_clipped():
    model = OnnxDemandPredictor(
        "unused.onnx",
        session=FakeSession(response=-5),
    )

    predictions = model.predict(valid_features())

    assert predictions.tolist() == [0.0]


def test_inconsistent_price_change_is_rejected():
    frame = valid_features()

    frame["CANDIDATE_PRICE"] = 3.52

    # We deliberately leave PRICE_CHANGE_PCT = 0.
    model = OnnxDemandPredictor(
        "unused.onnx",
        session=FakeSession(),
    )

    with pytest.raises(
        ValueError,
        match="PRICE_CHANGE_PCT",
    ):
        model.predict(frame)


def test_missing_features_are_rejected():
    frame = valid_features().drop(
        columns="LAG_1_UNITS"
    )

    model = OnnxDemandPredictor(
        "unused.onnx",
        session=FakeSession(),
    )

    with pytest.raises(
        ValueError,
        match="Missing feature columns",
    ):
        model.predict(frame)


def test_missing_values_are_rejected():
    frame = valid_features()

    frame.loc[0, "PROMO_CONTEXT"] = None

    model = OnnxDemandPredictor(
        "unused.onnx",
        session=FakeSession(),
    )

    with pytest.raises(
        ValueError,
        match="Missing values",
    ):
        model.predict(frame)
