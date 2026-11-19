
import pandas as pd

from scripts.clean_data import clean_chunk


def test_clean_and_reconcile():

    raw = pd.DataFrame([
        {
            "UPC": "001",
            "STORE": 12,
            "WEEK": 42,
            "OK": 1,
            "PRICE": 3.0,
            "MOVE": 120,
            "QTY": 3,
            "PROFIT": 25,
            "SALE": "S",
        },
        {
            "UPC": "001",
            "STORE": 12,
            "WEEK": 43,
            "OK": 0,
            "PRICE": 3.0,
            "MOVE": 12,
            "QTY": 3,
            "PROFIT": 25,
            "SALE": None,
        },
        {
            "UPC": "001",
            "STORE": 12,
            "WEEK": 44,
            "OK": 1,
            "PRICE": 0.0,
            "MOVE": 0,
            "QTY": 1,
            "PROFIT": 0,
            "SALE": None,
        },
        {
            "UPC": "002",
            "STORE": 12,
            "WEEK": 45,
            "OK": 1,
            "PRICE": 2.5,
            "MOVE": 10,
            "QTY": 1,
            "PROFIT": -20,
            "SALE": None,
        },
        {
            "UPC": "002",
            "STORE": 12,
            "WEEK": 46,
            "OK": 1,
            "PRICE": 0.0,
            "MOVE": 10,
            "QTY": 1,
            "PROFIT": 10,
            "SALE": None,
        },
    ])

    cleaned, counts = clean_chunk(raw)

    assert counts["input_rows"] == 5
    assert counts["suspect_ok"] == 1
    assert counts["nonpositive_price"] == 2
    assert counts["kept_rows"] == 2

    assert len(cleaned) == 2

    # Three-for-$3 bundle:
    assert cleaned.iloc[0]["UNIT_PRICE"] == 1.0
    assert cleaned.iloc[0]["REVENUE"] == 120.0
    assert cleaned.iloc[0]["GROSS_PROFIT_PROXY"] == 30.0
    assert cleaned.iloc[0]["UNIT_AAC_PROXY"] == 0.75

    # Negative margin is retained.
    assert cleaned.iloc[1]["PROFIT"] == -20
    assert cleaned.iloc[1]["UNIT_AAC_PROXY"] == 3.0
