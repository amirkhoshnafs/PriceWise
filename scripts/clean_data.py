
"""Phase 1.2: prepare auditable, priced cereal sales data."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from zipfile import ZipFile

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

from scripts.audit_data import (
    ROOT,
    find_file,
    load_csv,
    require_columns,
    REQUIRED_MOVEMENT,
)


# --------------------------------------------------
# 1. OUTPUT PATHS
# --------------------------------------------------

OUTPUT = (
    ROOT / "data" / "processed" / "cereal_priced.parquet"
)

REPORT = ROOT / "reports" / "cleaning_report.json"


# --------------------------------------------------
# 2. STREAM THE MOVEMENT DATA
# --------------------------------------------------

def movement_chunks(path: Path, chunk_size: int):
    """Read movement records in manageable batches."""

    options = {
        "chunksize": chunk_size,
        "dtype": {
            "UPC": "string",
            "upc": "string",
        },
        "low_memory": False,
    }

    if path.suffix.lower() == ".zip":

        with ZipFile(path) as archive:

            members = [
                name
                for name in archive.namelist()
                if name.lower().endswith(".csv")
                and not name.endswith("/")
            ]

            if len(members) != 1:
                raise ValueError(
                    f"Expected one CSV in {path.name}; "
                    f"found {members}"
                )

            with archive.open(members[0]) as stream:
                yield from pd.read_csv(stream, **options)

    else:
        yield from pd.read_csv(path, **options)


# --------------------------------------------------
# 3. CLEAN ONE DATA CHUNK
# --------------------------------------------------

def clean_chunk(
    raw: pd.DataFrame,
) -> tuple[pd.DataFrame, dict[str, int]]:

    df = raw.copy()

    df.columns = df.columns.str.strip().str.upper()

    require_columns(
        df,
        REQUIRED_MOVEMENT,
        "movement chunk",
    )

    counts = {"input_rows": len(df)}

    df["UPC"] = df["UPC"].astype("string").str.strip()

    numeric_columns = (
        "STORE", "WEEK", "OK", "PRICE",
        "MOVE", "QTY", "PROFIT",
    )

    for column in numeric_columns:
        df[column] = pd.to_numeric(
            df[column],
            errors="coerce",
        )

    # One row can be excluded by only one stage.
    def accept(mask: pd.Series, reason: str) -> None:
        nonlocal df

        mask = mask.fillna(False).astype(bool)

        counts[reason] = int((~mask).sum())

        df = df.loc[mask].copy()

    # Rule A: valid product, store, and week keys
    keys_valid = (
        df["UPC"].notna()
        & df["UPC"].ne("")
        & np.isfinite(df["STORE"])
        & np.isfinite(df["WEEK"])
        & (df["STORE"] % 1 == 0)
        & (df["WEEK"] % 1 == 0)
    )

    accept(keys_valid, "invalid_keys")

    # Rule B: source quality flag
    accept(df["OK"] == 1, "suspect_ok")

    # Rule C: required numeric fields
    numeric_valid = np.isfinite(
        df[["PRICE", "MOVE", "QTY", "PROFIT"]]
    ).all(axis=1)

    accept(
        numeric_valid,
        "missing_or_nonfinite_numbers",
    )

    # Rule D: usable recorded selling price
    accept(
        df["PRICE"] > 0,
        "nonpositive_price",
    )

    # Rule E: valid positive integer unit count
    accept(
        (df["MOVE"] > 0)
        & (df["MOVE"] % 1 == 0),
        "invalid_units",
    )

    # Rule F: valid positive bundle quantity
    accept(
        (df["QTY"] > 0)
        & (df["QTY"] % 1 == 0),
        "invalid_bundle",
    )

    # Negative margins are allowed.
    # Gross margins above 100% are not.
    accept(
        df["PROFIT"] <= 100,
        "margin_above_100",
    )

    # Make column types predictable for Parquet.
    for column in ("STORE", "WEEK", "MOVE", "QTY"):
        df[column] = df[column].astype("int64")

    for column in ("PRICE", "PROFIT"):
        df[column] = df[column].astype("float64")

    df["SALE"] = df["SALE"].astype("string")

    # Business calculations
    df["UNIT_PRICE"] = df["PRICE"] / df["QTY"]

    df["REVENUE"] = (
        df["UNIT_PRICE"] * df["MOVE"]
    )

    df["UNIT_AAC_PROXY"] = (
        df["UNIT_PRICE"]
        * (1 - df["PROFIT"] / 100)
    )

    df["GROSS_PROFIT_PROXY"] = (
        df["REVENUE"]
        * df["PROFIT"] / 100
    )

    counts["kept_rows"] = len(df)

    fields = [
        "UPC",
        "STORE",
        "WEEK",
        "MOVE",
        "QTY",
        "PRICE",
        "SALE",
        "PROFIT",
        "UNIT_PRICE",
        "REVENUE",
        "UNIT_AAC_PROXY",
        "GROSS_PROFIT_PROXY",
    ]

    return df[fields], counts


# --------------------------------------------------
# 4. EXECUTE THE FULL PIPELINE
# --------------------------------------------------

def run_cleaning(
    chunk_size: int = 200_000,
) -> dict:

    movement_file = find_file("wcer.zip", "wcer.csv")
    catalog_file = find_file("upccer.csv", "upccer.zip")

    products = load_csv(catalog_file)

    require_columns(
        products,
        {"UPC", "DESCRIP", "SIZE"},
        "product catalog",
    )

    if products["UPC"].duplicated().any():
        raise ValueError(
            "Product catalog contains duplicate UPCs"
        )

    products = products[
        ["UPC", "DESCRIP", "SIZE"]
    ].copy()

    for field in ("UPC", "DESCRIP", "SIZE"):
        products[field] = (
            products[field].astype("string")
        )

    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    REPORT.parent.mkdir(parents=True, exist_ok=True)

    # Avoid overwriting a valid dataset mid-run.
    temp_path = OUTPUT.with_suffix(".partial.parquet")
    temp_path.unlink(missing_ok=True)

    totals = Counter()
    writer = None

    try:
        for index, raw_chunk in enumerate(
            movement_chunks(movement_file, chunk_size),
            start=1,
        ):

            cleaned, counts = clean_chunk(raw_chunk)
            totals.update(counts)

            if cleaned.empty:
                continue

            # Attach product names and sizes.
            joined = cleaned.merge(
                products,
                how="left",
                on="UPC",
                validate="many_to_one",
                indicator=True,
            )

            if (joined["_merge"] != "both").any():
                missing = joined.loc[
                    joined["_merge"] != "both",
                    "UPC",
                ].unique()[:5]

                raise ValueError(
                    f"Sales UPCs missing in catalog: "
                    f"{list(missing)}"
                )

            joined = joined.drop(columns="_merge")

            # Write each processed chunk to Parquet.
            arrow_table = pa.Table.from_pandas(
                joined,
                preserve_index=False,
            )

            if writer is None:
                writer = pq.ParquetWriter(
                    temp_path,
                    arrow_table.schema,
                    compression="snappy",
                )
            else:
                arrow_table = arrow_table.cast(writer.schema)

            writer.write_table(arrow_table)

            print(
                f"Chunk {index:>2}: "
                f"{len(raw_chunk):,} read; "
                f"{len(joined):,} kept"
            )

    finally:
        if writer is not None:
            writer.close()

    if writer is None:
        raise ValueError(
            "No usable rows found; output not created"
        )

    reasons = (
        "invalid_keys",
        "suspect_ok",
        "missing_or_nonfinite_numbers",
        "nonpositive_price",
        "invalid_units",
        "invalid_bundle",
        "margin_above_100",
    )

    excluded = sum(totals[key] for key in reasons)

    if totals["input_rows"] != totals["kept_rows"] + excluded:
        raise AssertionError(
            "Row reconciliation failed"
        )

    # Publish the completed Parquet file.
    temp_path.replace(OUTPUT)

    report = {
        "raw_rows": totals["input_rows"],
        "clean_rows": totals["kept_rows"],
        "retained_pct": round(
            100 * totals["kept_rows"] / totals["input_rows"],
            2,
        ),
        "excluded_by_stage": {
            key: totals[key] for key in reasons
        },
        "output": str(OUTPUT.relative_to(ROOT)),
        "note": (
            "Positive-price, positive-sales subset. "
            "Zero-demand weeks are not modeled. "
            "Unit cost is derived from historical AAC "
            "margin, not replacement cost."
        ),
    }

    REPORT.write_text(
        json.dumps(report, indent=2),
        encoding="utf-8",
    )

    print("\nPRICEWISE | CLEANING FINISHED")
    print(f"Rows read: {report['raw_rows']:,}")

    for label, count in report["excluded_by_stage"].items():
        print(f"Excluded ({label}): {count:,}")

    print(
        f"Rows kept: {report['clean_rows']:,} "
        f"({report['retained_pct']}%)"
    )

    print(f"Dataset: {OUTPUT.relative_to(ROOT)}")
    print(f"Report:  {REPORT.relative_to(ROOT)}")

    return report


# --------------------------------------------------
# 5. COMMAND-LINE ENTRY POINT
# --------------------------------------------------

def main() -> None:

    parser = argparse.ArgumentParser(
        description="Prepare PriceWise priced-sales data"
    )

    parser.add_argument(
        "--chunk-size",
        type=int,
        default=200_000,
    )

    args = parser.parse_args()

    if args.chunk_size < 1:
        parser.error("--chunk-size must be positive")

    run_cleaning(args.chunk_size)


if __name__ == "__main__":
    main()
