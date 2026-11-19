
"""Phase 1.1: inspect raw cereal data without cleaning it."""

from __future__ import annotations

import argparse
from pathlib import Path
from zipfile import ZipFile

import pandas as pd


# 1. Locate the project and its raw data
ROOT = Path(__file__).resolve().parents[1]
RAW_DIR = ROOT / "data" / "raw"

REQUIRED_MOVEMENT = {
    "UPC", "STORE", "WEEK", "MOVE", "PRICE",
    "QTY", "PROFIT", "SALE", "OK"
}

REQUIRED_PRODUCT = {"UPC", "DESCRIP", "SIZE"}


# 2. Find our source files
def find_file(*names: str) -> Path:
    for name in names:
        candidate = RAW_DIR / name

        if candidate.is_file():
            return candidate

    raise FileNotFoundError(
        f"No file found among {names}. Check {RAW_DIR}"
    )


# 3. Load CSV or ZIP data

def load_csv(
    path: Path,
    limit: int | None = None
) -> pd.DataFrame:

    options = {
        "nrows": limit,
        "dtype": {
            "UPC": "string",
            "upc": "string"
        }
    }

    def read_with_fallback(source):
        try:
            return pd.read_csv(
                source,
                encoding="utf-8-sig",
                **options
            )

        except UnicodeDecodeError:
            if hasattr(source, "seek"):
                source.seek(0)

            print(
                f"UTF-8 failed for {path.name}. "
                "Trying Windows-1252."
            )

            return pd.read_csv(
                source,
                encoding="cp1252",
                **options
            )

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
                    f"{path.name} must contain exactly "
                    f"one CSV; found {members}"
                )

            with archive.open(members[0]) as file:
                data = read_with_fallback(file)

    else:
        data = read_with_fallback(path)

    data.columns = data.columns.str.strip().str.upper()

    if data.columns.duplicated().any():
        raise ValueError(
            f"{path.name} has duplicate column names"
        )

    if "UPC" not in data:
        raise ValueError(
            f"{path.name} is missing the UPC column"
        )

    data["UPC"] = (
        data["UPC"]
        .astype("string")
        .str.strip()
    )

    return data

# 4. Validate the expected schema
def require_columns(
    data: pd.DataFrame,
    required: set[str],
    name: str
) -> None:

    missing = sorted(required - set(data.columns))

    if missing:
        raise ValueError(
            f"{name}: missing expected columns {missing}"
        )


# 5. Run our dataset audit
def run_audit(limit: int | None = 50_000) -> None:

    movement_path = find_file("wcer.zip", "wcer.csv")
    product_path = find_file("upccer.csv", "upccer.zip")

    sales = load_csv(movement_path, limit=limit)
    products = load_csv(product_path)

    require_columns(
        sales, REQUIRED_MOVEMENT, movement_path.name
    )
    require_columns(
        products, REQUIRED_PRODUCT, product_path.name
    )

    # Convert numeric fields safely
    price = pd.to_numeric(sales["PRICE"], errors="coerce")
    units = pd.to_numeric(sales["MOVE"], errors="coerce")
    bundle = pd.to_numeric(sales["QTY"], errors="coerce")
    margin = pd.to_numeric(sales["PROFIT"], errors="coerce")
    ok_flag = pd.to_numeric(sales["OK"], errors="coerce")

    # Basic information
    print("\nPRICEWISE | PHASE 1.1 DATA AUDIT")
    print("=" * 46)

    print(f"Sales source:     {movement_path.name}")
    print(f"Catalog source:   {product_path.name}")
    print(f"Sales rows:       {len(sales):,}")
    print(f"Catalog rows:     {len(products):,}")
    print(f"Sales columns:    {', '.join(sales.columns)}")
    print(f"Catalog columns:  {', '.join(products.columns)}")

    # Coverage
    print("\nCOVERAGE (CHECKED SALES ROWS)")
    print(f"Unique UPCs:      {sales['UPC'].nunique():,}")
    print(f"Unique stores:    {sales['STORE'].nunique():,}")
    print(f"Unique weeks:     {sales['WEEK'].nunique():,}")

    # Data-quality diagnostics
    print("\nDATA QUALITY FLAGS (NOT REMOVED)")

    print(f"Missing prices:       {price.isna().sum():,}")
    print(f"Prices <= 0:         {(price <= 0).sum():,}")
    print(
        f"Invalid bundle size: "
        f"{(bundle.isna() | (bundle <= 0)).sum():,}"
    )
    print(f"Missing units:        {units.isna().sum():,}")
    print(f"Negative units:       {(units < 0).sum():,}")
    print(f"Zero units:           {(units == 0).sum():,}")
    print(f"Missing margin:       {margin.isna().sum():,}")
    print(
        f"Margin outside 0-100: "
        f"{((margin < 0) | (margin > 100)).sum():,}"
    )
    print(
        f"OK not equal to 1:    "
        f"{(ok_flag != 1).fillna(True).sum():,}"
    )

    print(
        f"Duplicate sale keys:  "
        f"{sales.duplicated(['UPC', 'STORE', 'WEEK']).sum():,}"
    )
    print(
        f"Unmatched catalog UPCs: "
        f"{(~sales['UPC'].isin(products['UPC'])).sum():,}"
    )
    print(
        f"Duplicate catalog UPCs: "
        f"{products.duplicated('UPC').sum():,}"
    )

    # Look at actual observations
    print("\nFIRST THREE SALES ROWS")

    display_cols = [
        "UPC", "STORE", "WEEK", "PRICE",
        "MOVE", "QTY", "PROFIT", "SALE", "OK"
    ]

    print(sales[display_cols].head(3).to_string(index=False))

    if limit is not None:
        print(
            "\nNOTE: first rows only, "
            "not a random or representative sample."
        )

    print("Raw files were not changed.\n")


# 6. Command-line entry point
def main() -> None:

    parser = argparse.ArgumentParser(
        description="Audit PriceWise cereal source data"
    )

    parser.add_argument(
        "--all",
        action="store_true",
        help="Scan the full movement file (high RAM use)"
    )

    parser.add_argument(
        "--rows",
        type=int,
        default=50_000,
        help="First N sales rows to scan"
    )

    args = parser.parse_args()

    if args.rows < 1:
        parser.error("--rows must be positive")

    try:
        run_audit(
            limit=None if args.all else args.rows
        )
    except (FileNotFoundError, ValueError) as exc:
        parser.exit(2, f"Audit failed: {exc}\n")


if __name__ == "__main__":
    main()
