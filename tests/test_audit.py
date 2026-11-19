
from pathlib import Path
from zipfile import ZipFile

import pandas as pd
import pytest

from scripts.audit_data import load_csv, require_columns


def test_preserve_upc_identifier(tmp_path: Path):
    path = tmp_path / "sample.csv"
    path.write_text(
        "UPC,STORE,WEEK\n00123,12,42\n",
        encoding="utf-8"
    )

    data = load_csv(path)

    assert data.loc[0, "UPC"] == "00123"


def test_windows_1252_encoding(tmp_path: Path):
    path = tmp_path / "products.csv"
    path.write_bytes(
        "UPC,DESCRIP\n00123,Crème Cereal\n".encode("cp1252")
    )

    data = load_csv(path)

    assert data.loc[0, "DESCRIP"] == "Crème Cereal"


def test_load_from_zip(tmp_path: Path):
    path = tmp_path / "movement.zip"

    with ZipFile(path, "w") as archive:
        archive.writestr(
            "movement.csv",
            "UPC,STORE,WEEK\n00123,12,42\n"
        )

    data = load_csv(path)

    assert len(data) == 1
    assert data.loc[0, "UPC"] == "00123"


def test_missing_required_column():
    data = pd.DataFrame({
        "UPC": ["00123"],
        "PRICE": [3.49]
    })

    with pytest.raises(ValueError, match="missing"):
        require_columns(
            data,
            {"UPC", "PRICE", "MOVE"},
            "test.csv"
        )
