
"""PriceWise ONNX demand inference."""

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import onnxruntime as ort

from pricewise.data.features import FEATURE_COLUMNS


CATEGORICAL = {
    "UPC",
    "STORE",
    "PROMO_CONTEXT",
}


class OnnxDemandPredictor:
    """Serve the trained demand model using ONNX Runtime."""

    def __init__(
        self,
        model_path: str | Path,
        *,
        session: Any | None = None,
    ):
        # Load the ONNX graph once.
        if session is None:
            path = Path(model_path)

            if not path.is_file():
                raise FileNotFoundError(path)

            session = ort.InferenceSession(
                str(path),
                providers=["CPUExecutionProvider"],
            )

        self.session = session

        # Check that the loaded graph matches our
        # feature engineering contract.
        inputs = {
            item.name: item
            for item in self.session.get_inputs()
        }

        if set(inputs) != set(FEATURE_COLUMNS):
            raise ValueError(
                "ONNX inputs mismatch: "
                f"expected {set(FEATURE_COLUMNS)}, "
                f"received {set(inputs)}"
            )

        for name in FEATURE_COLUMNS:
            expected = (
                "tensor(string)"
                if name in CATEGORICAL
                else "tensor(float)"
            )

            if inputs[name].type != expected:
                raise ValueError(
                    f"{name}: expected {expected}, "
                    f"got {inputs[name].type}"
                )

    def predict(
        self,
        features: pd.DataFrame,
    ) -> np.ndarray:
        """Predict nonnegative weekly units."""

        if not isinstance(features, pd.DataFrame):
            raise TypeError(
                "Features must be a pandas DataFrame"
            )

        if features.columns.duplicated().any():
            raise ValueError(
                "Duplicate feature columns"
            )

        missing = (
            set(FEATURE_COLUMNS) - set(features.columns)
        )

        if missing:
            raise ValueError(
                f"Missing feature columns: {sorted(missing)}"
            )

        if features.empty:
            return np.empty(0, dtype=np.float64)

        feeds = {}

        # Convert pandas columns to ONNX input tensors.
        for name in FEATURE_COLUMNS:
            column = features[name]

            if column.isna().any():
                raise ValueError(
                    f"Missing values in {name}"
                )

            if name in CATEGORICAL:
                text = column.astype(str).str.strip()

                if name == "PROMO_CONTEXT":
                    text = text.str.upper()

                elif not text.str.fullmatch(r"\d+").all():
                    raise ValueError(
                        f"{name} must be a numeric "
                        "identifier string"
                    )

                if text.eq("").any():
                    raise ValueError(
                        f"Empty categorical values in {name}"
                    )

                feeds[name] = (
                    text.to_numpy(dtype=str).reshape(-1, 1)
                )

            else:
                values = pd.to_numeric(
                    column,
                    errors="raise",
                ).to_numpy(dtype=np.float64)

                if not np.isfinite(values).all():
                    raise ValueError(
                        f"Nonfinite numeric input in {name}"
                    )

                floats = values.astype(np.float32)

                if not np.isfinite(floats).all():
                    raise ValueError(
                        f"Float32 overflow in {name}"
                    )

                feeds[name] = floats.reshape(-1, 1)

        # Check consistency of related price features.
        candidate = (
            feeds["CANDIDATE_PRICE"]
            .ravel().astype(float)
        )

        previous = (
            feeds["LAG_1_PRICE"]
            .ravel().astype(float)
        )

        if (
            (candidate <= 0).any()
            or (previous <= 0).any()
        ):
            raise ValueError(
                "Prices must be positive"
            )

        expected_change = 100 * (
            candidate / previous - 1
        )

        given_change = (
            feeds["PRICE_CHANGE_PCT"].ravel()
        )

        if not np.allclose(
            given_change,
            expected_change,
            rtol=1e-5,
            atol=1e-3,
        ):
            raise ValueError(
                "PRICE_CHANGE_PCT disagrees with "
                "candidate and previous prices"
            )

        # Run the model.
        output = self.session.run(None, feeds)[0]

        predictions = np.asarray(
            output,
            dtype=np.float64,
        ).reshape(-1)

        if (
            len(predictions) != len(features)
            or not np.isfinite(predictions).all()
        ):
            raise ValueError(
                "ONNX produced invalid predictions"
            )

        # Weekly demand cannot be negative.
        return np.maximum(predictions, 0.0)
