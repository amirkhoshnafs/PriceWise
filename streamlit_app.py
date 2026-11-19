
"""PriceWise dashboard: API client and scenario visualization."""

import os

import pandas as pd
import plotly.graph_objects as go
import requests
import streamlit as st


# --------------------------------------------------
# A. Configuration
# --------------------------------------------------

API_URL = os.environ.get(
    "PRICEWISE_API_URL",
    "http://127.0.0.1:8000",
).rstrip("/")

PRODUCTS = {
    "1600066610": "Cheerios 15 oz",
    "3800000120": "Corn Flakes 18 oz",
}

STRATEGY_COLORS = {
    "Conservative": "#159A74",
    "Balanced": "#5865E8",
    "Aggressive": "#D99032",
}


# --------------------------------------------------
# B. HTTP client
# --------------------------------------------------

def fetch_scenarios(upc: str, store: int) -> dict:
    """Call the existing FastAPI pricing endpoint."""

    try:
        response = requests.post(
            f"{API_URL}/api/v1/pricing/scenarios",
            json={
                "upc": upc,
                "store": store,
                "promo_context": "UNMARKED",
            },
            timeout=20,
        )

    except requests.RequestException as exc:
        raise RuntimeError(
            "Cannot reach the PriceWise API. "
            "Start Uvicorn in the first terminal "
            "(port 8000)."
        ) from exc

    try:
        payload = response.json()
    except ValueError as exc:
        raise RuntimeError(
            "The API did not return valid JSON."
        ) from exc

    if response.status_code != 200:
        detail = (
            payload.get("detail", {})
            if isinstance(payload, dict)
            else {}
        )

        if isinstance(detail, dict):
            message = detail.get(
                "message",
                "Request could not be evaluated",
            )
        else:
            message = str(detail)

        raise RuntimeError(
            f"HTTP {response.status_code}: {message}"
        )

    expected = {
        "reference",
        "strategies",
        "scenarios",
    }

    if (
        not isinstance(payload, dict)
        or not expected.issubset(payload)
    ):
        raise RuntimeError(
            "The API returned an unexpected "
            "response structure."
        )

    return payload


# --------------------------------------------------
# C. Chart builder
# --------------------------------------------------

def build_chart(
    result: dict,
    scenarios: pd.DataFrame,
    metric: str,
    label: str,
):
    """Plot price scenarios and strategy selections."""

    figure = go.Figure()

    figure.add_trace(
        go.Scatter(
            x=scenarios["price"],
            y=scenarios[metric],
            mode="lines",
            name=label,
            line={
                "color": "#5968F2",
                "width": 3,
            },
            hovertemplate=(
                "Price: $%{x:.2f}"
                "<br>Value: %{y:.2f}"
                "<extra></extra>"
            ),
        )
    )

    # Mark the chosen strategy prices.
    for name in (
        "Conservative",
        "Balanced",
        "Aggressive",
    ):
        decision = result["strategies"].get(
            name, {}
        )

        if decision.get("status") != "scenario":
            continue

        figure.add_trace(
            go.Scatter(
                x=[decision["price"]],
                y=[decision[metric]],
                mode="markers",
                name=name,
                marker={
                    "size": 12,
                    "color": STRATEGY_COLORS[name],
                    "symbol": "diamond",
                },
                hovertemplate=(
                    f"{name}"
                    "<br>Price: $%{x:.2f}"
                    "<br>Value: %{y:.2f}"
                    "<extra></extra>"
                ),
            )
        )

    # Mark the current reference price.
    figure.add_vline(
        x=result["reference"]["price"],
        line_dash="dash",
        line_color="gray",
        annotation_text="Reference",
        annotation_position="top left",
    )

    figure.update_layout(
        xaxis_title="Candidate price ($)",
        yaxis_title=label,
        height=440,
        margin={
            "l": 20,
            "r": 20,
            "t": 40,
            "b": 20,
        },
        legend={
            "orientation": "h",
            "y": 1.16,
            "x": 0,
        },
        hovermode="closest",
    )

    return figure


# --------------------------------------------------
# D. Page configuration
# --------------------------------------------------

st.set_page_config(
    page_title="PriceWise | Retail Pricing Intelligence",
    page_icon="📈",
    layout="wide",
)

st.title("PriceWise")
st.caption(
    "Retail Pricing Intelligence · "
    "Demand and margin scenarios"
)


# --------------------------------------------------
# E. Sidebar controls
# --------------------------------------------------

with st.sidebar:
    st.header("Scenario settings")

    with st.form("pricing_request"):

        selected_upc = st.selectbox(
            "Product",
            options=list(PRODUCTS),
            format_func=lambda upc: PRODUCTS[upc],
        )

        selected_store = st.number_input(
            "Store ID",
            min_value=1,
            value=101,
            step=1,
        )

        st.caption(
            "Promotion context: UNMARKED "
            "(current supported policy)"
        )

        submitted = st.form_submit_button(
            "Evaluate prices",
            type="primary",
            width="stretch",
        )

    st.divider()
    st.caption(
        "The API must be running locally "
        "on port 8000."
    )


# --------------------------------------------------
# F. Fetch and preserve the last result
# --------------------------------------------------

if submitted:

    # Clear previous results before a new request.
    st.session_state["pricewise_result"] = None
    st.session_state["pricewise_error"] = None

    try:
        with st.spinner(
            "Evaluating pricing scenarios..."
        ):
            st.session_state["pricewise_result"] = (
                fetch_scenarios(
                    selected_upc,
                    int(selected_store),
                )
            )

    except RuntimeError as exc:
        st.session_state["pricewise_error"] = str(exc)


error = st.session_state.get(
    "pricewise_error"
)

if error:
    if error.startswith(("HTTP 404", "HTTP 409")):
        st.warning(error)
    else:
        st.error(error)

    st.stop()


result = st.session_state.get(
    "pricewise_result"
)

if result is None:
    st.info(
        "Choose a product and store, "
        "then click **Evaluate prices**."
    )
    st.stop()


# --------------------------------------------------
# G. Reference forecast
# --------------------------------------------------

st.subheader(
    f"{result['product']} · Store {result['store']}"
)

st.caption(
    f"Historical forecast week "
    f"{result['forecast_week']} "
    f"(last observed: "
    f"{result['last_observed_week']}). "
    "This is not a current-market forecast."
)

reference = result["reference"]

summary_cols = st.columns(4)

summary_cols[0].metric(
    "Reference price",
    f"${reference['price']:.2f}",
)

summary_cols[1].metric(
    "Predicted weekly demand",
    f"{reference['predicted_units']:.1f} units",
)

summary_cols[2].metric(
    "Unit acquisition-cost proxy",
    f"${reference['unit_cost_proxy']:.3f}",
)

summary_cols[3].metric(
    "Reference modeled gross profit",
    f"${reference['modeled_gross_profit']:.2f}",
)


# --------------------------------------------------
# H. Three strategy cards
# --------------------------------------------------

st.subheader("Pricing strategies")

strategy_cols = st.columns(
    3,
    gap="medium",
)

for col, name in zip(
    strategy_cols,
    STRATEGY_COLORS,
):
    decision = result["strategies"].get(name)

    with col:
        with st.container(border=True):

            st.markdown(f"**{name}**")

            if (
                not decision
                or decision["status"] != "scenario"
            ):
                reason = (decision or {}).get(
                    "explanation",
                    "No scenario available",
                )

                st.warning(reason)
                continue

            st.metric(
                "Proposed price",
                f"${decision['price']:.2f}",
                delta=(
                    f"{decision['price_change_pct']:+.1f}% "
                    "vs reference"
                ),
                delta_color="off",
            )

            st.metric(
                "Modeled weekly gross profit",
                f"${decision['gross_profit']:.2f}",
                delta=(
                    f"{decision['profit_change_pct']:+.1f}% "
                    "vs reference"
                ),
            )

            st.caption(
                "Predicted demand: "
                f"{decision['predicted_units']:.1f} "
                "units "
                f"({decision['units_change_pct']:+.1f}%)"
            )

            st.caption(
                decision["explanation"]
            )


# --------------------------------------------------
# I. Interactive charts and scenario table
# --------------------------------------------------

scenarios = (
    pd.DataFrame(result["scenarios"])
    .sort_values("price")
)

st.subheader("Price-response analysis")

profit_tab, demand_tab, table_tab = st.tabs([
    "Gross profit",
    "Demand",
    "All price scenarios",
])

with profit_tab:
    st.plotly_chart(
        build_chart(
            result,
            scenarios,
            "gross_profit",
            "Modeled gross profit ($)",
        ),
        width="stretch",
        key="profit_chart",
    )

with demand_tab:
    st.plotly_chart(
        build_chart(
            result,
            scenarios,
            "predicted_units",
            "Predicted weekly units",
        ),
        width="stretch",
        key="demand_chart",
    )

with table_tab:

    visible = {
        "price": "Price ($)",
        "price_change_pct": "Price change (%)",
        "predicted_units": "Predicted units",
        "revenue": "Modeled revenue ($)",
        "gross_profit": "Modeled gross profit ($)",
        "profit_delta_vs_reference": (
            "Profit difference ($)"
        ),
    }

    st.dataframe(
        scenarios[list(visible)]
        .rename(columns=visible)
        .round(2),
        hide_index=True,
        width="stretch",
    )

    st.download_button(
        "Download scenario data (CSV)",
        data=(
            scenarios.to_csv(index=False)
            .encode("utf-8")
        ),
        file_name=(
            f"pricewise_{result['upc']}"
            f"_store{result['store']}"
            f"_week{result['forecast_week']}.csv"
        ),
        mime="text/csv",
    )


# --------------------------------------------------
# J. Business limitation
# --------------------------------------------------

st.divider()
st.caption(result["disclaimer"])
