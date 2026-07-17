import os
import base64
from typing import Any

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import requests
import streamlit as st
import textwrap

# =============================================================================
# Configuration
# =============================================================================

API_URL = os.environ["API_URL"].rstrip("/")

ID_COLUMNS = {
    "SK_ID_CURR",
    "SK_ID_BUREAU",
    "SK_ID_PREV",
}

st.set_page_config(
    page_title="Credit Scoring",
    page_icon="🏦",
    layout="wide",
)


# =============================================================================
# Style
# =============================================================================

st.markdown(
    textwrap.dedent(
        """
        <style>
            html,
            body,
            [class*="css"] {
                font-family: "DM Sans", sans-serif;
                background-color: #0d1117;
                color: #c9d1d9;
            }
            .title {
                font-family: "DM Mono", monospace;
                font-size: 2rem;
                font-weight: 500;
                color: #ffffff;
                letter-spacing: -0.02em;
                margin-bottom: 0.2rem;
            }
            .subtitle {
                font-size: 0.9rem;
                color: #8b949e;
                margin-bottom: 2rem;
            }
            .result-card {
                background: #161b22;
                border: 1px solid #30363d;
                border-radius: 12px;
                padding: 1.5rem;
                margin-bottom: 1rem;
                box-shadow: 0 4px 15px rgba(0, 0, 0, 0.45);
            }
            .approved {
                border-left: 4px solid #2ea043;
            }
            .rejected {
                border-left: 4px solid #f85149;
            }
            .metric-label {
                font-family: "DM Mono", monospace;
                font-size: 0.75rem;
                color: #8b949e;
                text-transform: uppercase;
                letter-spacing: 0.08em;
            }
            .metric-value {
                font-size: 2rem;
                font-weight: 600;
                color: #ffffff;
            }
            .status-approved {
                color: #2ea043;
                font-size: 1.4rem;
                font-weight: 600;
            }
            .status-rejected {
                color: #f85149;
                font-size: 1.4rem;
                font-weight: 600;
            }
            .section-title {
                font-family: "DM Mono", monospace;
                font-size: 0.8rem;
                font-weight: 400;
                color: #8b949e;
                text-transform: uppercase;
                letter-spacing: 0.1em;
                margin-top: 0;
                margin-bottom: 1rem;
                padding-bottom: 0.5rem;
                border-bottom: 1px solid #30363d;
            }
            .simulation-box {
                background: #161b22;
                border: 1px solid #30363d;
                border-radius: 12px;
                padding: 1.25rem;
                margin-bottom: 1rem;
            }
            div[data-testid="stSidebar"] {
                background-color: #0d1117 !important;
            }
        </style>
        """
    ),
    unsafe_allow_html=True,
)


# =============================================================================
# HTTP helpers
# =============================================================================

def check_response(response: requests.Response, action: str) -> dict:
    """Validate an API response and return its JSON payload."""
    if response.status_code == 404:
        raise ValueError("Customer ID not found.")

    if response.status_code != 200:
        raise RuntimeError(
            f"{action} failed — status {response.status_code}: "
            f"{response.text}"
        )

    try:
        return response.json()
    except ValueError as exc:
        raise RuntimeError(
            f"{action} returned invalid JSON."
        ) from exc


def get_prediction(sk_id: str) -> dict:
    """Get the original prediction."""
    response = requests.get(
        f"{API_URL}/predict/{sk_id}",
        timeout=30,
    )

    return check_response(response, "Prediction")


def fetch_customer_data(sk_id: str) -> dict:
    """Fetch the customer's raw data."""
    response = requests.get(
        f"{API_URL}/explore/{sk_id}",
        timeout=30,
    )

    return check_response(response, "Customer data loading")


def get_simulated_prediction(
    sk_id: str,
    table: str,
    column: str,
    value: Any,
) -> dict:
    """
    Run a simulated prediction with modified raw-data features.

    Expected request body:

    {
        "overrides": {
            "application": {
                "AMT_CREDIT": 250000
            }
        }
    }
    """
    payload = {
        "overrides": {
            table: {
                column: value,
            }
        }
    }

    response = requests.post(
        f"{API_URL}/custom_predict/{sk_id}",
        json=payload,
        timeout=30,
    )

    return check_response(response, "Simulated prediction")


@st.cache_data(show_spinner=False, ttl=3600)
def fetch_categorical_values(table: str, column: str) -> list:
    """
    Fetch possible values for one categorical column.

    Expected endpoint:

    GET /feature-values/{table}?column={column}

    Expected response:

    {
        "values": ["Cash loans", "Revolving loans"]
    }
    """
    try:
        response = requests.get(
            f"{API_URL}/feature-values/{table}",
            params={"column": column},
            timeout=15,
        )

        if response.status_code != 200:
            return []

        values = response.json().get("values", [])

        return list(dict.fromkeys(values))

    except (
        requests.exceptions.RequestException,
        ValueError,
        TypeError,
    ):
        return []


# =============================================================================
# Data helpers
# =============================================================================

def is_numeric(value: Any) -> bool:
    """Check whether a value is numeric, excluding booleans."""
    return (
        value is not None
        and not isinstance(value, (bool, np.bool_))
        and isinstance(
            value,
            (int, float, np.integer, np.floating),
        )
    )


def format_value(value: Any) -> str:
    """Format a value for display."""
    if value is None:
        return "Missing value"

    if isinstance(value, (float, np.floating)):
        return f"{value:,.6g}"

    return str(value)


def find_value(nested_data: dict, column: str) -> Any:
    """Find a column value across all raw-data tables."""
    for table_data in nested_data.values():
        if isinstance(table_data, dict) and column in table_data:
            return table_data[column]

    return None


def flatten_customer_data(nested_data: dict) -> dict:
    """
    Flatten the raw nested data into editable feature entries.

    Output example:

    {
        "application · AMT_CREDIT": {
            "table": "application",
            "column": "AMT_CREDIT",
            "value": 250000
        }
    }
    """
    features = {}

    for table_name, table_data in nested_data.items():
        if not isinstance(table_data, dict):
            continue

        for column_name, value in table_data.items():
            if column_name in ID_COLUMNS:
                continue

            label = f"{table_name} · {column_name}"

            features[label] = {
                "table": table_name,
                "column": column_name,
                "value": value,
            }

    return features


def numeric_columns(table_data: dict) -> list:
    """Return editable numeric columns from one raw-data table."""
    return sorted(
        column
        for column, value in table_data.items()
        if column not in ID_COLUMNS
        and is_numeric(value)
    )


# =============================================================================
# Rendering helpers
# =============================================================================

def render_section_title(title: str) -> None:
    """Render a consistent section heading."""
    st.markdown(
        f'<h3 class="section-title">{title}</h3>',
        unsafe_allow_html=True,
    )


def render_key_metrics(nested_data: dict) -> None:
    """Render the four main customer metrics."""
    days_birth = find_value(nested_data, "DAYS_BIRTH")
    income = find_value(nested_data, "AMT_INCOME_TOTAL")
    credit = find_value(nested_data, "AMT_CREDIT")
    annuity = find_value(nested_data, "AMT_ANNUITY")

    age = (
        int(-days_birth / 365)
        if is_numeric(days_birth)
        else "N/A"
    )

    income_text = (
        f"${income:,.0f}"
        if is_numeric(income)
        else "N/A"
    )

    credit_text = (
        f"${credit:,.0f}"
        if is_numeric(credit)
        else "N/A"
    )

    annuity_text = (
        f"${annuity:,.0f}"
        if is_numeric(annuity)
        else "N/A"
    )

    metric_1, metric_2, metric_3, metric_4 = st.columns(4)

    metric_1.metric("Age", age)
    metric_2.metric("Income", income_text)
    metric_3.metric("Credit Amount", credit_text)
    metric_4.metric("Annual Payment", annuity_text)

def clamp(value: float, minimum: float = 0.0, maximum: float = 1.0) -> float:
    """Keep a numeric value inside the requested interval."""
    return max(minimum, min(maximum, value))


def interpolate_color(
    color_1: tuple[int, int, int],
    color_2: tuple[int, int, int],
    ratio: float,
) -> str:
    """Interpolate between two RGB colors and return a CSS hex color."""
    ratio = clamp(ratio)

    red = round(color_1[0] + (color_2[0] - color_1[0]) * ratio)
    green = round(color_1[1] + (color_2[1] - color_1[1]) * ratio)
    blue = round(color_1[2] + (color_2[2] - color_1[2]) * ratio)

    return f"#{red:02x}{green:02x}{blue:02x}"


def score_color(score: float, threshold: float) -> str:
    """
    Return a continuous color based on the distance from the threshold.

    threshold - 0.30 or less -> red
    threshold             -> yellow/orange
    threshold + 0.30 or more -> green
    """
    red = (248, 81, 73)
    orange = (240, 136, 62)
    yellow = (210, 170, 45)
    green = (46, 160, 67)

    distance = score - threshold

    if distance <= -0.30:
        return "#f85149"

    if distance < 0:
        # Red -> orange/yellow when approaching the threshold.
        ratio = (distance + 0.30) / 0.30

        if ratio < 0.65:
            return interpolate_color(
                red,
                orange,
                ratio / 0.65,
            )

        return interpolate_color(
            orange,
            yellow,
            (ratio - 0.65) / 0.35,
        )

    if distance < 0.30:
        # Yellow/orange -> green after crossing the threshold.
        return interpolate_color(
            yellow,
            green,
            distance / 0.30,
        )

    return "#2ea043"


def render_score_gauge(
    score: float,
    threshold: float,
    label: str = "Payment Capability Score",
) -> None:
    """
    Render an accessible continuous score gauge.

    The gauge uses HTML/CSS only, so no additional dependency is required.
    """
    score = clamp(float(score))
    threshold = clamp(float(threshold))

    score_percentage = score * 100
    threshold_percentage = threshold * 100
    delta = score - threshold
    color = score_color(score, threshold)

    red_end = clamp(threshold - 0.30) * 100
    orange_point = clamp(threshold - 0.10) * 100
    threshold_point = threshold * 100
    light_green_point = clamp(threshold + 0.10) * 100
    green_start = clamp(threshold + 0.30) * 100

    if delta <= -0.30:
        interpretation = "Well below threshold"
    elif delta < 0:
        interpretation = "Below threshold"
    elif delta < 0.30:
        interpretation = "Above threshold"
    else:
        interpretation = "Well above threshold"

    delta_sign = "+" if delta >= 0 else ""

    st.markdown(
        textwrap.dedent(
            f"""
            <div
                style="
                    background: #161b22;
                    border: 1px solid #30363d;
                    border-radius: 12px;
                    padding: 1.25rem 1.4rem;
                    margin: 0.75rem 0 1.5rem 0;
                "
                role="img"
                aria-label="
                    {label}: {score:.4f}.
                    Threshold: {threshold:.4f}.
                    Difference: {delta_sign}{delta:.4f}.
                    {interpretation}.
                "
            >
                <div style="
                    display: flex;
                    justify-content: space-between;
                    align-items: flex-end;
                    gap: 1rem;
                    margin-bottom: 0.8rem;
                ">
                    <div>
                        <div class="metric-label">
                            {label}
                        </div>

                        <div style="
                            color: {color};
                            font-size: 2rem;
                            line-height: 1.1;
                            font-weight: 700;
                        ">
                            {score:.4f}
                        </div>
                    </div>
                    <div style="text-align: right;">
                        <div style="
                            color: {color};
                            font-size: 0.95rem;
                            font-weight: 600;
                        ">
                            {interpretation}
                        </div>

                        <div style="
                            color: #8b949e;
                            font-size: 0.8rem;
                            margin-top: 0.15rem;
                        ">
                            Threshold {threshold:.4f}
                            &nbsp;·&nbsp;
                            Δ {delta_sign}{delta:.4f}
                        </div>
                    </div>
                </div>

                <div style="
                    position: relative;
                    padding-top: 18px;
                    padding-bottom: 24px;
                ">
                    <!-- Score marker -->
                    <div style="
                        position: absolute;
                        left: {score_percentage:.2f}%;
                        top: 0;
                        transform: translateX(-50%);
                        z-index: 3;
                    ">
                        <div style="
                            width: 0;
                            height: 0;
                            border-left: 7px solid transparent;
                            border-right: 7px solid transparent;
                            border-top: 10px solid {color};
                            margin: auto;
                        "></div>
                    </div>
                    <!-- Gauge track -->
                    <div style="
                        height: 18px;
                        width: 100%;
                        border-radius: 999px;
                        border: 1px solid #30363d;
                        background: linear-gradient(
                            90deg,
                            #f85149 0%,
                            #f85149 {red_end:.2f}%,
                            #f0883e {orange_point:.2f}%,
                            #d2aa2d {threshold_point:.2f}%,
                            #77b255 {light_green_point:.2f}%,
                            #2ea043 {green_start:.2f}%,
                            #2ea043 100%
                        );
                        box-shadow:
                            inset 0 1px 3px rgba(0, 0, 0, 0.45),
                            0 0 12px rgba(0, 0, 0, 0.2);
                        overflow: hidden;
                    ">
                        <!-- Subtle score fill -->
                        <div style="
                            width: {score_percentage:.2f}%;
                            height: 100%;
                            background: rgba(255, 255, 255, 0.13);
                            border-right: 2px solid rgba(255, 255, 255, 0.9);
                        "></div>
                    </div>
                    <!-- Threshold marker -->
                    <div style="
                        position: absolute;
                        left: {threshold_percentage:.2f}%;
                        top: 15px;
                        width: 3px;
                        height: 24px;
                        background: #ffffff;
                        border-radius: 2px;
                        transform: translateX(-50%);
                        box-shadow: 0 0 4px rgba(0, 0, 0, 0.8);
                        z-index: 2;
                    "></div>
                    <!-- Scale labels -->
                    <div style="
                        position: absolute;
                        left: 0;
                        right: 0;
                        bottom: 0;
                        display: flex;
                        justify-content: space-between;
                        color: #8b949e;
                        font-size: 0.72rem;
                    ">
                        <span>0.00</span>
                        <span style="
                            position: absolute;
                            left: {threshold_percentage:.2f}%;
                            transform: translateX(-50%);
                            color: #c9d1d9;
                            font-weight: 600;
                        ">
                            Threshold
                        </span>
                        <span>1.00</span>
                    </div>
                </div>
                <div style="
                    display: flex;
                    align-items: center;
                    gap: 1.2rem;
                    flex-wrap: wrap;
                    margin-top: 0.5rem;
                    color: #8b949e;
                    font-size: 0.75rem;
                ">
                    <span>
                        <span style="
                            display: inline-block;
                            width: 9px;
                            height: 9px;
                            border-radius: 50%;
                            background: #f85149;
                            margin-right: 5px;
                        "></span>
                        Lower score
                    </span>
                    <span>
                        <span style="
                            display: inline-block;
                            width: 9px;
                            height: 9px;
                            border-radius: 50%;
                            background: #d2aa2d;
                            margin-right: 5px;
                        "></span>
                        Near threshold
                    </span>
                    <span>
                        <span style="
                            display: inline-block;
                            width: 9px;
                            height: 9px;
                            border-radius: 50%;
                            background: #2ea043;
                            margin-right: 5px;
                        "></span>
                        Higher score
                    </span>
                    <span style="margin-left: auto;">
                        White line = decision threshold
                    </span>
                </div>
            </div>
            """
        ),
        unsafe_allow_html=True,
    )

def render_prediction(
    prediction: dict,
    title: str = "Prediction Result",
    show_explanations: bool = True,
) -> None:
    """Render one prediction result with a continuous score gauge."""
    status = prediction.get("status", "Unknown")
    probability = prediction.get("probability")
    threshold = prediction.get("threshold")

    approved = status == "Approved"

    card_class = "approved" if approved else "rejected"
    status_class = (
        "status-approved"
        if approved
        else "status-rejected"
    )
    status_icon = "✓" if approved else "✗"

    threshold_text = (
        f"{threshold:.4f}"
        if is_numeric(threshold)
        else "N/A"
    )

    render_section_title(title)

    st.markdown(
        textwrap.dedent(
            f"""
            <div
                class="result-card {card_class}"
                role="region"
                aria-label="{title}: {status}"
            >
                <div style="
                    display: flex;
                    justify-content: space-between;
                    align-items: center;
                    gap: 1.5rem;
                    flex-wrap: wrap;
                ">
                    <div>
                        <div class="metric-label">Decision</div>

                        <div class="{status_class}">
                            {status_icon} {status}
                        </div>
                    </div>
                    <div style="text-align: right;">
                        <div class="metric-label">
                            Decision Threshold
                        </div>

                        <div class="metric-value">
                            {threshold_text}
                        </div>
                    </div>
                </div>
            </div>
            """
        ),
        unsafe_allow_html=True,
    )

    if is_numeric(probability) and is_numeric(threshold):
        render_score_gauge(
            score=float(probability),
            threshold=float(threshold),
        )
    else:
        st.warning(
            "The score gauge cannot be displayed because the score "
            "or threshold is missing."
        )

    if not show_explanations:
        return

    local_image = prediction.get("loc_imp")
    global_image = prediction.get("global_imp")

    if local_image:
        try:
            render_section_title("Local Feature Importance")

            st.image(
                base64.b64decode(local_image),
                caption=(
                    "Factors influencing this customer's prediction."
                ),
            )

        except (ValueError, TypeError):
            st.warning(
                "The local explanation image could not be decoded."
            )

    if global_image:
        try:
            render_section_title("Global Feature Importance")

            st.image(
                base64.b64decode(global_image),
                caption=(
                    "Most influential features across the model."
                ),
            )

        except (ValueError, TypeError):
            st.warning(
                "The global explanation image could not be decoded."
            )


# =============================================================================
# Charts
# =============================================================================

def plot_distribution(
    column_name: str,
    distribution: dict,
) -> plt.Figure:
    """Create a dark-themed distribution chart."""
    bin_edges = np.asarray(
        distribution.get("bin_edges", []),
        dtype=float,
    )

    counts = np.asarray(
        distribution.get("counts", []),
        dtype=float,
    )

    figure, axis = plt.subplots(
        figsize=(5.5, 3.2),
        facecolor="#161b22",
    )

    axis.set_facecolor("#0d1117")

    if len(bin_edges) >= 2 and len(counts) == len(bin_edges) - 1:
        centers = (bin_edges[:-1] + bin_edges[1:]) / 2
        widths = np.diff(bin_edges)

        axis.bar(
            centers,
            counts,
            width=widths * 0.92,
            color="#2d6cdf",
            alpha=0.75,
            edgecolor="none",
            zorder=2,
        )
    else:
        axis.text(
            0.5,
            0.5,
            "No distribution data",
            transform=axis.transAxes,
            color="#8b949e",
            ha="center",
            va="center",
        )

    customer_value = distribution.get("customer_value")

    if is_numeric(customer_value):
        axis.axvline(
            customer_value,
            color="#f85149",
            linewidth=2,
            label=f"Customer: {customer_value:,.4g}",
            zorder=5,
        )

    mean_value = distribution.get("mean")

    if is_numeric(mean_value):
        axis.axvline(
            mean_value,
            color="#c9d1d9",
            linewidth=1.2,
            linestyle="--",
            label=f"Mean: {mean_value:,.4g}",
            zorder=4,
        )

    percentile = distribution.get("percentile")

    if is_numeric(percentile):
        arrow = "▲" if percentile >= 50 else "▼"

        axis.text(
            0.97,
            0.95,
            f"{arrow} P{percentile:.0f}",
            transform=axis.transAxes,
            color="#c9d1d9",
            fontsize=9,
            fontweight="bold",
            ha="right",
            va="top",
        )

    sample_size = distribution.get("n", 0)

    if is_numeric(sample_size):
        axis.text(
            0.97,
            0.05,
            f"n={int(sample_size):,}",
            transform=axis.transAxes,
            color="#8b949e",
            fontsize=9,
            ha="right",
            va="bottom",
        )

    handles, labels = axis.get_legend_handles_labels()

    if handles:
        axis.legend(
            fontsize=9,
            loc="upper left",
            facecolor="#161b22",
            edgecolor="#30363d",
            labelcolor="#c9d1d9",
        )

    axis.set_title(
        column_name,
        color="#c9d1d9",
        fontsize=10,
        loc="left",
        pad=6,
    )

    axis.set_ylabel(
        "Count",
        color="#8b949e",
        fontsize=9,
    )

    axis.tick_params(
        axis="both",
        colors="#8b949e",
        labelsize=9,
    )

    for spine in axis.spines.values():
        spine.set_edgecolor("#30363d")

    figure.tight_layout(pad=0.8)

    return figure


def plot_bivariate(
    column_x: str,
    column_y: str,
    data: dict,
) -> plt.Figure:
    """Create a bivariate population scatter plot."""
    figure, axis = plt.subplots(
        figsize=(6, 4),
        facecolor="#161b22",
    )

    axis.set_facecolor("#0d1117")

    population_x = data.get("pop_x", [])
    population_y = data.get("pop_y", [])

    if len(population_x) and len(population_y):
        axis.scatter(
            population_x,
            population_y,
            color="#2d6cdf",
            alpha=0.35,
            s=14,
            edgecolor="none",
            label="Population",
            zorder=2,
        )

    customer_x = data.get("customer_x")
    customer_y = data.get("customer_y")

    if is_numeric(customer_x) and is_numeric(customer_y):
        axis.scatter(
            customer_x,
            customer_y,
            color="#f85149",
            s=110,
            marker="X",
            edgecolor="#ffffff",
            linewidth=0.9,
            label="Customer",
            zorder=5,
        )

    sample_size = data.get("n", 0)

    if is_numeric(sample_size):
        axis.text(
            0.97,
            0.05,
            f"n={int(sample_size):,}",
            transform=axis.transAxes,
            color="#8b949e",
            fontsize=9,
            ha="right",
            va="bottom",
        )

    axis.set_title(
        "Bivariate Space Analysis",
        color="#c9d1d9",
        fontsize=10,
        loc="left",
        pad=6,
    )

    axis.set_xlabel(
        column_x,
        color="#8b949e",
        fontsize=9,
    )

    axis.set_ylabel(
        column_y,
        color="#8b949e",
        fontsize=9,
    )

    axis.tick_params(
        axis="both",
        colors="#8b949e",
        labelsize=9,
    )

    for spine in axis.spines.values():
        spine.set_edgecolor("#30363d")

    handles, labels = axis.get_legend_handles_labels()

    if handles:
        axis.legend(
            fontsize=9,
            loc="upper right",
            facecolor="#161b22",
            edgecolor="#30363d",
            labelcolor="#c9d1d9",
        )

    figure.tight_layout(pad=0.8)

    return figure


# =============================================================================
# Distribution API
# =============================================================================

def fetch_distributions(
    sk_id: str,
    table: str,
    columns: list[str],
) -> dict:
    """Fetch and cache population distributions."""
    cache = st.session_state.setdefault(
        "distribution_cache",
        {},
    )

    distributions = {}

    for column in columns:
        cache_key = (
            str(sk_id),
            table,
            column,
        )

        if cache_key in cache:
            distributions[column] = cache[cache_key]
            continue

        try:
            response = requests.get(
                f"{API_URL}/distributions/{table}",
                params={
                    "column": column,
                    "sk_id": sk_id,
                },
                timeout=15,
            )

            if response.status_code != 200:
                st.warning(
                    f"Distribution unavailable for {column} "
                    f"(status {response.status_code})."
                )
                continue

            distribution = response.json()

            cache[cache_key] = distribution
            distributions[column] = distribution

        except requests.exceptions.RequestException as exc:
            st.warning(
                f"Distribution request failed for {column}: {exc}"
            )

    return distributions


# =============================================================================
# Session state
# =============================================================================

for key, default_value in {
    "current_sk_id": None,
    "prediction": None,
    "customer_data": None,
    "simulated_prediction": None,
    "simulation_details": None,
    "distribution_cache": {},
    "displayed_distributions": [],
    "displayed_distribution_table": None,
    "bivariate_result": None,
}.items():
    st.session_state.setdefault(key, default_value)


def clear_customer_state() -> None:
    """Clear state associated with the current customer."""
    st.session_state["prediction"] = None
    st.session_state["customer_data"] = None
    st.session_state["simulated_prediction"] = None
    st.session_state["simulation_details"] = None
    st.session_state["distribution_cache"] = {}
    st.session_state["displayed_distributions"] = []
    st.session_state["displayed_distribution_table"] = None
    st.session_state["bivariate_result"] = None


# =============================================================================
# Header
# =============================================================================

st.markdown(
    '<div class="title">Credit Scoring Dashboard</div>',
    unsafe_allow_html=True,
)

st.markdown(
    """
    <div class="subtitle">
        Home Credit Default Risk — Internal Risk Assessment Tool
    </div>
    """,
    unsafe_allow_html=True,
)


# =============================================================================
# Main controls
# =============================================================================

customer_id = st.text_input(
    "Customer ID (SK_ID_CURR)",
    placeholder="100001",
).strip()

button_column, health_column = st.columns([3, 1])

predict_button = button_column.button(
    "Predict & Explain",
    type="primary",
    use_container_width=True,
)

health_button = health_column.button(
    "Check API Health",
    use_container_width=True,
)

st.divider()


# =============================================================================
# Health check
# =============================================================================

if health_button:
    with st.spinner("Checking API status…"):
        try:
            response = requests.get(
                f"{API_URL}/check_api",
                timeout=5,
            )

            if response.status_code == 200:
                st.success("API is running.")
            else:
                st.error(
                    f"API health check failed "
                    f"(status {response.status_code})."
                )

        except requests.exceptions.RequestException as exc:
            st.error(f"Could not reach the API: {exc}")


# =============================================================================
# Prediction + raw-data loading
# =============================================================================

if predict_button:
    if not customer_id:
        st.warning("Please enter a Customer ID.")
    else:
        clear_customer_state()

        with st.spinner(
            "Running prediction and loading customer data…"
        ):
            try:
                prediction = get_prediction(customer_id)
                customer_data = fetch_customer_data(customer_id)

                st.session_state["current_sk_id"] = customer_id
                st.session_state["prediction"] = prediction
                st.session_state["customer_data"] = customer_data

            except ValueError as exc:
                st.error(str(exc))

            except RuntimeError as exc:
                st.error(str(exc))

            except requests.exceptions.Timeout:
                st.error("The API request timed out.")

            except requests.exceptions.RequestException as exc:
                st.error(f"Could not connect to the API: {exc}")


# =============================================================================
# Display customer workspace
# =============================================================================

customer_loaded = (
    customer_id
    and customer_id == st.session_state["current_sk_id"]
    and st.session_state["prediction"] is not None
    and st.session_state["customer_data"] is not None
)

if customer_loaded:
    prediction = st.session_state["prediction"]
    customer_data = st.session_state["customer_data"]

    # -------------------------------------------------------------------------
    # Original prediction
    # -------------------------------------------------------------------------

    render_prediction(
        prediction,
        title="Prediction Result",
        show_explanations=True,
    )

    st.divider()

    # -------------------------------------------------------------------------
    # What-if simulation
    # -------------------------------------------------------------------------

    render_section_title("What-if Simulation")

    st.caption(
        "Select one raw-data feature, change its value and rerun the model."
    )

    flattened_features = flatten_customer_data(customer_data)

    if flattened_features:
        feature_labels = sorted(flattened_features)

        selected_feature_label = st.selectbox(
            "Raw-data feature",
            options=feature_labels,
            key="simulation_feature",
            help="Format: source table · column",
        )

        selected_feature = flattened_features[
            selected_feature_label
        ]

        selected_table = selected_feature["table"]
        selected_column = selected_feature["column"]
        original_value = selected_feature["value"]

        st.caption(
            f"Current value: `{format_value(original_value)}`"
        )

        if is_numeric(original_value):
            integer_value = isinstance(
                original_value,
                (int, np.integer),
            )

            with st.form("numeric_simulation_form"):
                if integer_value:
                    simulated_value = st.number_input(
                        "New numerical value",
                        value=int(original_value),
                        step=1,
                    )
                else:
                    simulated_value = st.number_input(
                        "New numerical value",
                        value=float(original_value),
                        format="%.8f",
                    )

                simulation_button = st.form_submit_button(
                    "Run Simulated Prediction",
                    type="primary",
                    use_container_width=True,
                )

        else:
            possible_values = fetch_categorical_values(
                selected_table,
                selected_column,
            )

            if original_value not in possible_values:
                possible_values.insert(0, original_value)

            possible_values = list(
                dict.fromkeys(possible_values)
            )

            with st.form("categorical_simulation_form"):
                simulated_value = st.selectbox(
                    "New categorical value",
                    options=possible_values,
                    index=possible_values.index(original_value),
                    format_func=format_value,
                )

                simulation_button = st.form_submit_button(
                    "Run Simulated Prediction",
                    type="primary",
                    use_container_width=True,
                    disabled=len(possible_values) <= 1,
                )

            if len(possible_values) <= 1:
                st.info(
                    "No alternative category was returned by the API "
                    "for this feature."
                )

        if simulation_button:
            with st.spinner("Running simulated prediction…"):
                try:
                    simulated_prediction = (
                        get_simulated_prediction(
                            customer_id,
                            selected_table,
                            selected_column,
                            simulated_value,
                        )
                    )

                    st.session_state[
                        "simulated_prediction"
                    ] = simulated_prediction

                    st.session_state["simulation_details"] = {
                        "table": selected_table,
                        "column": selected_column,
                        "original_value": original_value,
                        "simulated_value": simulated_value,
                    }

                except ValueError as exc:
                    st.error(str(exc))

                except RuntimeError as exc:
                    st.error(str(exc))

                except requests.exceptions.Timeout:
                    st.error(
                        "The simulated prediction timed out."
                    )

                except requests.exceptions.RequestException as exc:
                    st.error(
                        f"Could not connect to the API: {exc}"
                    )

        # ---------------------------------------------------------------------
        # Simulated result
        # ---------------------------------------------------------------------

        simulated_prediction = st.session_state[
            "simulated_prediction"
        ]

        simulation_details = st.session_state[
            "simulation_details"
        ]

        if simulated_prediction and simulation_details:
            st.markdown("---")

            render_section_title("Simulation Comparison")

            original_probability = prediction.get("probability")
            simulated_probability = simulated_prediction.get(
                "probability"
            )

            comparison_1, comparison_2, comparison_3 = st.columns(3)

            comparison_1.metric(
                "Original Score",
                (
                    f"{original_probability:.4f}"
                    if is_numeric(original_probability)
                    else "N/A"
                ),
            )

            comparison_2.metric(
                "Simulated Score",
                (
                    f"{simulated_probability:.4f}"
                    if is_numeric(simulated_probability)
                    else "N/A"
                ),
            )

            if (
                is_numeric(original_probability)
                and is_numeric(simulated_probability)
            ):
                comparison_3.metric(
                    "Score Variation",
                    f"{simulated_probability - original_probability:+.4f}",
                )
            else:
                comparison_3.metric(
                    "Score Variation",
                    "N/A",
                )

            st.caption(
                f"Modified feature: "
                f"**{simulation_details['table']} · "
                f"{simulation_details['column']}** — "
                f"`{format_value(simulation_details['original_value'])}` "
                f"→ "
                f"`{format_value(simulation_details['simulated_value'])}`"
            )

            render_prediction(
                simulated_prediction,
                title="Simulated Prediction Result",
                show_explanations=True,
            )

            if st.button(
                "Clear Simulation",
                use_container_width=True,
            ):
                st.session_state["simulated_prediction"] = None
                st.session_state["simulation_details"] = None
                st.rerun()

    else:
        st.info("No editable raw-data features are available.")

    st.divider()

    # -------------------------------------------------------------------------
    # Key metrics
    # -------------------------------------------------------------------------

    render_section_title("Key Metrics")
    render_key_metrics(customer_data)

    st.divider()

    # -------------------------------------------------------------------------
    # Distribution explorer
    # -------------------------------------------------------------------------

    render_section_title("Distribution Explorer")

    available_tables = [
        table
        for table, table_data in customer_data.items()
        if isinstance(table_data, dict)
    ]

    distribution_table = st.selectbox(
        "Source table",
        options=available_tables,
        key="distribution_table",
    )

    available_numeric_columns = numeric_columns(
        customer_data[distribution_table]
    )

    if not available_numeric_columns:
        st.info(
            "No numeric columns are available in this table."
        )
    else:
        selected_distribution_columns = st.multiselect(
            "Columns to visualise",
            options=available_numeric_columns,
            key="distribution_columns",
        )

        if st.button(
            "Show Distributions",
            use_container_width=True,
        ):
            if selected_distribution_columns:
                st.session_state[
                    "displayed_distributions"
                ] = selected_distribution_columns

                st.session_state[
                    "displayed_distribution_table"
                ] = distribution_table
            else:
                st.info("Select at least one numeric column.")

        displayed_table = st.session_state[
            "displayed_distribution_table"
        ]

        displayed_columns = st.session_state[
            "displayed_distributions"
        ]

        if displayed_table == distribution_table and displayed_columns:
            with st.spinner("Loading distributions…"):
                distributions = fetch_distributions(
                    customer_id,
                    distribution_table,
                    displayed_columns,
                )

            for start in range(0, len(displayed_columns), 2):
                chart_columns = st.columns(2)

                current_columns = displayed_columns[
                    start:start + 2
                ]

                for index, column in enumerate(current_columns):
                    distribution = distributions.get(column)

                    if not distribution:
                        continue

                    with chart_columns[index]:
                        figure = plot_distribution(
                            column,
                                 )

                        st.pyplot(
                            figure,
                            clear_figure=True,
                        )

                        plt.close(figure)

                        caption = []

                        customer_value = distribution.get(
                            "customer_value"
                        )
                        percentile = distribution.get("percentile")
                        mean_value = distribution.get("mean")
                        standard_deviation = distribution.get("std")

                        if is_numeric(customer_value):
                            caption.append(
                                f"Customer: "
                                f"**{customer_value:,.4g}**"
                            )

                        if is_numeric(percentile):
                            caption.append(
                                f"Percentile: "
                                f"**P{percentile:.0f}**"
                            )

                        if is_numeric(mean_value):
                            caption.append(
                                f"Mean: {mean_value:,.4g}"
                            )

                        if is_numeric(standard_deviation):
                            caption.append(
                                f"Std: {standard_deviation:,.4g}"
                            )

                        if caption:
                            st.caption(" · ".join(caption))

    st.divider()

    # -------------------------------------------------------------------------
    # Raw column explorer
    # -------------------------------------------------------------------------

    render_section_title("Column Explorer")

    all_raw_features = flatten_customer_data(customer_data)

    selected_raw_features = st.multiselect(
        "Columns to inspect",
        options=sorted(all_raw_features),
        help="Format: source table · column",
    )

    if selected_raw_features:
        rows = []

        for feature_label in selected_raw_features:
            feature = all_raw_features[feature_label]

            rows.append(
                {
                    "Table": feature["table"],
                    "Feature": feature["column"],
                    "Value": feature["value"],
                }
            )

        st.dataframe(
            pd.DataFrame(rows),
            use_container_width=True,
            hide_index=True,
        )

    st.divider()

    # -------------------------------------------------------------------------
    # Bivariate explorer
    # -------------------------------------------------------------------------

    render_section_title("Bivariate Coordinate Analysis")

    bivariate_table = st.selectbox(
        "Bivariate source table",
        options=available_tables,
        key="bivariate_table",
    )

    bivariate_columns = numeric_columns(
        customer_data[bivariate_table]
    )

    if len(bivariate_columns) < 2:
        st.info(
            "At least two numeric columns are required."
        )
    else:
        axis_column_1, axis_column_2 = st.columns(2)

        column_x = axis_column_1.selectbox(
            "Horizontal Axis (X)",
            options=bivariate_columns,
            index=0,
            key="bivariate_x",
        )

        column_y = axis_column_2.selectbox(
            "Vertical Axis (Y)",
            options=bivariate_columns,
            index=1,
            key="bivariate_y",
        )

        if st.button(
            "Generate Bivariate Map",
            use_container_width=True,
        ):
            if column_x == column_y:
                st.warning(
                    "Select two different numeric columns."
                )
            else:
                with st.spinner(
                    "Loading bivariate population data…"
                ):
                    try:
                        response = requests.get(
                            f"{API_URL}/bivariate/{bivariate_table}",
                            params={
                                "col_x": column_x,
                                "col_y": column_y,
                                "sk_id": customer_id,
                            },
                            timeout=20,
                        )

                        bivariate_data = check_response(
                            response,
                            "Bivariate analysis",
                        )

                        st.session_state["bivariate_result"] = {
                            "table": bivariate_table,
                            "column_x": column_x,
                            "column_y": column_y,
                            "data": bivariate_data,
                        }

                    except ValueError as exc:
                        st.error(str(exc))

                    except RuntimeError as exc:
                        st.error(str(exc))

                    except requests.exceptions.RequestException as exc:
                        st.error(
                            f"Bivariate request failed: {exc}"
                        )

        bivariate_result = st.session_state[
            "bivariate_result"
        ]

        if (
            bivariate_result
            and bivariate_result["table"] == bivariate_table
        ):
            figure = plot_bivariate(
                bivariate_result["column_x"],
                bivariate_result["column_y"],
                bivariate_result["data"],
            )

            st.pyplot(
                figure,
                clear_figure=True,
            )

            plt.close(figure)

            bivariate_data = bivariate_result["data"]

            customer_x = bivariate_data.get("customer_x")
            customer_y = bivariate_data.get("customer_y")

            if is_numeric(customer_x) and is_numeric(customer_y):
                st.caption(
                    f"Customer coordinates — "
                    f"**{bivariate_result['column_x']}**: "
                    f"{customer_x:,.4g} · "
                    f"**{bivariate_result['column_y']}**: "
                    f"{customer_y:,.4g}"
                )
