import os
import streamlit as st
import requests
import pandas as pd
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import base64
 
matplotlib.use("Agg")  # headless backend — must come before any plt usage
 
API_URL = os.environ["API_URL"]
 
st.set_page_config(page_title="Credit Scoring", page_icon="🏦", layout="wide")
 
# ---------------------------------------------------------------------------
# CSS
# ---------------------------------------------------------------------------
st.markdown("""
    <style>
        html, body, [class*="css"] { 
            font-family: 'DM Sans', sans-serif; 
            background-color: #0d1117;
            color: #c9d1d9;
        }
        .title { 
            font-family: 'DM Mono', monospace; 
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
            font-weight: 300; 
        }
        .result-card { 
            background: #161b22;
            border-radius: 12px; 
            padding: 1.5rem; 
            border: 1px solid #30363d;
            margin-bottom: 1rem; 
            box-shadow: 0 4px 15px rgba(0, 0, 0, 0.5);
        }
        .approved { border-left: 4px solid #2ea043; }
        .rejected { border-left: 4px solid #f85149; }
        .metric-label { 
            font-family: 'DM Mono', monospace; 
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
        .status-approved { color: #2ea043; font-size: 1.4rem; font-weight: 600; }
        .status-rejected { color: #f85149; font-size: 1.4rem; font-weight: 600; }
        .section-title { 
            font-family: 'DM Mono', monospace; 
            font-size: 0.8rem; 
            font-weight: 400;       /* neutralise browser h3 bold default */
            color: #8b949e;
            text-transform: uppercase; 
            letter-spacing: 0.1em; 
            margin-top: 0;          /* neutralise browser h3 margin default */
            margin-bottom: 1rem; 
            padding-bottom: 0.5rem; 
            border-bottom: 1px solid #30363d;
        }
        div[data-testid="stSidebar"] { background-color: #0d1117 !important; }
    </style>
""", unsafe_allow_html=True)
 
 
# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
 
def find_val(nested_data: dict, col: str):
    """Search for a column name across all tables in the nested explore dict."""
    for table_cols in nested_data.values():
        if col in table_cols:
            return table_cols[col]
    return None
 
 
def render_key_metrics(nested_data: dict):
    """Display the four headline metrics from any table that holds them."""
    m1, m2, m3, m4 = st.columns(4)
 
    days_birth = find_val(nested_data, 'DAYS_BIRTH')
    age = int(-days_birth / 365) if days_birth is not None else "N/A"
 
    income_raw = find_val(nested_data, 'AMT_INCOME_TOTAL')
    income = f"${income_raw:,.0f}" if income_raw is not None else "N/A"
 
    credit_raw = find_val(nested_data, 'AMT_CREDIT')
    credit = f"${credit_raw:,.0f}" if credit_raw is not None else "N/A"
 
    annuity_raw = find_val(nested_data, 'AMT_ANNUITY')
    annuity = f"${annuity_raw:,.0f}" if annuity_raw is not None else "N/A"
 
    m1.metric("Age", str(age))
    m2.metric("Income", income)
    m3.metric("Credit Amount", credit)
    m4.metric("Annual Payment", annuity)
 
 
def plot_distribution(col_name: str, dist: dict) -> plt.Figure:
    """
    Build a dark-themed histogram for one column.
    Marks the customer's value in red and the population mean in dashed white.
    """
    bin_edges = np.array(dist["bin_edges"])
    counts = np.array(dist["counts"])
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
    bin_width = bin_edges[1] - bin_edges[0]
 
    fig, ax = plt.subplots(figsize=(5.5, 3.2), facecolor="#161b22")
    ax.set_facecolor("#0d1117")
 
    ax.bar(
        bin_centers, counts,
        width=bin_width * 0.92,
        color="#2d6cdf",
        alpha=0.75,
        edgecolor="none",
        zorder=2,
    )
 
    customer_val = dist.get("customer_value")
    if customer_val is not None:
        ax.axvline(
            customer_val, color="#f85149", linewidth=2.0,
            zorder=5, label=f"Customer: {customer_val:,.2f}"
        )
 
    mean_val = dist.get("mean")
    if mean_val is not None:
        ax.axvline(
            mean_val, color="#8b949e", linewidth=1.2,
            linestyle="--", zorder=4, label=f"Mean: {mean_val:,.2f}"
        )
 
    # Percentile badge — neutral colour so meaning is never colour-only.
    # A ▲/▼ arrow conveys above/below-median for users who cannot rely on colour.
    pct = dist.get("percentile")
    if pct is not None:
        arrow = "▲" if pct >= 50 else "▼"
        ax.text(
            0.97, 0.95, f"{arrow} P{pct:.0f}",
            transform=ax.transAxes,
            color="#c9d1d9", fontsize=9, fontweight="bold",
            ha="right", va="top",
        )
 
    # Legend (only when there are labelled artists)
    handles, labels = ax.get_legend_handles_labels()
    if handles:
        ax.legend(
            fontsize=9, loc="upper left",
            facecolor="#161b22", edgecolor="#30363d",
            labelcolor="#c9d1d9",
        )
 
    # Axis styling — minimum 9 px so labels are legible
    ax.set_title(col_name, color="#c9d1d9", fontsize=10, pad=6, loc="left")
    ax.tick_params(axis="both", colors="#8b949e", labelsize=9)
    for spine in ax.spines.values():
        spine.set_edgecolor("#30363d")
    ax.set_xlabel("")
    ax.set_ylabel("Count", color="#8b949e", fontsize=9)
 
    # Sample size annotation bottom-right
    n = dist.get("n", 0)
    ax.text(
        0.97, 0.05, f"n={n:,}",
        transform=ax.transAxes,
        color="#8b949e", fontsize=9, ha="right", va="bottom",
    )
 
    plt.tight_layout(pad=0.8)
    return fig


def plot_bivariate_scatter(col_x: str, col_y: str, data: dict) -> plt.Figure:
    """
    Generates a high-performance dark-themed scatter interface.
    Highlights specific tracking client with an isolated, high-contrast marker.
    """
    fig, ax = plt.subplots(figsize=(5.5, 3.8), facecolor="#161b22")
    ax.set_facecolor("#0d1117")
 
    # Background Sample Population Scatter Map
    ax.scatter(
        data["pop_x"], data["pop_y"],
        color="#2d6cdf", alpha=0.35, s=14, edgecolor="none", zorder=2,
        label="Population Background"
    )
 
    # Target Highlighted Customer Coordinates 
    cx = data.get("customer_x")
    cy = data.get("customer_y")
    if cx is not None and cy is not None:
        ax.scatter(
            cx, cy,
            color="#f85149", s=110, marker="X", edgecolor="#ffffff", linewidths=0.9, zorder=5,
            label=f"Target Customer"
        )
 
    # Label styling and contextual framing matching UI palette
    ax.set_title(f"Bivariate Space Analysis", color="#c9d1d9", fontsize=10, pad=6, loc="left")
    ax.tick_params(axis="both", colors="#8b949e", labelsize=9)
    for spine in ax.spines.values():
        spine.set_edgecolor("#30363d")
        
    ax.set_xlabel(col_x, color="#8b949e", fontsize=9)
    ax.set_ylabel(col_y, color="#8b949e", fontsize=9)
 
    ax.legend(
        fontsize=9, loc="upper right",
        facecolor="#161b22", edgecolor="#30363d", labelcolor="#c9d1d9"
    )
 
    n = data.get("n", 0)
    ax.text(
        0.97, 0.05, f"n={n:,}",
        transform=ax.transAxes,
        color="#8b949e", fontsize=9, ha="right", va="bottom"
    )
 
    plt.tight_layout(pad=0.8)
    return fig
 
 
def fetch_distributions(sk_id: str, table: str, columns: list[str], filter_col: str = None, filter_val: str = None) -> dict:
    """Fetch distribution data, factoring in optional filters."""
    # We include filter parameters in the cache key so switching filters refetches data
    cache = st.session_state.setdefault("dist_cache", {})
    results = {}
    missing = []
 
    for col in columns:
        key = (table, col, filter_col, filter_val)
        if key in cache:
            results[col] = cache[key]
        else:
            missing.append((col, key))
 
    if missing:
        with st.spinner(f"Fetching distributions for {len(missing)} column(s)…"):
            for col, key in missing:
                try:
                    params = {"column": col, "sk_id": sk_id}
                    if filter_col and filter_val is not None:
                        params["filter_col"] = filter_col
                        params["filter_val"] = filter_val
                        
                    resp = requests.get(f"{API_URL}/distributions/{table}", params=params, timeout=15)
                    if resp.status_code == 200:
                        data = resp.json()
                        cache[key] = data
                        results[col] = data
                    else:
                        st.warning(f"Could not fetch distribution for **{col}** (status {resp.status_code}).")
                except requests.exceptions.RequestException as e:
                    st.warning(f"Request failed for **{col}**: {e}")
 
    return results
 
 
# ---------------------------------------------------------------------------
# Header
# ---------------------------------------------------------------------------
st.markdown('<div class="title">Credit Scoring Dashboard</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="subtitle">Home Credit Default Risk — Internal Risk Assessment Tool</div>',
    unsafe_allow_html=True,
)
 
# ---------------------------------------------------------------------------
# Input row
# ---------------------------------------------------------------------------
sk_id = st.text_input("Customer ID (SK_ID_CURR)", placeholder="100001")
col1, col2, col3 = st.columns([1, 1, 0.5])
predict_btn     = col1.button("Predict & Explain", type="primary")
explore_btn     = col2.button("Load Customer Data")
check_api_btn   = col3.button("Check API Health")
 
st.divider()
 
# ---------------------------------------------------------------------------
# API health check
# ---------------------------------------------------------------------------
if check_api_btn:
    with st.spinner("Checking API status…"):
        try:
            resp = requests.get(f"{API_URL}/check_api", timeout=5)
            if resp.status_code == 200 and "API is running" in resp.text:
                st.success("API Health Check Successful: API is running.")
            else:
                st.error(f"API Health Check Failed (Status {resp.status_code}).")
        except requests.exceptions.ConnectionError:
            st.error("Connection Error: Could not reach the API endpoint.")
 
# ---------------------------------------------------------------------------
# Prediction
# ---------------------------------------------------------------------------
if predict_btn:
    if not sk_id:
        st.warning("Please enter a Customer ID to run prediction.")
    else:
        with st.spinner("Running prediction…"):
            try:
                resp = requests.get(f"{API_URL}/predict/{sk_id}", timeout=30)
                if resp.status_code == 404:
                    st.error("Customer ID not found in the system.")
                elif resp.status_code != 200:
                    st.error(f"API error: Status Code {resp.status_code}")
                else:
                    data = resp.json()
                    loc_imp    = base64.b64decode(data["loc_imp"])
                    global_imp = base64.b64decode(data["global_imp"])
                    status_icon  = "✓" if data["status"] == "Approved" else "✗"
                    status_class = "approved" if data["status"] == "Approved" else "rejected"
                    status_color = "status-approved" if data["status"] == "Approved" else "status-rejected"
 
                    st.markdown(f"""
                        <div class="result-card {status_class}"
                             role="region"
                             aria-label="Prediction result: {data['status']}">
                            <div style="display:flex; justify-content:space-between; align-items:center;">
                                <div>
                                    <div class="metric-label">Decision</div>
                                    <div class="{status_color}">{status_icon} {data["status"]}</div>
                                </div>
                                <div>
                                    <div class="metric-label">Payment Capability Score</div>
                                    <div class="metric-value">{data["probability"]:.4f}</div>
                                </div>
                                <div>
                                    <div class="metric-label">Threshold</div>
                                    <div class="metric-value">{data["threshold"]:.4f}</div>
                                </div>
                            </div>
                        </div>
                    """, unsafe_allow_html=True)
 
                    st.markdown('<h3 class="section-title">Local Feature Importance</h3>', unsafe_allow_html=True)
                    st.image(
                        loc_imp,
                        caption=(
                            "Local SHAP feature importance: the top factors that pushed this "
                            "customer's credit score above or below the decision threshold."
                        ),
                    )
                    st.markdown('<h3 class="section-title">Global Feature Importance</h3>', unsafe_allow_html=True)
                    st.image(
                        global_imp,
                        caption=(
                            "Global feature importance: the most influential features across "
                            "all customers in the trained model."
                        ),
                    )
 
            except requests.exceptions.ConnectionError:
                st.error("Could not connect to API.")
 
# ---------------------------------------------------------------------------
# Load customer data (explore endpoint)
# ---------------------------------------------------------------------------
if explore_btn:
    if not sk_id:
        st.warning("Please enter a Customer ID to load data.")
    else:
        with st.spinner("Loading customer data…"):
            try:
                resp = requests.get(f"{API_URL}/explore/{sk_id}", timeout=30)
                if resp.status_code == 404:
                    st.error("Customer ID not found.")
                elif resp.status_code != 200:
                    st.error(f"API error: Status Code {resp.status_code}")
                else:
                    # Store nested { table: { col: val } } in session
                    st.session_state["customer_data"]  = resp.json()
                    st.session_state["current_sk_id"]  = sk_id
                    st.session_state["dist_cache"]     = {}   # clear cache for new customer
            except requests.exceptions.ConnectionError:
                st.error("Could not connect to API.")
 
# ---------------------------------------------------------------------------
# Explorer UI — shown whenever valid customer data is in session state
# (persists across button clicks so the user can freely switch between
#  Predict and Explorer without re-loading each time)
# ---------------------------------------------------------------------------
if (
    "customer_data" in st.session_state
    and sk_id == st.session_state.get("current_sk_id")
):
    nested_data: dict = st.session_state["customer_data"]
 
    st.markdown("---")
 
    # ── Key metrics ──────────────────────────────────────────────────────────
    st.markdown('<h3 class="section-title">Key Metrics</h3>', unsafe_allow_html=True)
    render_key_metrics(nested_data)
    st.divider()
 
    # ── Distribution explorer ─────────────────────────────────────────────────
    st.markdown('<h3 class="section-title">Distribution Explorer</h3>', unsafe_allow_html=True)
 
    # 1. Table selector
    all_tables = list(nested_data.keys())
    selected_table = st.selectbox(
        "Source table",
        options=all_tables,
        help="Each table corresponds to a different data source for this customer.",
    )
 
    # 2. Column selector — only numeric, non-null columns
    table_cols = sorted([
        col for col, val in nested_data[selected_table].items()
        if val is not None and col not in ("SK_ID_CURR", "SK_ID_BUREAU", "SK_ID_PREV")
        and isinstance(val, (int, float))
    ])
 
    if not table_cols:
        st.info("No numeric columns available in this table.")
    else:
        selected_cols = st.multiselect(
            "Columns to visualise",
            options=table_cols,
            help="Select one or more numeric columns to see their population distribution.",
        )
 
        show_btn = st.button("Show Distributions")
 
        if show_btn:
            if not selected_cols:
                st.info("Select at least one column above, then click Show Distributions.")
            else:
                st.session_state["selected_cols_display"] = selected_cols
                st.session_state["selected_table_display"] = selected_table
 
        # Render plots if we have a committed selection
        disp_table = st.session_state.get("selected_table_display")
        disp_cols  = st.session_state.get("selected_cols_display", [])
 
        # Reset display if the table changed
        if disp_table != selected_table:
            disp_cols = []
 
        if disp_cols:
            dist_data = fetch_distributions(
                st.session_state["current_sk_id"], disp_table, disp_cols
            )
 
            if dist_data:
                # Grid: 2 plots per row
                cols_per_row = 2
                for row_start in range(0, len(disp_cols), cols_per_row):
                    grid = st.columns(cols_per_row)
                    for j, col in enumerate(disp_cols[row_start: row_start + cols_per_row]):
                        with grid[j]:
                            if col in dist_data:
                                fig = plot_distribution(col, dist_data[col])
                                st.pyplot(fig, clear_figure=True)
                                plt.close(fig)
 
                                d = dist_data[col]
                                pct   = d.get("percentile")
                                cval  = d.get("customer_value")
                                mean  = d.get("mean")
                                std   = d.get("std")
                                caption_parts = []
                                if cval  is not None: caption_parts.append(f"Customer: **{cval:,.4g}**")
                                if pct   is not None: caption_parts.append(f"Percentile: **P{pct:.0f}**")
                                if mean  is not None: caption_parts.append(f"Mean: {mean:,.4g}")
                                if std   is not None: caption_parts.append(f"Std: {std:,.4g}")
                                if caption_parts:
                                    st.caption("  ·  ".join(caption_parts))
 
    st.divider()
 
    # ── Column value table ────────────────────────────────────────────────────
    st.markdown('<h3 class="section-title">Column Explorer</h3>', unsafe_allow_html=True)
 
    all_flat_cols = sorted([
        f"{table} · {col}"
        for table, cols in nested_data.items()
        for col, val in cols.items()
        if val is not None and col not in ("SK_ID_CURR",)
    ])
 
    selected_flat = st.multiselect(
        "Select columns to inspect (any table)",
        options=all_flat_cols,
        help="Format: table · column_name",
    )
 
    if selected_flat:
        rows = []
        for entry in selected_flat:
            table_name, col_name = entry.split(" · ", 1)
            val = nested_data.get(table_name, {}).get(col_name)
            rows.append({"Table": table_name, "Feature": col_name, "Value": val})
 
        st.dataframe(
            pd.DataFrame(rows),
            use_container_width=True,
            hide_index=True,
        )



    st.divider()

    # ── Bivariate Explorer Interface ──────────────────────────────────────────
    st.markdown('<h3 class="section-title">Bivariate Coordinate Analysis</h3>', unsafe_allow_html=True)
 
    # Dynamic Column Filtering mapping from the selected workspace table
    biv_cols = sorted([
        col for col, val in nested_data[selected_table].items()
        if val is not None and col not in ("SK_ID_CURR", "SK_ID_BUREAU", "SK_ID_PREV")
        and isinstance(val, (int, float))
    ])
 
    if not biv_cols:
        st.info("Insufficient quantitative data vectors available in this workspace table.")
    else:
        bx1, bx2 = st.columns(2)
        with bx1:
            col_x = st.selectbox("Horizontal Axis (X)", options=biv_cols, index=0, key="biv_axis_x")
        with bx2:
            y_init = 1 if len(biv_cols) > 1 else 0
            col_y = st.selectbox("Vertical Axis (Y)", options=biv_cols, index=y_init, key="biv_axis_y")
 
        # Add filtering using existing columns and values
        filter_options = ["No Active Segment Filter"] + sorted(list(nested_data[selected_table].keys()))
        selected_filter_col = st.selectbox("Filter Reference Frame (Optional)", options=filter_options, key="biv_filter_col")
        
        filter_params = {}
        if selected_filter_col != "No Active Segment Filter":
            target_filter_val = nested_data[selected_table].get(selected_filter_col)
            filter_params["filter_col"] = selected_filter_col
            filter_params["filter_val"] = target_filter_val
            st.caption(f"Comparing user against population where **{selected_filter_col}** matches customer value: `{target_filter_val}`")
 
        if st.button("Generate Bivariate Space Map"):
            if col_x == col_y:
                st.error("Please pick unique structural column targets for distinct dimensions.")
            else:
                with st.spinner("Processing coordinate maps from server database..."):
                    try:
                        req_payload = {
                            "col_x": col_x,
                            "col_y": col_y,
                            "sk_id": st.session_state["current_sk_id"],
                            **filter_params
                        }
                        resp = requests.get(f"{API_URL}/bivariate/{selected_table}", params=req_payload, timeout=20)
                        
                        if resp.status_code == 200:
                            biv_results = resp.json()
                            fig_biv = plot_bivariate_scatter(col_x, col_y, biv_results)
                            st.pyplot(fig_biv, clear_figure=True)
                            plt.close(fig_biv)
                            
                            # Render contextual readout summary
                            cx_val = biv_results.get("customer_x")
                            cy_val = biv_results.get("customer_y")
                            if cx_val is not None and cy_val is not None:
                                st.caption(f"Target Location Coordinates inside layout space — **{col_x}**: {cx_val:,.4g} · **{col_y}**: {cy_val:,.4g}")
                            else:
                                st.warning("Target Client lacks data for at least one chosen dimension. Background context rendered below.")
                        else:
                            st.error(f"Failed retrieval interface pipeline: Code {resp.status_code} - {resp.text}")
                    except requests.exceptions.RequestException as err:
                        st.error(f"Network transport pipeline disrupted: {err}")