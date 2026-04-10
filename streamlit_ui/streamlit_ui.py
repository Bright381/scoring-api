import os
import streamlit as st
import requests
import pandas as pd
import base64

API_URL = os.environ["API_URL"]

st.set_page_config(page_title="Credit Scoring", page_icon="🏦", layout="wide")

# --- CSS Styling Adjustments ---
st.markdown("""
    <style>
        @import url('https://fonts.googleapis.com/css2?family=DM+Mono:wght@400;500&family=DM+Sans:wght@300;400;600&display=swap');
        html, body, [class*="css"] { font-family: 'DM Sans', sans-serif; background-color: #121218; color: #e0e0e0; }
        .title { font-family: 'DM Mono', monospace; font-size: 2rem; font-weight: 500; color: #ffffff; letter-spacing: -0.02em; margin-bottom: 0.2rem; }
        .subtitle { font-size: 0.9rem; color: #a0a0b0; margin-bottom: 2rem; font-weight: 300; }
        .result-card { background: #1e212c; border-radius: 12px; padding: 1.5rem; border: 1px solid #3a3f58; margin-bottom: 1rem; box-shadow: 0 4px 12px rgba(0, 0, 0, 0.3); }
        .approved { border-left: 4px solid #10b981; } /* Emerald Green */
        .rejected { border-left: 4px solid #ef4444; } /* Red */
        .metric-label { font-family: 'DM Mono', monospace; font-size: 0.75rem; color: #a0a0b0; text-transform: uppercase; letter-spacing: 0.08em; }
        .metric-value { font-size: 2rem; font-weight: 600; color: #ffffff; }
        .status-approved { color: #10b981; font-size: 1.4rem; font-weight: 600; }
        .status-rejected { color: #ef4444; font-size: 1.4rem; font-weight: 600; }
        .section-title { font-family: 'DM Mono', monospace; font-size: 0.8rem; color: #a0a0b0; text-transform: uppercase; letter-spacing: 0.1em; margin-bottom: 1rem; padding-bottom: 0.5rem; border-bottom: 1px solid #3a3f58; }
    </style>
""", unsafe_allow_html=True)

st.markdown('<div class="title">Credit Scoring Dashboard</div>', unsafe_allow_html=True)
st.markdown('<div class="subtitle">Home Credit Default Risk — Internal Risk Assessment Tool</div>', unsafe_allow_html=True)

# --- Input and Control Columns ---
sk_id = st.text_input("Customer ID (SK_ID_CURR)", placeholder="100001")
col1, col2, col3 = st.columns([1, 1, 0.5]) # Adjusted layout for the new button

predict_btn = col1.button("Predict & Explain", type="primary")
explore_btn = col2.button("Load Customer Data")
check_api_btn = col3.button("Check API Health") # New Button

st.divider()

# --- API Health Check Logic ---
if check_api_btn:
    with st.spinner("Checking API status..."):
        try:
            resp = requests.get(f"{API_URL}/check_api")
            if resp.status_code == 200 and "API is running" in resp.text:
                st.success("API Health Check Successful: API is running.")
            else:
                st.error(f"API Health Check Failed (Status {resp.status_code}): Could not confirm API status.")
        except requests.exceptions.ConnectionError:
            st.error("Connection Error: Could not reach the API endpoint.")

# --- Prediction Logic ---
if predict_btn:
    if not sk_id:
        st.warning("Please enter a Customer ID to run prediction.")
    else:
        with st.spinner("Running prediction..."):
            try:
                resp = requests.get(f"{API_URL}/predict/{sk_id}")
                if resp.status_code == 404:
                    st.error("Customer ID not found in the system.")
                elif resp.status_code != 200:
                    st.error(f"API error occurred: Status Code {resp.status_code}")
                else:
                    resp = resp.json()
                    image_bytes = base64.b64decode(resp['loc_imp'])
                    status_class = "approved" if resp["status"] == "Approved" else "rejected"
                    status_color = "status-approved" if resp["status"] == "Approved" else "status-rejected"

                    st.markdown(f"""
                        <div class="result-card {status_class}">
                            <div style="display:flex; justify-content:space-between; align-items:center;">
                                <div><div class="metric-label">Decision</div><div class="{status_color}">{resp["status"]}</div></div>
                                <div><div class="metric-label">Default Probability</div><div class="metric-value">{resp["probability"]:.4f}</div></div>
                                <div><div class="metric-label">Threshold</div><div class="metric-value">{resp["threshold"]:.4f}</div></div>
                            </div>
                        </div>
                    """, unsafe_allow_html=True)

                    st.markdown('<div class="section-title">Local Feature Importance</div>', unsafe_allow_html=True)
                    st.image(image_bytes, caption="Feature Importance Plot")

            except requests.exceptions.ConnectionError:
                st.error("Could not connect to API. Please ensure the service is running.")

# --- Exploration Logic (Initial Load) ---
if explore_btn:
    if not sk_id:
        st.warning("Please enter a Customer ID to load data.")
    else:
        with st.spinner("Loading customer data..."):
            try:
                explore_resp = requests.get(f"{API_URL}/explore/{sk_id}")
                if explore_resp.status_code == 404:
                    st.error("Customer ID not found.")
                elif explore_resp.status_code != 200:
                    st.error(f"API error occurred: Status Code {explore_resp.status_code}")
                else:
                    data = explore_resp.json()
                    customer = pd.Series(data)
                    # Store data in session state for potential later use/display consistency
                    st.session_state["customer_data"] = data 

                    st.markdown('<div class="section-title">Key Metrics</div>', unsafe_allow_html=True)
                    m1, m2, m3, m4 = st.columns(4)
                    # Using .get() with default values for safety
                    age = int(-customer.get('DAYS_BIRTH', 0) / 365) if customer.get('DAYS_BIRTH') is not None else 'N/A'
                    income = f"${customer.get('AMT_INCOME_TOTAL', 0):,.0f}" if customer.get('AMT_INCOME_TOTAL') is not None else "N/A"
                    credit = f"${customer.get('AMT_CREDIT', 0):,.0f}" if customer.get('AMT_CREDIT') is not None else "N/A"
                    annuity = f"${customer.get('AMT_ANNUITY', 0):,.0f}" if customer.get('AMT_ANNUITY') is not None else "N/A"

                    m1.metric("Age", str(age))
                    m2.metric("Income", income)
                    m3.metric("Credit", credit)
                    m4.metric("Annuity", annuity)
                    st.divider()

                    st.markdown('<div class="section-title">Column Explorer</div>', unsafe_allow_html=True)
                    all_cols = [k for k, v in data.items() if v is not None and k != 'SK_ID_CURR'] # Exclude ID from general explorer list
                    selected = st.multiselect("Select columns to inspect", options=all_cols)
                    if selected:
                        # Create a DataFrame for display, handling potential non-numeric types gracefully
                        explorer_data = {col: customer[col] for col in selected}
                        explorer_df = pd.DataFrame(list(explorer_data.items()), columns=["Feature", "Value"])
                        st.dataframe(explorer_df, use_container_width=True, hide_index=True)

            except requests.exceptions.ConnectionError:
                st.error("Could not connect to API. Is it running?")


# --- Session State Display (If data was loaded previously or via explore button) ---
if "customer_data" in st.session_state and sk_id == st.session_state["current_sk_id"]:
    data = st.session_state["customer_data"]
    customer = pd.Series(data)

    st.markdown("---")
    st.markdown('<div class="section-title">Session Data Review</div>', unsafe_allow_html=True)
    m1, m2, m3, m4 = st.columns(4)
    age = int(-customer.get('DAYS_BIRTH', 0) / 365) if customer.get('DAYS_BIRTH') is not None else 'N/A'
    income = f"${customer.get('AMT_INCOME_TOTAL', 0):,.0f}" if customer.get('AMT_INCOME_TOTAL') is not None else "N/A"
    credit = f"${customer.get('AMT_CREDIT', 0):,.0f}" if customer.get('AMT_CREDIT') is not None else "N/A"
    annuity = f"${customer.get('AMT_ANNUITY', 0):,.0f}" if customer.get('AMT_ANNUITY') is not None else "N/A"

    m1.metric("Age", str(age))
    m2.metric("Income", income)
    m3.metric("Credit", credit)
    m4.metric("Annuity", annuity)

    all_cols = [k for k, v in data.items() if v is not None and k != 'SK_ID_CURR']
    selected = st.multiselect("Select columns to display from loaded data", options=all_cols)
    if selected:
        display = pd.DataFrame({"Column": selected, "Value": [customer[c] for c in selected]})
        st.dataframe(display, use_container_width=True, hide_index=True)

# Update session state tracking if a new ID is entered manually after initial load
if sk_id and st.session_state.get("current_sk_id") != sk_id:
    st.session_state["current_sk_id"] = sk_id