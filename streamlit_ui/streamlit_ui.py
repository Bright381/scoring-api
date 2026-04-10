import streamlit as st
import requests
import pandas as pd
import base64
import os

API_URL = os.environ["API_URL"]

st.set_page_config(page_title="Credit Scoring", page_icon="🏦", layout="wide")

st.markdown("""
    <style>
        @import url('https://fonts.googleapis.com/css2?family=DM+Mono:wght@400;500&family=DM+Sans:wght@300;400;600&display=swap');
        html, body, [class*="css"] { font-family: 'DM Sans', sans-serif; }
        .title { font-family: 'DM Mono', monospace; font-size: 2rem; font-weight: 500; color: #f0f0f0; letter-spacing: -0.02em; margin-bottom: 0.2rem; }
        .subtitle { font-size: 0.9rem; color: #6b7280; margin-bottom: 2rem; font-weight: 300; }
        .result-card { background: #1a1d27; border-radius: 12px; padding: 1.5rem; border: 1px solid #2d3047; margin-bottom: 1rem; }
        .approved { border-left: 4px solid #22c55e; }
        .rejected { border-left: 4px solid #ef4444; }
        .metric-label { font-family: 'DM Mono', monospace; font-size: 0.75rem; color: #6b7280; text-transform: uppercase; letter-spacing: 0.08em; }
        .metric-value { font-size: 2rem; font-weight: 600; color: #f0f0f0; }
        .status-approved { color: #22c55e; font-size: 1.4rem; font-weight: 600; }
        .status-rejected { color: #ef4444; font-size: 1.4rem; font-weight: 600; }
        .section-title { font-family: 'DM Mono', monospace; font-size: 0.8rem; color: #6b7280; text-transform: uppercase; letter-spacing: 0.1em; margin-bottom: 1rem; padding-bottom: 0.5rem; border-bottom: 1px solid #2d3047; }
    </style>
""", unsafe_allow_html=True)

st.markdown('<div class="title">Credit Scoring Dashboard</div>', unsafe_allow_html=True)
st.markdown('<div class="subtitle">Home Credit Default Risk — Internal Risk Assessment Tool</div>', unsafe_allow_html=True)

sk_id = st.text_input("Customer ID (SK_ID_CURR)", placeholder="100001")
col1, col2 = st.columns([1, 1])
predict_btn = col1.button("Predict & Explain")
explore_btn = col2.button("Load Customer Data")
st.divider()

if predict_btn:
    if not sk_id:
        st.warning("Please enter a Customer ID.")
    else:
        with st.spinner("Running prediction..."):
            try:
                resp = requests.get(f"{API_URL}/predict/{sk_id}")
                if resp.status_code == 404:
                    st.error("Customer ID not found.")
                elif resp.status_code != 200:
                    st.error(f"API error: {resp.status_code}")
                else:
                    resp = resp.json()
                    image_bytes = base64.b64decode(resp['loc_imp'])
                    status_class = "approved" if resp["status"] == "Approved" else "rejected"
                    status_color = "status-approved" if resp["status"] == "Approved" else "status-rejected"
                    st.markdown(f"""
                        <div class="result-card {status_class}">
                            <div style="display:flex; justify-content:space-between; align-items:center;">
                                <div><div class="metric-label">Decision</div><div class="{status_color}">{resp["status"]}</div></div>
                                <div><div class="metric-label">Default Probability</div><div class="metric-value">{resp["probability"]:.1%}</div></div>
                                <div><div class="metric-label">Threshold</div><div class="metric-value">{resp["threshold"]:.1%}</div></div>
                            </div>
                        </div>
                    """, unsafe_allow_html=True)

                    st.markdown('<div class="section-title">Local Feature Importance</div>', unsafe_allow_html=True)
                    
                    st.image(image_bytes)
            except requests.exceptions.ConnectionError:
                st.error("Could not connect to API. Is it running?")

if explore_btn:
    if not sk_id:
        st.warning("Please enter a Customer ID.")
    else:
        with st.spinner("Loading customer data..."):
            try:
                explore_resp = requests.get(f"{API_URL}/explore/{sk_id}")
                if explore_resp.status_code == 404:
                    st.error("Customer ID not found.")
                elif explore_resp.status_code != 200:
                    st.error(f"API error: {explore_resp.status_code}")
                else:
                    data = explore_resp.json()
                    customer = pd.Series(data)

                    st.markdown('<div class="section-title">Key Metrics</div>', unsafe_allow_html=True)
                    m1, m2, m3, m4 = st.columns(4)
                    m1.metric("Age", f"{int(-customer.get('DAYS_BIRTH', 0) / 365)} yrs")
                    m2.metric("Income", f"${customer.get('AMT_INCOME_TOTAL', 0):,.0f}")
                    m3.metric("Credit", f"${customer.get('AMT_CREDIT', 0):,.0f}")
                    m4.metric("Annuity", f"${customer.get('AMT_ANNUITY', 0):,.0f}")
                    st.divider()

                    st.markdown('<div class="section-title">Column Explorer</div>', unsafe_allow_html=True)
                    all_cols = [k for k, v in data.items() if v is not None]
                    selected = st.multiselect("Select columns to inspect", options=all_cols)
                    if selected:
                        explorer_df = customer[selected].to_frame(name="Value").reset_index()
                        explorer_df.columns = ["Feature", "Value"]
                        st.dataframe(explorer_df, use_container_width=True, hide_index=True)

            except requests.exceptions.ConnectionError:
                st.error("Could not connect to API. Is it running?")

if "customer_data" in st.session_state:
    data = st.session_state["customer_data"]
    customer = pd.Series(data)

    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Age", f"{int(-customer.get('DAYS_BIRTH', 0) / 365)} yrs")
    m2.metric("Income", f"${customer.get('AMT_INCOME_TOTAL', 0):,.0f}")
    m3.metric("Credit", f"${customer.get('AMT_CREDIT', 0):,.0f}")
    m4.metric("Annuity", f"${customer.get('AMT_ANNUITY', 0):,.0f}")

    st.markdown("---")
    all_cols = [k for k, v in data.items() if v is not None]
    selected = st.multiselect("Select columns to display", options=all_cols)
    if selected:
        display = pd.DataFrame({"Column": selected, "Value": [customer[c] for c in selected]})
        st.dataframe(display, use_container_width=True, hide_index=True)