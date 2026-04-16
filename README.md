# scoring-api

## Live Link
* **API:** [https://scoring-api-bhx1.onrender.com](https://scoring-api-bhx1.onrender.com)
* **Web UI:** [https://streamlit-ui-zuyw.onrender.com](https://streamlit-ui-zuyw.onrender.com)
**Click on the API link to start it up. Clicking only the Web UI will not start up the API.**

## Project Overview
This project implements a **Credit Scoring System** for a financial company. It predicts the probability of a customer defaulting on a loan and provides transparency through local/global feature importance.

## Architecture
1.  **ML Model:** LightGBM classification model optimized for "Business Cost".
2.  **API:** FastAPI endpoint serving real-time predictions and SHAP explanations.
3.  **Streamlit UI:** Streamlit interface for loan officers to visualize client data and scores.

## Structure

├── api_model_info/  # Model data
├── streamlit_ui/    # Streamlit UI code
├── utils/           # Data fecthing tools
└── data/            # Data sample