# scoring-api

## Live Link
* **Web UI:** [https://your-streamlit-app-link.streamlit.app/](https://your-streamlit-app-link.streamlit.app/)


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