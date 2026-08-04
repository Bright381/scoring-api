from fastapi import FastAPI, HTTPException, Body, Query
import traceback
from pydantic import BaseModel, Field
from typing import Dict, Any
import joblib
import pandas as pd
import numpy as np
import base64
from utils.get_data import (
    fetch_unique_values,
    get_raw_tables_dic,
    TABLES,
    get_column_stats,
    fetch_target
)
from utils.get_shap import get_importances, plot
from utils.single_row_preprocessing import preprocess, apply_custom_values


app = FastAPI(title="Home Credit Default Risk API")

# load model
LGBM_MODEL = joblib.load('api_model_info/model.pkl').named_steps['lgbm']

# get best threshold
with open('api_model_info/params/threshold.txt', 'rt') as f:
    threshold_value = float(f.read().strip())

# global feature importance
with open("api_model_info/lgbm_importances.png", "rb") as image_file:
    global_imp = base64.b64encode(image_file.read()).decode('utf-8')


@app.get('/check_api')
def running():
    return "API is running."

@app.get("/get_target/{sk_id}")
def get_target(sk_id: int):
    return fetch_target(sk_id)
    
@app.get("/predict/{sk_id}")
def predict(sk_id: int):
    try:
        # Get raw tables for the customer
        raw_tables_dict = get_raw_tables_dic(sk_id)

        if raw_tables_dict is None or all(df.empty for df in raw_tables_dict.values()):
            raise HTTPException(status_code=404, detail="Customer ID not found")

        customer_features = preprocess(raw_tables_dict, sk_id)

        for col in customer_features.columns:
            customer_features[col] = pd.to_numeric(customer_features[col], errors='coerce')

        # List of features the model expects
        customer_features = customer_features.drop(columns=['SK_ID_CURR', 'TARGET', 'Unnamed: 0'], errors='ignore')
        expected_features = LGBM_MODEL.feature_name_
        expected_truncated = [f[:63] for f in expected_features]

        customer_features = customer_features[expected_features]

        probability = LGBM_MODEL.predict_proba(customer_features)[0][1]
        prediction = 1 if probability >= threshold_value else 0

        ev, importances, sv = get_importances(customer_features, LGBM_MODEL)

        return {
            "sk_id": sk_id,
            "prediction": prediction,
            "probability": round(float(probability), 4),
            "threshold": round(threshold_value, 4),
            "status": "Rejected" if prediction == 0 else "Approved",
            "loc_imp": plot(customer_features, ev, importances, sv),
            "global_imp": global_imp
        }
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=traceback.format_exc())


@app.get("/feature-values/{table}")
def get_feature_values(
    table: str,
    column: str = Query(..., description="Column name")
):
    """
    Return all distinct values for a column.

    Example response:
    {
        "values": [
            "Cash loans",
            "Revolving loans"
        ]
    }
    """

    if table not in TABLES:
        raise HTTPException(
            status_code=400,
            detail=f"Unknown table '{table}'."
        )

    try:
        values = fetch_unique_values(table, column)
        return values

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=str(e)
        )

class FeatureOverrides(BaseModel):
    overrides: Dict[str, Any] = Field(default_factory=dict)

@app.post("/custom_predict/{sk_id}")
def custom_predict(sk_id: int, overrides: FeatureOverrides = None):
    try:
        # Get raw tables for the customer
        raw_tables_dict = get_raw_tables_dic(sk_id)

        if raw_tables_dict is None or all(df.empty for df in raw_tables_dict.values()):
            raise HTTPException(status_code=404, detail="Customer ID not found")

        override_dict = overrides.model_dump(exclude_none=True)
        raw_custom_features_dict = apply_custom_values(raw_tables_dict, override_dict['overrides'])
        customer_features = preprocess(raw_custom_features_dict, sk_id)

        for col in customer_features.columns:
            customer_features[col] = pd.to_numeric(customer_features[col], errors='coerce')

        # List of features the model expects
        customer_features = customer_features.drop(columns=['SK_ID_CURR', 'TARGET', 'Unnamed: 0'], errors='ignore')
        expected_features = LGBM_MODEL.feature_name_
        expected_truncated = [f[:63] for f in expected_features]

        customer_features = customer_features[expected_features]

        probability = LGBM_MODEL.predict_proba(customer_features)[0][1]
        prediction = 1 if probability >= threshold_value else 0

        ev, importances, sv = get_importances(customer_features, LGBM_MODEL)

        return {
            "sk_id": sk_id,
            "prediction": prediction,
            "probability": round(float(probability), 4),
            "threshold": round(threshold_value, 4),
            "status": "Rejected" if prediction == 0 else "Approved",
            "loc_imp": plot(customer_features, ev, importances, sv),
            "global_imp": global_imp,
            "overrides_used": override_dict,
        }
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=traceback.format_exc())


@app.get("/explore/{sk_id}")
def explore(sk_id: int):
    try:
        tables_dic = get_raw_tables_dic(sk_id)
        response_data: Dict[str, Dict[str, Any]] = {}
 
        for table_name, df in tables_dic.items():
            if not isinstance(df, pd.DataFrame) or df.empty:
                continue
 
            row_dict = df.iloc[0].to_dict()
            table_data: Dict[str, Any] = {}
 
            for col, val in row_dict.items():
                if pd.isna(val):
                    table_data[col] = None
                elif isinstance(val, (np.integer, int)):
                    table_data[col] = int(val)
                elif isinstance(val, (np.floating, float)):
                    table_data[col] = float(val)
                else:
                    table_data[col] = str(val)
 
            if table_data:
                response_data[table_name] = table_data
 
        if not response_data:
            raise HTTPException(status_code=404, detail="Customer ID not found")
 
        return response_data
 
    except HTTPException:
        raise
    except Exception as e:
        print(f"Explore Endpoint Crash: {repr(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/distributions/{table}")
def distributions(
    table: str,
    column: str = Query(..., description="Column name to compute the distribution for"),
    sk_id: int = Query(..., description="Customer SK_ID_CURR"),
    filter_col: str = Query(None, description="Optional column name to filter the baseline population by"),
    filter_val: str = Query(None, description="Optional baseline column matching criterion value"),
):
    """
    Returns histogram data for `column` in `table` together with the
    customer's own value and their percentile rank, optionally filtered.
    """
    if table not in TABLES:
        raise HTTPException(status_code=400, detail=f"Unknown table '{table}'.")
 
    try:
        stats = get_column_stats(table, column, sk_id, filter_col, filter_val)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
 
    if stats["n"] == 0:
        raise HTTPException(
            status_code=404,
            detail=f"No data found for column '{column}' in table '{table}'.",
        )
 
    return stats

@app.get("/bivariate/{table}")
def bivariate_distribution(
    table: str,
    col_x: str = Query(..., description="Column name for the X axis representation"),
    col_y: str = Query(..., description="Column name for the Y axis representation"),
    sk_id: int = Query(..., description="Customer SK_ID_CURR identification"),
    filter_col: str = Query(None, description="Optional column name to filter the baseline population by"),
    filter_val: str = Query(None, description="Optional baseline column matching criterion value"),
):
    """
    Returns scatter coordinate groups for col_x and col_y within table, alongside 
    the specialized client coordinates and active sample counts.
    """
    if table not in TABLES:
        raise HTTPException(status_code=400, detail=f"Unknown table '{table}'.")
 
    try:
        from utils.get_data import get_bivariate_data
        stats = get_bivariate_data(table, col_x, col_y, sk_id, filter_col, filter_val)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
 
    if stats["n"] == 0:
        raise HTTPException(
            status_code=404,
            detail=f"No matching base distribution records found for features inside table '{table}'.",
        )
 
    return stats
