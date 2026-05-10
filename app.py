from fastapi import FastAPI, HTTPException, Body, Query
from pydantic import BaseModel, Field
from typing import Optional, Dict, Any
import joblib
import pandas as pd
import json
from utils.get_data import (
    get_preprocessed_features,
    get_custom_features,
    get_raw_tables_dic
)
from utils.get_shap import get_png, get_importances, plot
from utils.single_row_preprocessing import preprocess
import base64

app = FastAPI(title="Home Credit Default Risk API")

# load model
MODEL = joblib.load('api_model_info/model.pkl')

# get best threshold
with open('api_model_info/params/threshold.txt', 'rt') as f:
    threshold_value = float(f.read().strip())

    # global feature importance
with open("api_model_info/lgbm_importances.png", "rb") as image_file:
        global_imp = base64.b64encode(image_file.read()).decode('utf-8')


@app.get('/check_api')
def running():
    return "API is running."

@app.get("/predict/{sk_id}")
def predict(sk_id: int):
    try:
        # Transform ID into features
        customer_features = get_preprocessed_features(sk_id)

        if customer_features is None or customer_features.shape[0]==0:
            raise HTTPException(status_code=404, detail="Customer ID not found")

        # Predict
        customer_features = customer_features.drop(columns=['SK_ID_CURR', 'TARGET', 'Unnamed: 0'], errors='ignore')

        #############################################
        # List of features the model expects
        expected_features = MODEL.named_steps['lgbm'].feature_name_
        expected_truncated = [f[:63] for f in expected_features]
        customer_features = customer_features[expected_truncated]

        # List of columns currently in your dataframe
        # current = customer_features.columns.tolist()

        # # Find columns that are in the dataframe but NOT in the model
        # extra_cols = [c for c in current if c not in expected]

        # # Find columns the model wants but are MISSING from the dataframe
        # missing_cols = [c for c in expected if c not in current]

        # print(f"DEBUG: Found {len(current)} columns.")
        # print(f"DEBUG: Extra columns (779 vs 777): {extra_cols}")
        # print(f"DEBUG: Missing columns: {missing_cols}")

        probability = MODEL.named_steps['lgbm'].predict_proba(customer_features)[0][1]

        prediction = 1 if probability >= threshold_value else 0

        ev, importances, sv = get_importances(customer_features, MODEL)
        
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
        raise HTTPException(status_code=500, detail=str(e))
    

class FeatureOverrides(BaseModel):
    overrides: Dict[str, Any] = Field(default_factory=dict)


@app.post("/custom_predict/{sk_id}")
def custom_predict(sk_id: int, overrides: FeatureOverrides = None):
    try:
        if overrides is None:
            customer_features = get_preprocessed_features(sk_id)
            if customer_features is None or customer_features.shape[0] == 0:
                raise HTTPException(status_code=404, detail="Customer ID not found")

        else:
            override_dict = overrides.model_dump(exclude_none=True)
            raw_tables_dict = get_raw_tables_dic(sk_id)
            ##### TO REWRITE
            # if raw_tables_dict is None or customer_raw_features.shape[0] == 0:
            #     raise HTTPException(status_code=404, detail="Customer ID not found")

            customer_features = preprocess(customer_raw_features)

        probability = MODEL.named_steps['lgbm'].predict_proba(customer_features)[0][1]
        prediction = 1 if probability >= threshold_value else 0

        ev, importances, sv = get_importances(customer_features, MODEL)

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
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/explore/{sk_id}")
def explore(sk_id: int):
    try:
        tables_dic = get_raw_tables_dic(sk_id)
        response_data: Dict[str, Dict[str, Any]] = {}
 
        for table_name, df in tables_dic.items():
            if df is None or df.empty:
                continue
 
            row_dict = df.iloc[0].to_dict()
            table_data: Dict[str, Any] = {}
 
            for col, val in row_dict.items():
                try:
                    is_na = pd.isna(val)
                except (TypeError, ValueError):
                    is_na = False
 
                if is_na:
                    table_data[col] = None
                else:
                    try:
                        table_data[col] = float(val) if '.' in str(val) else int(val)
                    except (TypeError, ValueError):
                        table_data[col] = str(val)
 
            if table_data:
                response_data[table_name] = table_data
 
        if not response_data:
            raise HTTPException(status_code=404, detail="Customer ID not found")
 
        return response_data
 
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/distributions/{table}")
def distributions(
    table: str,
    column: str = Query(..., description="Column name to compute the distribution for"),
    sk_id: int = Query(..., description="Customer SK_ID_CURR"),
):
    """
    Returns histogram data for `column` in `table` together with the
    customer's own value and their percentile rank.
    """
    if table not in TABLES:
        raise HTTPException(status_code=400, detail=f"Unknown table '{table}'.")
 
    try:
        stats = get_column_stats(table, column, sk_id)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
 
    if stats["n"] == 0:
        raise HTTPException(
            status_code=404,
            detail=f"No data found for column '{column}' in table '{table}'.",
        )
 
    return stats
