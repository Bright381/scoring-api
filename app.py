from fastapi import FastAPI, HTTPException, Body
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
        customer_features = customer_features.drop(columns=['SK_ID_CURR', 'TARGET'])
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
        tables_dic = get_raw_features(sk_id)
        ########""
        # if customer_features is None or customer_features.shape[0]==0:       
        #     raise HTTPException(status_code=404, detail="Customer ID not found")
        
        ###### MAKE PLOTS HERE ?
        return tables_dic

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))