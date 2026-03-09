from fastapi import FastAPI, HTTPException
import joblib
import pandas as pd
import json
import uvicorn
from utils.get_data import get_customer_features
from utils.get_shap import get_png

app = FastAPI(title="Home Credit Default Risk API")

DATA_FOLDER='data'

# load model
MODEL = joblib.load('api_model_info/model.pkl')

# get best threshold
with open('api_model_info/params/threshold.txt', 'rt') as f:
    threshold_value = float(f.read().strip())


@app.get('/')
def running():
    return "API is running."

@app.get("/predict/{sk_id}")
def predict(sk_id: int):
    try:
        # Transform ID into features
        features_row = get_customer_features(sk_id)

        if features_row is None or features_row.shape[0]==0:
            raise HTTPException(status_code=404, detail="Customer ID not found")

        # Predict
        probability = MODEL.predict_proba(features_row)[0][1]

        prediction = 1 if probability >= threshold_value else 0

        return {
            "sk_id": sk_id,
            "prediction": prediction,
            "probability": round(float(probability), 4),
            "threshold": round(threshold_value, 4),
            "status": "Rejected" if prediction == 1 else "Approved"
        }
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/explain/{sk_id}")
def explain(sk_id: int):
    try:
        features_row = get_customer_features(sk_id)
        if features_row is None or features_row.shape[0]==0:
            raise HTTPException(status_code=404, detail="Customer ID not found")
        
        return get_png(features_row, MODEL)

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/explore/{sk_id}")
def explore(sk_id: int):
    try:
        features_row = get_customer_features(sk_id)
        if features_row is None or features_row.shape[0]==0:       
            raise HTTPException(status_code=404, detail="Customer ID not found")

        dic = features_row.iloc[0].to_dict()
        dic = {k: (None if pd.isna(v) else v) for k, v in dic.items()}

        return dic

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)