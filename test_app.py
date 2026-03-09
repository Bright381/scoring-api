import pytest
import json
import math
from fastapi.testclient import TestClient
from app import app, MODEL, threshold_value

client = TestClient(app)

VALID_ID   = 100001
INVALID_ID = 999999999


# One class per endpoint
# Local test until the api can have access to data without uploading it to github

# /predict

class TestPredict:

    def test_valid_id_returns_200(self):
        r = client.get(f"/predict/{VALID_ID}")
        assert r.status_code == 200

    def test_invalid_id_returns_404(self):
        r = client.get(f"/predict/{INVALID_ID}")
        assert r.status_code == 404

    def test_response_has_expected_fields(self):
        r = client.get(f"/predict/{VALID_ID}")
        data = r.json()
        assert "sk_id"       in data
        assert "prediction"  in data
        assert "probability" in data
        assert "threshold"   in data
        assert "status"      in data

    def test_probability_is_between_0_and_1(self):
        r = client.get(f"/predict/{VALID_ID}")
        prob = r.json()["probability"]
        assert 0.0 <= prob <= 1.0

    def test_prediction_is_binary(self):
        r = client.get(f"/predict/{VALID_ID}")
        assert r.json()["prediction"] in [0, 1]

    def test_threshold_is_between_0_and_1(self):
        r = client.get(f"/predict/{VALID_ID}")
        assert 0.0 <= r.json()["threshold"] <= 1.0

    def test_status_is_approved_or_rejected(self):
        r = client.get(f"/predict/{VALID_ID}")
        assert r.json()["status"] in ["Approved", "Rejected"]

    def test_status_consistent_with_prediction(self):
        r = client.get(f"/predict/{VALID_ID}")
        data = r.json()
        if data["prediction"] == 1:
            assert data["status"] == "Rejected"
        else:
            assert data["status"] == "Approved"

    def test_status_consistent_with_threshold(self):
        r = client.get(f"/predict/{VALID_ID}")
        data = r.json()
        expected = "Rejected" if data["probability"] >= data["threshold"] else "Approved"
        assert data["status"] == expected


# /explain

class TestExplain:

    def test_valid_id_returns_200(self):
        r = client.get(f"/explain/{VALID_ID}")
        assert r.status_code == 200

    def test_invalid_id_returns_404(self):
        r = client.get(f"/explain/{INVALID_ID}")
        assert r.status_code == 404

    def test_response_is_png(self):
        r = client.get(f"/explain/{VALID_ID}")
        assert r.headers["content-type"] == "image/png"

    def test_response_is_not_empty(self):
        r = client.get(f"/explain/{VALID_ID}")
        assert len(r.content) > 0

    def test_response_starts_with_png_signature(self):
        r = client.get(f"/explain/{VALID_ID}")
        # PNG files start with this 8-byte signature
        assert r.content[:8] == b'\x89PNG\r\n\x1a\n'


# /explore

class TestExplore:

    def test_valid_id_returns_200(self):
        r = client.get(f"/explore/{VALID_ID}")
        assert r.status_code == 200

    def test_invalid_id_returns_404(self):
        r = client.get(f"/explore/{INVALID_ID}")
        assert r.status_code == 404

    def test_response_is_dict(self):
        r = client.get(f"/explore/{VALID_ID}")
        assert isinstance(r.json(), dict)

    def test_response_is_not_empty(self):
        r = client.get(f"/explore/{VALID_ID}")
        assert len(r.json()) > 0

    def test_no_nan_values_in_response(self):
        r = client.get(f"/explore/{VALID_ID}")
        for k, v in r.json().items():
            assert v is None or not (isinstance(v, float) and math.isnan(v)), \
                f"NaN found in field: {k}"

    def test_response_has_numeric_values(self):
        r = client.get(f"/explore/{VALID_ID}")
        values = [v for v in r.json().values() if v is not None]
        assert all(isinstance(v, (int, float)) for v in values)


# quick model check

class TestModel:

    def test_model_is_loaded(self):
        assert MODEL is not None

    def test_threshold_is_valid(self):
        assert isinstance(threshold_value, float)
        assert 0.0 < threshold_value < 1.0

    def test_model_has_lgbm_step(self):
        assert "lgbm" in MODEL.named_steps

    def test_model_predict_proba_output_shape(self):
        import pandas as pd
        import numpy as np
        # build a dummy single row of zeros with the right feature names
        feature_names = MODEL.named_steps['lgbm'].booster_.feature_name()
        dummy = pd.DataFrame([np.zeros(len(feature_names))], columns=feature_names)
        proba = MODEL.predict_proba(dummy)
        assert proba.shape == (1, 2)

    def test_model_predict_proba_sums_to_1(self):
        import pandas as pd
        import numpy as np
        feature_names = MODEL.named_steps['lgbm'].booster_.feature_name()
        dummy = pd.DataFrame([np.zeros(len(feature_names))], columns=feature_names)
        proba = MODEL.predict_proba(dummy)
        assert abs(proba[0].sum() - 1.0) < 1e-5