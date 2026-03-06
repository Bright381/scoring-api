import matplotlib.pyplot as plt
import pandas
import shap
from fastapi.responses import StreamingResponse
import io


def get_importances(features_row, model):
    explainer = shap.TreeExplainer(model.named_steps['lgbm'])

    shap_values = explainer.shap_values(features_row)
    if isinstance(shap_values, list):
        sv = shap_values[1][0]
    else:
        sv = shap_values[0]

    local_importance = dict(zip(
        features_row.columns.tolist(),
        sv.tolist()
    ))
        
    # sort by absolute importance
    local_importance = dict(sorted(local_importance.items(), key=lambda x: abs(x[1]), reverse=True))
    
    ev = explainer.expected_value
    if hasattr(ev, '__len__'):
        ev = ev[1]

    return ev, local_importance, sv

def plot(features_row, ev, importances, shap_values):

    fig, ax = plt.subplots()
    shap.waterfall_plot(
        shap.Explanation(
            values=shap_values,
            base_values=ev,
            data=features_row.iloc[0],
            feature_names=features_row.columns.tolist()
        ),
        max_display=15,
        show=False
    )

    buf = io.BytesIO()
    plt.savefig(buf, format='png', bbox_inches='tight')
    plt.close()
    buf.seek(0)

    return StreamingResponse(buf, media_type="image/png")


def get_png(features_row, model):

    ev, importances, sv = get_importances(features_row, model)
    return plot(features_row, ev, importances, sv)