import io
import base64                        
import numpy as np
import matplotlib.pyplot as plt
import shap
import matplotlib
matplotlib.use('Agg')

def get_importances(features_row, model):
    explainer = shap.TreeExplainer(model.named_steps['lgbm'])
    shap_values = explainer.shap_values(features_row)

    if isinstance(shap_values, list):
        sv = shap_values[0][0]
        if isinstance(explainer.expected_value, (list, np.ndarray)):
            ev = explainer.expected_value[0]
        else:
            ev = explainer.expected_value
    else:
        sv = shap_values[0]
        ev = explainer.expected_value


    local_importance = dict(zip(features_row.columns.tolist(), sv.tolist()))
    local_importance = dict(
        sorted(local_importance.items(), key=lambda x: abs(x[1]), reverse=True)
    )

    return ev, local_importance, sv


def plot(features_row, ev, importances, shap_values):
    exp = shap.Explanation(
        values=shap_values,
        base_values=ev,
        data=features_row.iloc[0].values,
        feature_names=features_row.columns.tolist()
    )

    fig, ax = plt.subplots()
    shap.waterfall_plot(exp, max_display=20, show=False)

    buf = io.BytesIO()
    plt.savefig(buf, format='png', bbox_inches='tight')
    plt.close()
    buf.seek(0)

    return base64.b64encode(buf.read()).decode('utf-8')  


def get_png(features_row, model):
    ev, importances, sv = get_importances(features_row, model)
    return plot(features_row, ev, importances, sv) 