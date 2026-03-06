import pandas as pd

PREPROCESSED_DATA='data/preprocessed_test.csv'

def get_customer_features(id, path=PREPROCESSED_DATA):
    
    df=pd.read_csv(path)
    df=df[df['SK_ID_CURR']==id].copy()
    feats = [f for f in df.columns if f not in ['TARGET', 'SK_ID_CURR', 'SK_ID_BUREAU', 'SK_ID_PREV', 'index']]
    df=df[feats]

    return df
