import pandas as pd
import pickle
import numpy as np
import json
from typing import Optional
# from sklearn import set_config
# set_config(transform_output="pandas")

# preprocess
def clean_df(df):
    df.columns = ["".join(c if c.isalnum() else "_" for c in str(x)) for x in df.columns]
    df=df.replace([np.inf, -np.inf], np.nan)
    for col in df.columns:
        if col not in ['TARGET', 'SK_ID_CURR', 'index']:
            df[col] = pd.to_numeric(df[col]).astype(np.float32)
    return df

def one_hot_encoder(df: pd.DataFrame, name: str, sk_id):

    categorical_columns = [col for col in df.columns if df[col].dtype == 'object']
    if len(categorical_columns)>0:

        with open(f'api_model_info/params/preproc/OHE_{name}.pkl', 'rb') as f:
            ohe = pickle.load(f)

        ohe.set_params(handle_unknown='ignore')

        if ohe is None or not hasattr(ohe, 'feature_names_in_'):
            raise ValueError('No attribute feature_names_in_ !')
        categorical_columns = list(ohe.feature_names_in_)

        if df.shape[0]==0:
            df.loc[0]=np.nan
            df.loc[0, 'SK_ID_CURR']=sk_id

        for col in categorical_columns:
            df[col] = df[col].astype(object)
            df[col] = df[col].where(pd.notnull(df[col]), other=np.nan)

        encoded = ohe.transform(df[categorical_columns])
        encoded_df = pd.DataFrame(encoded, columns=ohe.get_feature_names_out(), index=df.index)

        df = pd.concat([df.drop(columns=categorical_columns), encoded_df], axis=1)
        return df, list(encoded_df.columns)
    
    print(f'skipped ohe for {name}', flush=True)
    return df, []

# mapping for binary categorical variables
def apply_binary_maps(df: pd.DataFrame) -> pd.DataFrame:
    if df is None or df.empty:
        return df
    
    with open("api_model_info/params/preproc/bin_map.json") as f:
        BIN_MAP = json.load(f)
    df = df.copy()
    for col, mapping in BIN_MAP.items():
        if col in df.columns:
            df[col] = df[col].map(mapping)
    return df


# Preprocess application_train.csv and application_test.csv
def application_train_test(df, sk_id: int, overrides: Optional[int] = None):
    overrides = overrides or {}
    # Optional: Remove 4 applications with XNA CODE_GENDER (train set)
    df = df[df['CODE_GENDER'] != 'XNA']

    # df = df[df['SK_ID_CURR']==sk_id].copy()
    
    if df.empty:
        return df

    for k, v in overrides.items():
        if k in df.columns:
            df.loc[:, k] = v

    df=apply_binary_maps(df)
    
    # Categorical features with One-Hot encode
    df, _ = one_hot_encoder(df, 'app_train_test', sk_id)
    
    # NaN values for DAYS_EMPLOYED: 365.243 -> nan
    df['DAYS_EMPLOYED']=df['DAYS_EMPLOYED'].replace(365243, np.nan)
    # Some simple new features (percentages)
    df['DAYS_EMPLOYED_PERC'] = df['DAYS_EMPLOYED'] / df['DAYS_BIRTH']
    df['INCOME_CREDIT_PERC'] = df['AMT_INCOME_TOTAL'] / df['AMT_CREDIT']
    df['INCOME_PER_PERSON'] = df['AMT_INCOME_TOTAL'] / df['CNT_FAM_MEMBERS']
    df['ANNUITY_INCOME_PERC'] = df['AMT_ANNUITY'] / df['AMT_INCOME_TOTAL']
    df['PAYMENT_RATE'] = df['AMT_ANNUITY'] / df['AMT_CREDIT']
    df = clean_df(df)
    return df

# Preprocess bureau.csv and bureau_balance.csv
def bureau_and_balance(bureau, bureau_balance, sk_id: int, num_rows = None):
    bb = bureau_balance

    bureau_ids = bureau["SK_ID_BUREAU"].unique()
    bb = bb[bb["SK_ID_BUREAU"].isin(bureau_ids)].copy()

    bb, bb_cat = one_hot_encoder(bb, 'bb', np.nan)
    bureau, bureau_cat = one_hot_encoder(bureau, 'bureau', sk_id)

    # Bureau balance: Perform aggregations and merge with bureau.csv
    bb_aggregations = {'MONTHS_BALANCE': ['min', 'max', 'size']}
    for col in bb_cat:
        bb_aggregations[col] = ['mean']
    bb_agg = bb.groupby('SK_ID_BUREAU').agg(bb_aggregations)
    bb_agg.columns = pd.Index([e[0] + "_" + e[1].upper() for e in bb_agg.columns.tolist()])
    bureau = bureau.join(bb_agg, how='left', on='SK_ID_BUREAU')
    bureau=bureau.drop(['SK_ID_BUREAU'], axis=1)
    del bb, bb_agg

    # Bureau and bureau_balance numeric features
    num_aggregations = {
        'DAYS_CREDIT': ['min', 'max', 'mean', 'var'],
        'DAYS_CREDIT_ENDDATE': ['min', 'max', 'mean'],
        'DAYS_CREDIT_UPDATE': ['mean'],
        'CREDIT_DAY_OVERDUE': ['max', 'mean'],
        'AMT_CREDIT_MAX_OVERDUE': ['mean'],
        'AMT_CREDIT_SUM': ['max', 'mean', 'sum'],
        'AMT_CREDIT_SUM_DEBT': ['max', 'mean', 'sum'],
        'AMT_CREDIT_SUM_OVERDUE': ['mean'],
        'AMT_CREDIT_SUM_LIMIT': ['mean', 'sum'],
        'AMT_ANNUITY': ['max', 'mean'],
        'CNT_CREDIT_PROLONG': ['sum'],
        'MONTHS_BALANCE_MIN': ['min'],
        'MONTHS_BALANCE_MAX': ['max'],
        'MONTHS_BALANCE_SIZE': ['mean', 'sum']
    }
    # Bureau and bureau_balance categorical features
    cat_aggregations = {}
    for cat in bureau_cat:
        cat_aggregations[cat] = ['mean']
    for cat in bb_cat:
        cat_aggregations[cat + "_MEAN"] = ['mean']
    
    bureau_agg = bureau.groupby('SK_ID_CURR').agg({**num_aggregations, **cat_aggregations})
    bureau_agg.columns = pd.Index(['BURO_' + e[0] + "_" + e[1].upper() for e in bureau_agg.columns.tolist()])
    # Bureau: Active credits - using only numerical aggregations
    active = bureau[bureau['CREDIT_ACTIVE_Active'] == 1]
    active_agg = active.groupby('SK_ID_CURR').agg(num_aggregations)
    active_agg.columns = pd.Index(['ACTIVE_' + e[0] + "_" + e[1].upper() for e in active_agg.columns.tolist()])
    bureau_agg = bureau_agg.join(active_agg, how='left', on='SK_ID_CURR')
    del active, active_agg
    # Bureau: Closed credits - using only numerical aggregations
    closed = bureau[bureau['CREDIT_ACTIVE_Closed'] == 1]
    closed_agg = closed.groupby('SK_ID_CURR').agg(num_aggregations)
    closed_agg.columns = pd.Index(['CLOSED_' + e[0] + "_" + e[1].upper() for e in closed_agg.columns.tolist()])
    bureau_agg = bureau_agg.join(closed_agg, how='left', on='SK_ID_CURR')
    del closed, closed_agg, bureau
    bureau_agg = clean_df(bureau_agg)
    return bureau_agg

# Preprocess previous_applications.csv
def previous_applications(prev, sk_id: int, num_rows = None):
    if prev.empty:
        return pd.DataFrame(index=[sk_id])

    prev, cat_cols = one_hot_encoder(prev, 'prev', sk_id)
    # Days 365.243 values -> nan
    prev['DAYS_FIRST_DRAWING']=prev['DAYS_FIRST_DRAWING'].replace(365243, np.nan)
    prev['DAYS_FIRST_DUE']=prev['DAYS_FIRST_DUE'].replace(365243, np.nan)
    prev['DAYS_LAST_DUE_1ST_VERSION']=prev['DAYS_LAST_DUE_1ST_VERSION'].replace(365243, np.nan)
    prev['DAYS_LAST_DUE']=prev['DAYS_LAST_DUE'].replace(365243, np.nan)
    prev['DAYS_TERMINATION']=prev['DAYS_TERMINATION'].replace(365243, np.nan)
    # Add feature: value ask / value received percentage
    prev['APP_CREDIT_PERC'] = prev['AMT_APPLICATION'] / prev['AMT_CREDIT']
    # Previous applications numeric features
    num_aggregations = {
        'AMT_ANNUITY': ['min', 'max', 'mean'],
        'AMT_APPLICATION': ['min', 'max', 'mean'],
        'AMT_CREDIT': ['min', 'max', 'mean'],
        'APP_CREDIT_PERC': ['min', 'max', 'mean', 'var'],
        'AMT_DOWN_PAYMENT': ['min', 'max', 'mean'],
        'AMT_GOODS_PRICE': ['min', 'max', 'mean'],
        'HOUR_APPR_PROCESS_START': ['min', 'max', 'mean'],
        'RATE_DOWN_PAYMENT': ['min', 'max', 'mean'],
        'DAYS_DECISION': ['min', 'max', 'mean'],
        'CNT_PAYMENT': ['mean', 'sum'],
    }
    # Previous applications categorical features
    cat_aggregations = {}
    for cat in cat_cols:
        cat_aggregations[cat] = ['mean']
    
    prev_agg = prev.groupby('SK_ID_CURR').agg({**num_aggregations, **cat_aggregations})
    prev_agg.columns = pd.Index(['PREV_' + e[0] + "_" + e[1].upper() for e in prev_agg.columns.tolist()])
    # Previous Applications: Approved Applications - only numerical features
    approved = prev[prev['NAME_CONTRACT_STATUS_Approved'] == 1]
    approved_agg = approved.groupby('SK_ID_CURR').agg(num_aggregations)
    approved_agg.columns = pd.Index(['APPROVED_' + e[0] + "_" + e[1].upper() for e in approved_agg.columns.tolist()])
    prev_agg = prev_agg.join(approved_agg, how='left', on='SK_ID_CURR')
    # Previous Applications: Refused Applications - only numerical features
    refused = prev[prev['NAME_CONTRACT_STATUS_Refused'] == 1]
    refused_agg = refused.groupby('SK_ID_CURR').agg(num_aggregations)
    refused_agg.columns = pd.Index(['REFUSED_' + e[0] + "_" + e[1].upper() for e in refused_agg.columns.tolist()])
    prev_agg = prev_agg.join(refused_agg, how='left', on='SK_ID_CURR')
    del refused, refused_agg, approved, approved_agg, prev
    prev_agg = clean_df(prev_agg)
    return prev_agg

# Preprocess POS_CASH_balance.csv
def pos_cash(pos, sk_id: int, num_rows = None):

    pos, cat_cols = one_hot_encoder(pos, 'pos', sk_id)
    # Features
    aggregations = {
        'MONTHS_BALANCE': ['max', 'mean', 'size'],
        'SK_DPD': ['max', 'mean'],
        'SK_DPD_DEF': ['max', 'mean']
    }
    for cat in cat_cols:
        aggregations[cat] = ['mean']
    
    pos_agg = pos.groupby('SK_ID_CURR').agg(aggregations)
    pos_agg.columns = pd.Index(['POS_' + e[0] + "_" + e[1].upper() for e in pos_agg.columns.tolist()])
    # Count pos cash accounts
    pos_agg['POS_COUNT'] = pos.groupby('SK_ID_CURR').size()
    del pos
    pos_agg = clean_df(pos_agg)
    return pos_agg
    
# Preprocess installments_payments.csv
def installments_payments(ins, sk_id: int, num_rows = None):
    ins, cat_cols = one_hot_encoder(ins, 'ins', sk_id)
    # Percentage and difference paid in each installment (amount paid and installment value)
    ins['PAYMENT_PERC'] = ins['AMT_PAYMENT'] / ins['AMT_INSTALMENT']
    ins['PAYMENT_DIFF'] = ins['AMT_INSTALMENT'] - ins['AMT_PAYMENT']
    # Days past due and days before due (no negative values)
    ins['DPD'] = ins['DAYS_ENTRY_PAYMENT'] - ins['DAYS_INSTALMENT']
    ins['DBD'] = ins['DAYS_INSTALMENT'] - ins['DAYS_ENTRY_PAYMENT']
    ins['DPD'] = ins['DPD'].apply(lambda x: x if x > 0 else 0)
    ins['DBD'] = ins['DBD'].apply(lambda x: x if x > 0 else 0)
    # Features: Perform aggregations
    aggregations = {
        'NUM_INSTALMENT_VERSION': ['nunique'],
        'DPD': ['max', 'mean', 'sum'],
        'DBD': ['max', 'mean', 'sum'],
        'PAYMENT_PERC': ['max', 'mean', 'sum', 'var'],
        'PAYMENT_DIFF': ['max', 'mean', 'sum', 'var'],
        'AMT_INSTALMENT': ['max', 'mean', 'sum'],
        'AMT_PAYMENT': ['min', 'max', 'mean', 'sum'],
        'DAYS_ENTRY_PAYMENT': ['max', 'mean', 'sum']
    }
    for cat in cat_cols:
        aggregations[cat] = ['mean']
    ins_agg = ins.groupby('SK_ID_CURR').agg(aggregations)
    ins_agg.columns = pd.Index(['INSTAL_' + e[0] + "_" + e[1].upper() for e in ins_agg.columns.tolist()])
    # Count installments accounts
    ins_agg['INSTAL_COUNT'] = ins.groupby('SK_ID_CURR').size()
    del ins
    ins_agg = clean_df(ins_agg)
    return ins_agg

# Preprocess credit_card_balance.csv
def credit_card_balance(cc, sk_id: int, num_rows = None):
    cc, _ = one_hot_encoder(cc, 'cc', sk_id)
    # General aggregations
    cc=cc.drop(['SK_ID_PREV'], axis= 1)
    cc_agg = cc.groupby('SK_ID_CURR').agg(['min', 'max', 'mean', 'sum', 'var'])
    cc_agg.columns = pd.Index(['CC_' + e[0] + "_" + e[1].upper() for e in cc_agg.columns.tolist()])

    cc_agg=cc_agg.apply(pd.to_numeric)
    # Count credit card lines
    cc_agg['CC_COUNT'] = cc.groupby('SK_ID_CURR').size()
    del cc
    cc_agg = clean_df(cc_agg)
    return cc_agg


def preprocess(tables_dic: dict, sk_id) -> pd.DataFrame:
    """
    Take a dictionary of tables {'table': dataframe or dict}.
    Returns a dataframe.
    """
    df = application_train_test(tables_dic['application_test'], sk_id)
    df = df.merge(bureau_and_balance(tables_dic['bureau'], tables_dic['bureau_balance'], sk_id), on='SK_ID_CURR', how='left')
    df = df.merge(previous_applications(tables_dic['previous_application'], sk_id), on='SK_ID_CURR', how='left')
    df = df.merge(pos_cash(tables_dic['POS_CASH_balance'], sk_id), on='SK_ID_CURR', how='left')
    df = df.merge(installments_payments(tables_dic['installments_payments'], sk_id), on='SK_ID_CURR', how='left')
    df = df.merge(credit_card_balance(tables_dic['credit_card_balance'], sk_id), on='SK_ID_CURR', how='left')
    df = clean_df(df)
    return df

def apply_custom_values(tables_dic: dict, overrides: dict) -> dict:
    """
    Take a dictionary tables_dic of dataframes, or dictionaries, with the
    original data, and a dictionary overrides of values to edit {'column': value}.
    Return a dictionary in the same format as tables_dic.
    """
    if overrides is None:
        return tables_dic

    for table_name, table in tables_dic.items():
        for col, v in overrides.items():
            if col in table.columns:
                table[col]=v
        tables_dic[table_name]=table
    return tables_dic