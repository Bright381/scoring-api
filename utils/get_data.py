import psycopg
import pandas as pd
import os

from single_row_preprocessing import (
    preprocess,
    apply_custom_values
)

DB_URL = os.environ['DB_URL']

data={
    'preprocessed_data': 'scoring-api/data/preprocessed_test.csv',
    # 'application_train': 'application_train.csv',
    # 'application_test': 'application_test.csv',
    # 'bureau': 'bureau.csv',
    # 'bureau_balance': 'bureau_balance.csv',
    # 'previous_application': 'previous_application.csv',
    # 'POS_CASH_balance': 'POS_CASH_balance.csv',
    # 'installments_payments': 'installments_payments.csv',
    # 'credit_card_balance': 'credit_card_balance.csv'
}

def fetch_data(sk_id, table):
    with psycopg.connect(DB_URL) as conn:
        with conn.cursor() as cur:
            cur.execute(f"""SELECT
                                *
                            FROM 
                                {table} 
                            WHERE 
                                "SK_ID_CURR"={sk_id};""")
            rows=cur.fetchall()
            cols = [desc[0] for desc in cur.description]
            df=pd.DataFrame(rows, columns=cols)

    return df

def get_raw_features(sk_id):
    for table in data.keys():
        data[table]=fetch_data(sk_id=sk_id, table=table)
    return data

def get_custom_features(sk_id, overrides = None):
    tables_dic = get_raw_features(sk_id)
    for table in tables_dic.values():
        tables_dic[table] = apply_custom_values(table, overrides)
        tables_dic[table] = preprocess(table)
    return tables_dic

def get_preprocessed_features(sk_id):
    return fetch_data(sk_id, 'preprocessed_data')