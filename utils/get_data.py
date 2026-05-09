import psycopg
import pandas as pd
import os
from typing import Any

from utils.single_row_preprocessing import (
    preprocess,
    apply_custom_values
)

DB_URL = os.environ['DB_URL']

def get_table_names():
    with psycopg.connect(DB_URL) as conn:
        with conn.cursor() as cur:
            cur.execute(f"""
                SELECT table_name 
                FROM information_schema.tables 
                WHERE table_schema = 'public';"""
            )
            tables=cur.fetchall()
    return [t[0] for t in tables]

TABLES = get_table_names()

def fetch_table_rows(sk_id, table):
    with psycopg.connect(DB_URL) as conn:
        with conn.cursor() as cur:
            cur.execute(f"""SELECT
                                *
                            FROM 
                                {table} 
                            WHERE 
                                "SK_ID_CURR"={sk_id};""")
            rows=cur.fetchall()
            if not rows:
                return pd.DataFrame()
            cols = [desc[0] for desc in cur.description]
            df=pd.DataFrame(rows, columns=cols)
            if 'TARGET' in df.columns:
                df = df.drop(columns=['TARGET'])
            df = df.apply(pd.to_numeric, errors='coerce')

    return df

def get_raw_tables_dic(sk_id: int) -> dict[str, Any]:
    tables_dic={}
    for table in TABLES:
        tables_dic[table]=fetch_table_rows(sk_id=sk_id, table=table)
    return tables_dic

def get_custom_features(sk_id, overrides = None):
    tables_dic = get_raw_features(sk_id)
    tables_dic = apply_custom_values(tables_dic, overrides)

    for table in tables_dic.values():    
        tables_dic[table] = preprocess(table)
    return tables_dic

def get_preprocessed_features(sk_id):
    return fetch_table_rows(sk_id, 'preprocessed_data')