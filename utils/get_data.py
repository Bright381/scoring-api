import psycopg
import pandas as pd
import os
from typing import Any

from utils.single_row_preprocessing import (
    preprocess,
    apply_custom_values
)

DB_URL = os.environ['DB_URL']

with psycopg.connect(DB_URL) as conn:
    with conn.cursor() as cur:
        cur.execute(f"""SELECT * FROM table_names;""")
        TABLES=cur.fetchall()


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
            cols = [desc[0] for desc in cur.description]
            df=pd.DataFrame(rows, columns=cols)

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