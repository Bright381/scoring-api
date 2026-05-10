import psycopg
import pandas as pd
import numpy as np
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
            if table!="bureau_balance":
                query=f"""
                SELECT
                    *
                FROM 
                    {table}
                WHERE 
                    "SK_ID_CURR"=%s;"""
            else:
                query=f"""
                SELECT
                    *
                FROM
                    {table}
                WHERE
                    "SK_ID_BUREAU" IN (
                        SELECT 
                            DISTINCT("SK_ID_BUREAU")
                        FROM
                            bureau
                        WHERE
                            "SK_ID_CURR"=%s
                    );
                """
            cur.execute(query, (sk_id,))
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
    tables_dic = get_raw_tables_dic(sk_id)
    tables_dic = apply_custom_values(tables_dic, overrides)

    for table in tables_dic.values():    
        tables_dic[table] = preprocess(table)
    return tables_dic

def get_preprocessed_features(sk_id):
    return fetch_table_rows(sk_id, 'preprocessed_data')


### shoud treat table without sk_id apart
def get_column_stats(table: str, column: str, sk_id: int) -> dict:
    """
    Fetch population histogram data for a column and the customer's own value.
 
    Samples up to 10 000 rows from the given table for the histogram, then
    retrieves the customer's specific value separately. Returns a dict with:
        - bin_edges, counts  — histogram arrays (30 bins)
        - customer_value     — the customer's raw value (None if missing)
        - percentile         — customer position in the population (0-100)
        - mean, median, std  — summary statistics
        - n                  — sample size used for the histogram
    """
    with psycopg.connect(DB_URL) as conn:
        with conn.cursor() as cur:
            # Population sample (cap at 10k to keep it fast)
            cur.execute(
                f'SELECT "{column}" FROM {table} WHERE "{column}" IS NOT NULL LIMIT 10000'
            )
            pop_rows = cur.fetchall()
 
            # Customer value (first matching row)
            cur.execute(
                f'SELECT "{column}" FROM {table} WHERE "SK_ID_CURR" = %s LIMIT 1',
                (sk_id,)
            )
            cust_row = cur.fetchone()
 
    population = np.array([r[0] for r in pop_rows], dtype=float)
    population = population[~np.isnan(population)]
 
    customer_val = None
    if cust_row is not None and cust_row[0] is not None:
        try:
            customer_val = float(cust_row[0])
        except (TypeError, ValueError):
            customer_val = None
 
    if population.size == 0:
        return {
            "bin_edges": [],
            "counts": [],
            "customer_value": customer_val,
            "percentile": None,
            "mean": None,
            "median": None,
            "std": None,
            "n": 0,
        }
 
    counts, bin_edges = np.histogram(population, bins=30)
 
    percentile = None
    if customer_val is not None:
        percentile = float(np.mean(population < customer_val) * 100)
 
    return {
        "bin_edges": bin_edges.tolist(),
        "counts": counts.tolist(),
        "customer_value": customer_val,
        "percentile": round(percentile, 1) if percentile is not None else None,
        "mean": round(float(np.mean(population)), 4),
        "median": round(float(np.median(population)), 4),
        "std": round(float(np.std(population)), 4),
        "n": int(population.size),
    }
 
