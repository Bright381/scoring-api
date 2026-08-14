import psycopg
import pandas as pd
import numpy as np
import os
from typing import Any
from utils.single_row_preprocessing import (
    preprocess,
    apply_custom_values
)
# from dotenv import load_dotenv
# load_dotenv()
DB_URL = os.environ['DB_URL']

def get_table_names() -> list:
    with psycopg.connect(DB_URL) as conn:
        with conn.cursor() as cur:
            cur.execute(f"""
                SELECT table_name 
                FROM information_schema.tables 
                WHERE table_schema = 'public';"""
            )
            tables=cur.fetchall()
    return [t[0] for t in tables]

def get_table_columns(table: str) -> list:
    with psycopg.connect(DB_URL) as conn:
        with conn.cursor() as cur:
            cur.execute(f"""
                SELECT column_name 
                FROM information_schema.columns
                WHERE table_name = '{table}'
                ORDER BY ordinal_position
                ;"""
            )
            cols=cur.fetchall()
    return [c[0] for c in cols]

TABLES = get_table_names()

def fetch_target(sk_id: int):
    """Return the TARGET value for a specific SK_ID_CURR as a JSON-serializable dict.

    Returns {"TARGET": 0|1|None}.
    """
    query = 'SELECT "TARGET" FROM application_test WHERE "SK_ID_CURR" = %s'
    with psycopg.connect(DB_URL) as conn:
        with conn.cursor() as cur:
            cur.execute(query, (sk_id,))
            row = cur.fetchone()
    return {"TARGET": (int(row[0]) if row is not None and row[0] is not None else None)}


def fetch_population_targets(table: str, column: str, filter_col: str = None, filter_val: Any = None) -> list:
    """Return a list of TARGET values for the population sample used to build distributions.

    The list preserves ordering and contains 0, 1, or None for SQL NULLs. Caps at 10k rows to match distribution sampling.
    """
    if table not in TABLES:
        raise ValueError(f"Unknown table '{table}'.")

    with psycopg.connect(DB_URL) as conn:
        with conn.cursor() as cur:
            base_where = f'"{column}" IS NOT NULL'
            params = []
            if filter_col and filter_val is not None:
                base_where += f' AND "{filter_col}" = %s'
                params.append(str(filter_val))

            query = f'SELECT "TARGET" FROM "{table}" WHERE {base_where}'
            cur.execute(query, tuple(params))
            rows = cur.fetchall()

    return [(int(r[0]) if r[0] is not None else None) for r in rows]

def fetch_unique_values(table: str, column: str):
    with psycopg.connect(DB_URL) as conn:
        with conn.cursor() as cur:
            cur.execute(f"""
                SELECT DISTINCT "{column}"
                FROM {table}
                ;"""
            )
            values=cur.fetchall()
    return {'values': [v[0] for v in values]}

def try_numeric(col):
    if col.dtype != object:
        return col  # already numeric, skip
    if col.isna().all():
        return col  # all-NaN object column → keep as object (likely categorical)
    converted = pd.to_numeric(col, errors='coerce')
    if converted.isna().sum() > col.isna().sum():
        return col  # conversion introduced new NaNs → it was categorical
    return converted
    
def fetch_table_rows(sk_id: int, table: str) -> pd.DataFrame:
    try:
        with psycopg.connect(DB_URL) as conn:
            with conn.cursor() as cur:
                if table!="bureau_balance":
                    query=f"""
                    SELECT
                        *
                    FROM 
                        "{table}"
                    WHERE 
                        "SK_ID_CURR"=%s
                    ;"""
                else:
                    query=f"""
                    SELECT
                        *
                    FROM
                        "{table}"
                    WHERE
                        "SK_ID_BUREAU" IN (
                            SELECT 
                                DISTINCT("SK_ID_BUREAU")
                            FROM
                                bureau
                            WHERE
                                "SK_ID_CURR"=%s
                        )
                    ;
                    """
                cur.execute(query, (sk_id,))
                rows=cur.fetchall()
                rows=list(rows)
                cols=get_table_columns(table)

                if not rows:
                    return pd.DataFrame(columns=[c for c in cols if c != 'TARGET'])
                
                df = pd.DataFrame(rows, columns=cols)                
                df = df.apply(try_numeric)
    
    except Exception as e:
        raise ValueError(f"Failed fetching table: {table}, due to error: {e}")
    
    return df

def get_raw_tables_dic(sk_id: int) -> dict[str, Any]:
    """
    Return a dictionary with keys being table names and values being dataframes
    """
    tables_dic={}
    for table in TABLES:
        tables_dic[table]=fetch_table_rows(sk_id=sk_id, table=table)
    return tables_dic

def get_custom_features(sk_id: int, overrides: dict = None) -> pd.DataFrame:
    """
    Apply get_raw_tables_dic, apply_custom_values then preprocess.
    """
    tables_dic = get_raw_tables_dic(sk_id)
    tables_dic = apply_custom_values(tables_dic, overrides)

    return preprocess(tables_dic)

def get_preprocessed_features(sk_id) -> pd.DataFrame:
    return fetch_table_rows(sk_id, 'preprocessed_data')


def get_column_stats(table: str, column: str, sk_id: int, filter_col: str = None, filter_val: Any = None) -> dict:
    """
    Fetch population histogram data for a column and the customer's own value.
    Now supports optional segment filtering and correctly groups by TARGET.
    """
    with psycopg.connect(DB_URL) as conn:
        with conn.cursor() as cur:
            # 1. Customer value (first matching row)
            cur.execute(
                f'SELECT "{column}" FROM "{table}" WHERE "SK_ID_CURR" = %s',
                (sk_id,)
            )
            cust_row = cur.fetchone()
 
            base_where = f'"{column}" IS NOT NULL'
            params = []
            
            if filter_col and filter_val is not None:
                base_where += f' AND "{filter_col}" = %s'
                params.append(str(filter_val))
                
            cur.execute(
                f'SELECT "{column}" FROM "{table}" WHERE {base_where}',
                tuple(params)
            )
            pop_rows = cur.fetchall()
 
    # Clean population and parse floats. Keep track of TARGETs.
    valid_rows = [r for r in pop_rows if r[0] is not None]
    population = np.array([r[0] for r in valid_rows], dtype=float)
 
    customer_val = None
    if cust_row is not None and cust_row[0] is not None:
        try:
            customer_val = float(cust_row[0])
        except (TypeError, ValueError):
            pass
 
    if population.size == 0:
        return {
            "bin_edges": [], "count_target_0": [], "count_target_1": [], "count_target_na": [],
            "customer_value": customer_val,
            "percentile": None, "mean": None, "median": None, "std": None, "n": 0,
        }
 
    # Calculate overall bin edges
    _, bin_edges = np.histogram(population, bins=30)
    
    # Separate populations by target class
    pop_target_0 = np.array([r[0] for r in valid_rows if r[1] == 0], dtype=float)
    pop_target_1 = np.array([r[0] for r in valid_rows if r[1] == 1], dtype=float)
    pop_target_na = np.array([r[0] for r in valid_rows if r[1] is None], dtype=float)

    # Compute per-target histograms
    count_target_0, _ = np.histogram(pop_target_0, bins=bin_edges)
    count_target_1, _ = np.histogram(pop_target_1, bins=bin_edges)
    count_target_na, _ = np.histogram(pop_target_na, bins=bin_edges)
 
    percentile = None
    if customer_val is not None:
        percentile = float(np.mean(population < customer_val) * 100)
 
    return {
        "bin_edges": bin_edges.tolist(),
        "count_target_0": count_target_0.tolist(),
        "count_target_1": count_target_1.tolist(),
        "count_target_na": count_target_na.tolist(),
        "customer_value": customer_val,
        "percentile": round(percentile, 1) if percentile is not None else None,
        "mean": round(float(np.mean(population)), 4),
        "median": round(float(np.median(population)), 4),
        "std": round(float(np.std(population)), 4),
        "n": int(population.size),
    }

def get_bivariate_data(table: str, col_x: str, col_y: str, sk_id: int, filter_col: str = None, filter_val: Any = None) -> dict:
    """
    Fetch background population coordinates for two numeric columns and the customer's own coordinates.
    Supports optional filtering on a third database column.
    """
    with psycopg.connect(DB_URL) as conn:
        with conn.cursor() as cur:
            # Fetch Targeted Customer Coordinates
            if table != "bureau_balance":
                cust_query = f'SELECT "{col_x}", "{col_y}" FROM "{table}" WHERE "SK_ID_CURR" = %s'
            else:
                cust_query = f"""
                    SELECT "{col_x}", "{col_y}" FROM "{table}" 
                    WHERE "SK_ID_BUREAU" IN (
                        SELECT DISTINCT("SK_ID_BUREAU") FROM bureau WHERE "SK_ID_CURR" = %s
                    )
                """
            cur.execute(cust_query, (sk_id,))
            cust_row = cur.fetchone()

            # Build Population Sample Query with Optional Database Filtering
            base_where = f'"{col_x}" IS NOT NULL AND "{col_y}" IS NOT NULL'
            params = []
            
            if filter_col and filter_val is not None:
                base_where += f' AND "{filter_col}" = %s'
                params.append(str(filter_val))
            
            pop_query = f'SELECT "{col_x}", "{col_y}", "TARGET" FROM "{table}" WHERE {base_where}'
            cur.execute(pop_query, tuple(params))
            pop_rows = cur.fetchall()

    # Parse targeted customer coordinates safely
    customer_x = None
    customer_y = None
    if cust_row is not None:
        try:
            customer_x = float(cust_row[0]) if cust_row[0] is not None else None
            customer_y = float(cust_row[1]) if cust_row[1] is not None else None
        except (TypeError, ValueError):
            pass

    # Parse and decouple background population coordinates and their TARGETs
    pop_x = []
    pop_y = []
    pop_target = []
    for r in pop_rows:
        # r expected: (col_x_value, col_y_value, target_value)
        if r[0] is not None and r[1] is not None:
            try:
                pop_x.append(float(r[0]))
                pop_y.append(float(r[1]))
                # preserve None for SQL NULLs; if not None coerce to int when possible
                pop_target.append(int(r[2]) if r[2] is not None else None)
            except (TypeError, ValueError):
                continue

    return {
        "col_x": col_x,
        "col_y": col_y,
        "customer_x": customer_x,
        "customer_y": customer_y,
        "pop_x": pop_x,
        "pop_y": pop_y,
        "TARGET": pop_target,
        "n": len(pop_x)
    }
