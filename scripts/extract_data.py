# scripts/extract_data.py
import pandas as pd
from sqlalchemy import create_engine

def extract_data(db_url, query):
    engine = create_engine(db_url)
    with engine.connect() as connection:
        df = pd.read_sql(query, connection)
    return df
