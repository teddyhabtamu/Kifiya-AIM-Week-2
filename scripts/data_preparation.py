import pandas as pd
from sqlalchemy import create_engine
from sklearn.impute import SimpleImputer

def load_data(db_url, query):
    engine = create_engine(db_url)
    with engine.connect() as connection:
        df = pd.read_sql(query, connection)
    return df

def clean_data_with_iqr(df):
    # Separate numeric and non-numeric columns
    numeric_cols = df.select_dtypes(include=['number']).columns
    non_numeric_cols = df.select_dtypes(exclude=['number']).columns

    # Handle numeric columns
    df_numeric = df[numeric_cols].copy()
    
    for col in numeric_cols:
        # Calculate Q1, Q3, and IQR
        Q1 = df_numeric[col].quantile(0.25)
        Q3 = df_numeric[col].quantile(0.75)
        IQR = Q3 - Q1
        
        # Define bounds
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        
        # Cap outliers to bounds
        df_numeric[col] = df_numeric[col].apply(
            lambda x: lower_bound if x < lower_bound else upper_bound if x > upper_bound else x
        )

    # Handle missing values in numeric data
    imputer = SimpleImputer(strategy='mean')
    df_numeric = pd.DataFrame(imputer.fit_transform(df_numeric), columns=numeric_cols)

    # Handle non-numeric columns
    df_non_numeric = df[non_numeric_cols].fillna('missing')

    # Combine numeric and non-numeric columns back
    df_cleaned = pd.concat([df_numeric, df_non_numeric], axis=1)

    return df_cleaned

