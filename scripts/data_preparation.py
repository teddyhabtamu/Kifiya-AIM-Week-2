import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sqlalchemy import create_engine
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

import os

def load_data(db_url, query):
    engine = create_engine(db_url)
    try:
        with engine.connect() as connection:
            df = pd.read_sql(query, connection)
        return df
    except Exception as e:
        print(f"Error occurred: {e}")
        return None

def clean_data(df):
    numeric_cols = df.select_dtypes(include=['number']).columns
    non_numeric_cols = df.select_dtypes(exclude=['number']).columns

    df_numeric = df[numeric_cols].copy()
    
    for col in numeric_cols:

        Q1 = df_numeric[col].quantile(0.25)
        Q3 = df_numeric[col].quantile(0.75)
        IQR = Q3 - Q1
        
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        
        df_numeric[col] = df_numeric[col].apply(
            lambda x: lower_bound if x < lower_bound else upper_bound if x > upper_bound else x
        )

    imputer = SimpleImputer(strategy='mean')
    df_numeric = pd.DataFrame(imputer.fit_transform(df_numeric), columns=numeric_cols)

    df_non_numeric = df[non_numeric_cols].fillna('missing')

    df_cleaned = pd.concat([df_numeric, df_non_numeric], axis=1)

    return df_cleaned

def aggregate_user_behavior(df):
    aggregation = {
        'Bearer Id': 'count',
        'Dur. (ms)': 'sum',
        'Total DL (Bytes)': 'sum',
        'Total UL (Bytes)': 'sum',
        'Social Media DL (Bytes)': 'sum',
        'Social Media UL (Bytes)': 'sum',
        'Google DL (Bytes)': 'sum',
        'Google UL (Bytes)': 'sum',
        'Email DL (Bytes)': 'sum',
        'Email UL (Bytes)': 'sum',
        'Youtube DL (Bytes)': 'sum',
        'Youtube UL (Bytes)': 'sum',
        'Netflix DL (Bytes)': 'sum',
        'Netflix UL (Bytes)': 'sum',
        'Gaming DL (Bytes)': 'sum',
        'Gaming UL (Bytes)': 'sum',
        'Other DL (Bytes)': 'sum',
        'Other UL (Bytes)': 'sum'
    }
    df_aggregated = df.groupby('MSISDN/Number').agg(aggregation).reset_index()
    df_aggregated.rename(columns={'Bearer Id': 'xDR_sessions', 'Dur. (ms)': 'session_duration'}, inplace=True)
    return df_aggregated

def sanitize_filename(filename):
    return "".join([c if c.isalnum() else "_" for c in filename])

def graphical_univariate_analysis(df, output_dir='plots', sample_size=1000):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # Sample the data to reduce processing time
    if len(df) > sample_size:
        df_sample = df.sample(n=sample_size, random_state=42)
    else:
        df_sample = df
    
    for column in df_sample.select_dtypes(include=['float64', 'int64']).columns:
        plt.figure()
        sns.histplot(df_sample[column])
        plt.title(f'Distribution of {column}')
        sanitized_column = sanitize_filename(column)
        plt.savefig(os.path.join(output_dir, f'{sanitized_column}_distribution.png'))
        plt.close()

def bivariate_analysis(df, output_dir='bivariate_plots', sample_size=1000):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # Sample the data to reduce processing time
    if len(df) > sample_size:
        df_sample = df.sample(n=sample_size, random_state=42)
    else:
        df_sample = df
    
    numeric_columns = df_sample.select_dtypes(include=['float64', 'int64']).columns
    for i, col1 in enumerate(numeric_columns):
        for col2 in numeric_columns[i+1:]:
            plt.figure()
            sns.scatterplot(x=df_sample[col1], y=df_sample[col2])
            plt.title(f'{col1} vs {col2}')
            sanitized_col1 = sanitize_filename(col1)
            sanitized_col2 = sanitize_filename(col2)
            plt.savefig(os.path.join(output_dir, f'{sanitized_col1}_vs_{sanitized_col2}.png'))
            plt.close()

def correlation_analysis(df, output_file='correlation_matrix.png'):
    # Select only numeric columns
    numeric_df = df.select_dtypes(include=['float64', 'int64'])
    
    # Compute the correlation matrix
    correlation_matrix = numeric_df.corr()
    
    # Plot the heatmap
    plt.figure(figsize=(12, 10))
    sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt='.2f')
    plt.title('Correlation Matrix')
    plt.show()
    
    return correlation_matrix

def dimensionality_reduction(df, n_components=2):
    # Select only numeric columns
    numeric_df = df.select_dtypes(include=['float64', 'int64'])
    
    # Impute missing values
    imputer = SimpleImputer(strategy='mean')
    numeric_df_imputed = pd.DataFrame(imputer.fit_transform(numeric_df), columns=numeric_df.columns)
    
    # Standardize the data
    scaler = StandardScaler()
    numeric_df_scaled = scaler.fit_transform(numeric_df_imputed)
    
    # Perform PCA
    pca = PCA(n_components=n_components)
    components = pca.fit_transform(numeric_df_scaled)
    
    # Create a DataFrame with the principal components
    df_pca = pd.DataFrame(data=components, columns=[f'PC{i+1}' for i in range(n_components)])
    return df_pca