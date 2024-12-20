import pandas as pd
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt


def aggregate_engagement_metrics(df):
    aggregation = {
        'Bearer Id': 'count',
        'Dur. (ms)': 'sum',
        'Total DL (Bytes)': 'sum',
        'Total UL (Bytes)': 'sum'
    }
    df_aggregated = df.groupby('MSISDN/Number').agg(aggregation).reset_index()
    df_aggregated.rename(columns={'Bearer Id': 'session_frequency', 'Dur. (ms)': 'session_duration', 'Total DL (Bytes)': 'total_download', 'Total UL (Bytes)': 'total_upload'}, inplace=True)
    df_aggregated['total_traffic'] = df_aggregated['total_download'] + df_aggregated['total_upload']
    return df_aggregated

def calculate_engagement_score(df):
    df['engagement_score'] = df['session_frequency'] + df['session_duration'] + df['total_traffic']
    return df

def aggregate_traffic_per_application(df, application_dl_col, application_ul_col):
    df['total_traffic'] = df[application_dl_col] + df[application_ul_col]
    df_aggregated = df.groupby('MSISDN/Number')['total_traffic'].sum().reset_index()
    return df_aggregated

def plot_top_users_per_application(df, application_name, top_n=10):
    top_users = df.nlargest(top_n, 'total_traffic')
    
    plt.figure(figsize=(12, 8))
    plt.barh(top_users['MSISDN/Number'].astype(str), top_users['total_traffic'], color='skyblue')
    plt.xlabel('Total Traffic (Bytes)')
    plt.ylabel('User (MSISDN/Number)')
    plt.title(f'Top {top_n} Users by {application_name} Traffic')
    plt.gca().invert_yaxis()  # Invert y-axis to have the highest traffic on top
    plt.show()

def plot_top_users_by_engagement(df, top_n=10):
    top_users = df.nlargest(top_n, 'engagement_score')
    
    plt.figure(figsize=(12, 8))
    plt.barh(top_users['MSISDN/Number'].astype(str), top_users['engagement_score'], color='skyblue')
    plt.xlabel('Engagement Score')
    plt.ylabel('User (MSISDN/Number)')
    plt.title(f'Top {top_n} Users by Engagement')
    plt.gca().invert_yaxis()  # Invert y-axis to have the highest score on top
    plt.show()

def aggregate_traffic_over_time(df, time_col, app_columns):
    # Ensure time_col exists
    if time_col not in df.columns:
        raise KeyError(f"Column '{time_col}' not found in the dataframe.")
    
    # Ensure app_columns exist
    missing_columns = [col for col in app_columns if col not in df.columns]
    if missing_columns:
        raise KeyError(f"Columns {missing_columns} not found in the dataframe.")
    
    df[time_col] = pd.to_datetime(df[time_col])  # Convert to datetime
    df.set_index(time_col, inplace=True)  # Set time_col as index
    df_aggregated = df[app_columns].resample('D').sum().reset_index()  # Aggregate by day
    return df_aggregated

def plot_usage_trends(df, time_col, app_columns, top_n=3):
    total_traffic = df[app_columns].sum().sort_values(ascending=False)  # Sum traffic per app
    top_apps = total_traffic.head(top_n).index  # Get top N apps
    
    plt.figure(figsize=(12, 8))
    for app in top_apps:
        plt.plot(df[time_col], df[app], label=app)  # Plot trends
    
    plt.xlabel('Date')
    plt.ylabel('Total Traffic (Bytes)')
    plt.title(f'Usage Trends for Top {top_n} Applications')
    plt.legend()
    plt.show()
    
def normalize_metrics(df):
    scaler = StandardScaler()
    df_normalized = pd.DataFrame(scaler.fit_transform(df[['session_frequency', 'session_duration', 'total_traffic']]), columns=['session_frequency', 'session_duration', 'total_traffic'])
    return df_normalized

def run_kmeans(df, n_clusters=3):
    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    df['cluster'] = kmeans.fit_predict(df)
    return df, kmeans

def compute_cluster_metrics(df):
    cluster_metrics = df.groupby('cluster').agg({
        'session_frequency': ['min', 'max', 'mean', 'sum'],
        'session_duration': ['min', 'max', 'mean', 'sum'],
        'total_traffic': ['min', 'max', 'mean', 'sum']
    })
    return cluster_metrics

def plot_elbow_method(df):
    distortions = []
    K = range(1, 11)
    for k in K:
        kmeans = KMeans(n_clusters=k, random_state=42)
        kmeans.fit(df)
        distortions.append(kmeans.inertia_)
    
    plt.figure(figsize=(8, 6))
    plt.plot(K, distortions, 'bx-')
    plt.xlabel('k')
    plt.ylabel('Distortion')
    plt.title('Elbow Method for Optimal k')
    plt.show()
    
def compute_correlation_matrix(df, columns):
    correlation_matrix = df[columns].corr()
    return correlation_matrix

def plot_correlation_heatmap(correlation_matrix):
    plt.figure(figsize=(10, 8))
    sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt='.2f')
    plt.title('Correlation Matrix Heatmap')
    plt.show()