import pandas as pd
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