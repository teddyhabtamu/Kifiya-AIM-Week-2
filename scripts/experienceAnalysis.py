import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sqlalchemy import create_engine
from sklearn.impute import SimpleImputer

def aggregate_user_experience(df):
    df['Average TCP Retransmission'] = (df['TCP DL Retrans. Vol (Bytes)'] + df['TCP UL Retrans. Vol (Bytes)']) / 2
    df['Average RTT'] = (df['Avg RTT DL (ms)'] + df['Avg RTT UL (ms)']) / 2
    df['Average Throughput'] = (df['Avg Bearer TP DL (kbps)'] + df['Avg Bearer TP UL (kbps)']) / 2
    
    aggregation = {
        'Average TCP Retransmission': 'mean',
        'Average RTT': 'mean',
        'Handset Type': 'first',
        'Average Throughput': 'mean'
    }
    df_aggregated = df.groupby('MSISDN/Number').agg(aggregation).reset_index()
    return df_aggregated

def compute_top_bottom_frequent(df, column, top_n=10):
    top_values = df[column].nlargest(top_n)
    bottom_values = df[column].nsmallest(top_n)
    most_frequent_values = df[column].value_counts().head(top_n)
    
    return top_values, bottom_values, most_frequent_values

def plot_distribution(df, column, group_by, title):
    plt.figure(figsize=(12, 8))
    sns.boxplot(x=group_by, y=column, data=df)
    plt.title(title)
    plt.xticks(rotation=90)
    plt.show()