import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from scipy.spatial.distance import cdist
from sqlalchemy import create_engine

def assign_engagement_score(df, engagement_columns):
  scaler = StandardScaler()
  df_scaled = scaler.fit_transform(df[engagement_columns])
  
  kmeans = KMeans(n_clusters=3, random_state=42)
  df['engagement_cluster'] = kmeans.fit_predict(df_scaled)
  
  # Calculate Euclidean distance to the least engaged cluster center
  cluster_centers = kmeans.cluster_centers_
  least_engaged_center = cluster_centers[np.argmin(np.linalg.norm(cluster_centers, axis=1))]
  df['engagement_score'] = cdist(df_scaled, [least_engaged_center]).flatten()
  
  return df

def assign_experience_score(df, experience_columns):
  scaler = StandardScaler()
  df_scaled = scaler.fit_transform(df[experience_columns])
  
  kmeans = KMeans(n_clusters=3, random_state=42)
  df['experience_cluster'] = kmeans.fit_predict(df_scaled)
  
  # Calculate Euclidean distance to the worst experience cluster center
  cluster_centers = kmeans.cluster_centers_
  worst_experience_center = cluster_centers[np.argmax(np.linalg.norm(cluster_centers, axis=1))]
  df['experience_score'] = cdist(df_scaled, [worst_experience_center]).flatten()
  
  return df
  
def calculate_satisfaction_score(df_engagement, df_experience):
  df = pd.merge(df_engagement[['MSISDN/Number', 'engagement_score']], df_experience[['MSISDN/Number', 'experience_score']], on='MSISDN/Number')
  df['satisfaction_score'] = (df['engagement_score'] + df['experience_score']) / 2
  return df
  
def build_regression_model(df):
  X = df[['engagement_score', 'experience_score']]
  y = df['satisfaction_score']
  
  X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
  
  model = LinearRegression()
  model.fit(X_train, y_train)
  
  y_pred = model.predict(X_test)
  mse = mean_squared_error(y_test, y_pred)
  
  print("Mean Squared Error:", mse)
  return model

def run_kmeans_clustering(df, n_clusters=2):
  scaler = StandardScaler()
  df_scaled = scaler.fit_transform(df[['engagement_score', 'experience_score']])
  
  kmeans = KMeans(n_clusters=n_clusters, random_state=42)
  df['cluster'] = kmeans.fit_predict(df_scaled)
  
  return df

def aggregate_scores_per_cluster(df):
  cluster_aggregation = df.groupby('cluster').agg({
      'engagement_score': 'mean',
      'experience_score': 'mean',
      'satisfaction_score': 'mean'
  }).reset_index()
  return cluster_aggregation

def export_to_mysql(df, db_url, table_name):
  engine = create_engine(db_url)
  df.to_sql(table_name, engine, if_exists='replace', index=False)