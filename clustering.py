import numpy as np
import pandas as pd
from sklearn.cluster import KMeans

# Load dataset
df = pd.read_csv('total_df.csv')
print(df.head())

# Index dataframe with LNR id
X = df.set_index('LNR').astype(np.int8)

# KMeans Clustering
kmeans = KMeans(n_clusters=6, random_state=42, max_iter=700)

print("Clustering ... \n")
kmeans.fit(X)
clusters = kmeans.predict(X)
X['cluster'] = clusters

# Store DataFrame
print("Storing clustered data...")
X.to_csv('clustered_df.csv', header=X.columns)
