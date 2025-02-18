import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.cluster import KMeans
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, jaccard_score, silhouette_score
from sklearn.decomposition import PCA

# Load netML dataset
data = pd.read_csv("../Data_Resources/netML/netML_dataset.csv")

# Define label (assumed 'Label' column contains ground truth)
data['label'] = data['Label'].apply(lambda x: 1 if x == "Attack" else 0)
features = data.drop(columns=["label"])

# Feature-specific embedding and preprocessing

# 1. Categorical Data Embedding
categorical_features = ['Protocol']  # Assuming 'Protocol' is categorical
encoder = OneHotEncoder(sparse=False, handle_unknown='ignore')
categorical_data = encoder.fit_transform(data[categorical_features])

# 2. Timing Features
time_features = [
    'Flow IAT Max', 'Flow IAT Min', 'Flow IAT Mean', 'Flow IAT Std',
    'Fwd IAT Total', 'Fwd IAT Mean', 'Fwd IAT Max', 'Fwd IAT Min',
    'Bwd IAT Total', 'Bwd IAT Mean', 'Bwd IAT Max', 'Bwd IAT Min'
]
scaler_time = StandardScaler()
time_data = scaler_time.fit_transform(data[time_features])

# 3. Packet Length Features
packet_length_features = [
    'Total Length of Fwd Packets', 'Total Length of Bwd Packets',
    'Fwd Packet Length Max', 'Fwd Packet Length Min', 'Fwd Packet Length Mean', 'Fwd Packet Length Std',
    'Bwd Packet Length Max', 'Bwd Packet Length Min', 'Bwd Packet Length Mean', 'Bwd Packet Length Std'
]
scaler_packet = StandardScaler()
packet_length_data = scaler_packet.fit_transform(data[packet_length_features])

# 4. Count Features
count_features = [
    'Total Fwd Packets', 'Total Backward Packets', 'Flow Packets/s',
    'Fwd Packets/s', 'Bwd Packets/s', 'Subflow Fwd Packets', 'Subflow Bwd Packets'
]
scaler_count = StandardScaler()
count_data = scaler_count.fit_transform(data[count_features])

# 5. Binary Features
binary_features = [
    'FIN Flag Count', 'SYN Flag Count', 'RST Flag Count', 'PSH Flag Count',
    'ACK Flag Count', 'URG Flag Count', 'CWE Flag Count', 'ECE Flag Count'
]
binary_data = data[binary_features].values

# Combine processed features
X = np.hstack([categorical_data, time_data, packet_length_data, count_data, binary_data])

# Dimensionality reduction
pca = PCA(n_components=10, random_state=42)
X_reduced = pca.fit_transform(X)

# Apply K-Means clustering
n_clusters = 2  # Assuming binary classification
kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
data['cluster'] = kmeans.fit_predict(X_reduced)

# Adjust cluster labels if necessary
cluster_mapping = {0: 1, 1: 0}  # Adjust mapping if needed
data['adjusted_cluster'] = data['cluster'].map(cluster_mapping)

# Evaluate clustering performance
def evaluate_metrics(y_true, y_pred):
    metrics = {}
    for avg in ["macro", "micro", "weighted"]:
        metrics[f"accuracy_{avg}"] = accuracy_score(y_true, y_pred)
        metrics[f"precision_{avg}"] = precision_score(y_true, y_pred, average=avg, zero_division=0)
        metrics[f"recall_{avg}"] = recall_score(y_true, y_pred, average=avg, zero_division=0)
        metrics[f"f1_{avg}"] = f1_score(y_true, y_pred, average=avg, zero_division=0)
        metrics[f"jaccard_{avg}"] = jaccard_score(y_true, y_pred, average=avg, zero_division=0)
    metrics["silhouette"] = silhouette_score(features, y_pred)
    return metrics
# Save results
data[['cluster', 'adjusted_cluster', 'label']].to_csv("./netML_KMeans_clustering_Compare.csv", index=False)

# Compute metrics for both cluster and adjusted_cluster
metrics_cluster = evaluate_metrics(data["label"], data["cluster"].values)
metrics_adjusted = evaluate_metrics(data["label"], data["adjusted_cluster"].values)

# Save results to CSV
metrics_df = pd.DataFrame([metrics_cluster, metrics_adjusted], index=["Cluster", "Adjusted_Cluster"])
metrics_df.to_csv("netML_KMeans_clustering_Compare_Metrics.csv")
