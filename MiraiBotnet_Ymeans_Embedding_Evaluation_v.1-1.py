import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, jaccard_score, silhouette_score
from sklearn.cluster import KMeans

# Load sample data (replace with your actual file path)
data = pd.read_csv("./output-dataset_ESSlab.csv")

# Define label for attack vs benign
data['label'] = data[['reconnaissance', 'infection', 'action']].any(axis=1).astype(int)

# Feature-specific embedding and preprocessing
# 1. Categorical Data Embedding
categorical_features = ['flow_protocol']
encoder = OneHotEncoder(sparse=False, handle_unknown='ignore')
categorical_data = encoder.fit_transform(data[categorical_features])

# 2. Timing Features
time_features = [
    'flow_iat_max', 'flow_iat_min', 'flow_iat_mean', 'flow_iat_total', 'flow_iat_std',
    'forward_iat_max', 'forward_iat_min', 'forward_iat_mean', 'forward_iat_total', 'forward_iat_std',
    'backward_iat_max', 'backward_iat_min', 'backward_iat_mean', 'backward_iat_total', 'backward_iat_std'
]
scaler_time = StandardScaler()
time_data = scaler_time.fit_transform(data[time_features])

# 3. Packet Length Features
packet_length_features = [
    'forward_packet_length_mean', 'forward_packet_length_min', 'forward_packet_length_max', 'forward_packet_length_std',
    'backward_packet_length_mean', 'backward_packet_length_min', 'backward_packet_length_max', 'backward_packet_length_std'
]
scaler_packet = StandardScaler()
packet_length_data = scaler_packet.fit_transform(data[packet_length_features])

# 4. Count Features
count_features = [
    'fpkts_per_second', 'bpkts_per_second', 'total_forward_packets', 'total_backward_packets',
    'total_length_of_forward_packets', 'total_length_of_backward_packets', 'flow_packets_per_second',
    'flow_psh', 'flow_syn', 'flow_urg', 'flow_fin', 'flow_ece', 'flow_ack', 'flow_rst', 'flow_cwr'
]
scaler_count = StandardScaler()
count_data = scaler_count.fit_transform(data[count_features])

# Combine processed features
X = np.hstack([categorical_data, time_data, packet_length_data, count_data])

# Y-Means Clustering Function
def y_means_clustering(X, max_clusters=10):
    best_score = -1
    best_model = None
    best_k = 2
    for k in range(2, max_clusters + 1):
        model = KMeans(n_clusters=k, random_state=42, n_init=50)
        labels = model.fit_predict(X)
        silhouette_avg = silhouette_score(X, labels)
        if silhouette_avg > best_score:
            best_score = silhouette_avg
            best_model = model
            best_k = k
    return best_model, best_k

# Perform Y-Means Clustering
model, optimal_k = y_means_clustering(X, max_clusters=10)
data['cluster'] = model.labels_

# Adjust cluster labels to match ground truth if necessary
cluster_mapping = {0: 1, 1: 0}  # Example mapping, adjust as needed
data['adjusted_cluster'] = data['cluster'].map(cluster_mapping)

# Replace NaN with a default cluster or remove them
data['adjusted_cluster'] = data['adjusted_cluster'].fillna(-1).astype(int)

# Metrics Calculation Function
def calculate_metrics(y_true, y_pred, X=None):
    if X is not None:
        silhouette = silhouette_score(X, y_pred)
    else:
        silhouette = np.nan
    
    metrics = {}
    for avg in ['macro', 'micro', 'weighted']:
        metrics[f"precision_{avg}"] = precision_score(y_true, y_pred, average=avg, zero_division=0)
        metrics[f"recall_{avg}"] = recall_score(y_true, y_pred, average=avg, zero_division=0)
        metrics[f"f1_score_{avg}"] = f1_score(y_true, y_pred, average=avg, zero_division=0)
        metrics[f"jaccard_{avg}"] = jaccard_score(y_true, y_pred, average=avg, zero_division=0)
    metrics["accuracy"] = accuracy_score(y_true, y_pred)
    metrics["silhouette"] = silhouette
    return metrics

# Filter out noise points (-1) for evaluation
data_filtered = data[data['cluster'] != -1]
data_filtered_adjusted = data[data['adjusted_cluster'] != -1]

# Evaluate Metrics
metrics_original = calculate_metrics(data_filtered['label'], data_filtered['cluster'], X[data['cluster'] != -1])
metrics_adjusted = calculate_metrics(data_filtered_adjusted['label'], data_filtered_adjusted['adjusted_cluster'], X[data['adjusted_cluster'] != -1])

# Save metrics to CSV
metrics_list = []
for key, val in metrics_original.items():
    metrics_list.append({"Metric": key, "Original": val, "Adjusted": metrics_adjusted[key]})
metrics = pd.DataFrame(metrics_list)
metrics.to_csv("./MiraiBotnet_Ymeans_clustering_Compare_Metrics_v.1-1.csv", index=False)

# Save results to CSV
data[['cluster', 'adjusted_cluster', 'label']].to_csv("./MiraiBotnet_Ymeans_clustering_Compare_v.1-1.csv", index=False)

# Print evaluation metrics
print("Optimal Number of Clusters (k):", optimal_k)
print("\nOriginal Clustering Metrics:")
for key, val in metrics_original.items():
    print(f"{key.capitalize()}: {val:.2f}")

print("\nAdjusted Clustering Metrics:")
for key, val in metrics_adjusted.items():
    print(f"{key.capitalize()}: {val:.2f}")
