import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.cluster import DBSCAN
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, jaccard_score, silhouette_score
from sklearn.decomposition import PCA

# Load sample data (replace with your actual file path)
data = pd.read_csv("./output-dataset_ESSlab.csv")

# Define label for attack vs benign
data['label'] = data[['reconnaissance', 'infection', 'action']].any(axis=1).astype(int)

# Feature-specific embedding and preprocessing

# 1. Categorical Data Embedding
categorical_features = ['flow_protocol']
encoder = OneHotEncoder(sparse=False, handle_unknown='ignore')
categorical_data = encoder.fit_transform(data[categorical_features])

# 2. Timing Features (e.g., iat: inter-arrival time)
time_features_group1 = [
    'flow_iat_max', 'flow_iat_min', 'flow_iat_mean', 'flow_iat_total', 'flow_iat_std'
]
time_features_group2 = [
    'forward_iat_max', 'forward_iat_min', 'forward_iat_mean', 'forward_iat_total', 'forward_iat_std'
]
time_features_group3 = [
    'backward_iat_max', 'backward_iat_min', 'backward_iat_mean', 'backward_iat_total', 'backward_iat_std'
]
scaler_time1 = StandardScaler()
scaler_time2 = StandardScaler()
scaler_time3 = StandardScaler()
time_data1 = scaler_time1.fit_transform(data[time_features_group1])
time_data2 = scaler_time2.fit_transform(data[time_features_group2])
time_data3 = scaler_time3.fit_transform(data[time_features_group3])

# 3. Packet Length Features
packet_length_forward = [
    'forward_packet_length_mean', 'forward_packet_length_min', 'forward_packet_length_max', 'forward_packet_length_std'
]
packet_length_backward = [
    'backward_packet_length_mean', 'backward_packet_length_min', 'backward_packet_length_max', 'backward_packet_length_std'
]
scaler_packet_forward = StandardScaler()
scaler_packet_backward = StandardScaler()
packet_length_data_forward = scaler_packet_forward.fit_transform(data[packet_length_forward])
packet_length_data_backward = scaler_packet_backward.fit_transform(data[packet_length_backward])

# 4. Count Features
packet_count_features = [
    'fpkts_per_second', 'bpkts_per_second', 'total_forward_packets', 'total_backward_packets',
    'total_length_of_forward_packets', 'total_length_of_backward_packets', 'flow_packets_per_second'
]
flow_flag_features = [
    'flow_psh', 'flow_syn', 'flow_urg', 'flow_fin', 'flow_ece', 'flow_ack', 'flow_rst', 'flow_cwr'
]
scaler_count = StandardScaler()
packet_count_data = scaler_count.fit_transform(data[packet_count_features])
flow_flag_data = data[flow_flag_features].values

# Combine processed features
X = np.hstack([
    categorical_data, time_data1, time_data2, time_data3,
    packet_length_data_forward, packet_length_data_backward,
    packet_count_data, flow_flag_data
])

# Dimensionality reduction for clustering (optional)
pca = PCA(n_components=10, random_state=42)
X_reduced = pca.fit_transform(X)

# Apply DBSCAN clustering
dbscan = DBSCAN(eps=0.5, min_samples=5)
data['cluster'] = dbscan.fit_predict(X_reduced)

# Adjust cluster labels to match ground truth if necessary
cluster_mapping = {
    0: 1,
    1: 0
}
data['adjusted_cluster'] = data['cluster'].map(cluster_mapping).fillna(-1).astype(int)

# Filter out noise points (-1) for evaluation
data_filtered = data[data['cluster'] != -1]

# Evaluate clustering performance (Avg=macro)
if not data_filtered.empty:
    ma_accuracy = accuracy_score(data_filtered['label'], data_filtered['cluster'])
    ma_precision = precision_score(data_filtered['label'], data_filtered['cluster'], average='macro', zero_division=0)
    ma_recall = recall_score(data_filtered['label'], data_filtered['cluster'], average='macro', zero_division=0)
    ma_f1 = f1_score(data_filtered['label'], data_filtered['cluster'], average='macro', zero_division=0)
    ma_jaccard = jaccard_score(data_filtered['label'], data_filtered['cluster'], average='macro', zero_division=0)
    ma_silhouette = silhouette_score(X_reduced[data['cluster'] != -1], data_filtered['cluster'])
else:
    ma_accuracy = ma_precision = ma_recall = ma_f1 = ma_jaccard = ma_silhouette = np.nan

# Evaluate adjusted clustering performance (Avg=macro)
data_filtered_adjusted = data[data['adjusted_cluster'] != -1]
if not data_filtered_adjusted.empty:
    ma_accuracy_adjusted = accuracy_score(data_filtered_adjusted['label'], data_filtered_adjusted['adjusted_cluster'])
    ma_precision_adjusted = precision_score(data_filtered_adjusted['label'], data_filtered_adjusted['adjusted_cluster'], average='macro', zero_division=0)
    ma_recall_adjusted = recall_score(data_filtered_adjusted['label'], data_filtered_adjusted['adjusted_cluster'], average='macro', zero_division=0)
    ma_f1_adjusted = f1_score(data_filtered_adjusted['label'], data_filtered_adjusted['adjusted_cluster'], average='macro', zero_division=0)
    ma_jaccard_adjusted = jaccard_score(data_filtered_adjusted['label'], data_filtered_adjusted['adjusted_cluster'], average='macro', zero_division=0)
    ma_silhouette_adjusted = silhouette_score(X_reduced[data['adjusted_cluster'] != -1], data_filtered_adjusted['adjusted_cluster'])
else:
    ma_accuracy_adjusted = ma_precision_adjusted = ma_recall_adjusted = ma_f1_adjusted = ma_jaccard_adjusted = ma_silhouette_adjusted = np.nan

# Evaluate clustering performance (Avg=micro)
if not data_filtered.empty:
    mi_accuracy = accuracy_score(data_filtered['label'], data_filtered['cluster'])
    mi_precision = precision_score(data_filtered['label'], data_filtered['cluster'], average='micro', zero_division=0)
    mi_recall = recall_score(data_filtered['label'], data_filtered['cluster'], average='micro', zero_division=0)
    mi_f1 = f1_score(data_filtered['label'], data_filtered['cluster'], average='micro', zero_division=0)
    mi_jaccard = jaccard_score(data_filtered['label'], data_filtered['cluster'], average='micro', zero_division=0)
    mi_silhouette = silhouette_score(X_reduced[data['cluster'] != -1], data_filtered['cluster'])
else:
    mi_accuracy = mi_precision = mi_recall = mi_f1 = mi_jaccard = mi_silhouette = np.nan

# Evaluate adjusted clustering performance (Avg=micro)
if not data_filtered_adjusted.empty:
    mi_accuracy_adjusted = accuracy_score(data_filtered_adjusted['label'], data_filtered_adjusted['adjusted_cluster'])
    mi_precision_adjusted = precision_score(data_filtered_adjusted['label'], data_filtered_adjusted['adjusted_cluster'], average='micro', zero_division=0)
    mi_recall_adjusted = recall_score(data_filtered_adjusted['label'], data_filtered_adjusted['adjusted_cluster'], average='micro', zero_division=0)
    mi_f1_adjusted = f1_score(data_filtered_adjusted['label'], data_filtered_adjusted['adjusted_cluster'], average='micro', zero_division=0)
    mi_jaccard_adjusted = jaccard_score(data_filtered_adjusted['label'], data_filtered_adjusted['adjusted_cluster'], average='micro', zero_division=0)
    mi_silhouette_adjusted = silhouette_score(X_reduced[data['adjusted_cluster'] != -1], data_filtered_adjusted['adjusted_cluster'])
else:
    mi_accuracy_adjusted = mi_precision_adjusted = mi_recall_adjusted = mi_f1_adjusted = mi_jaccard_adjusted = mi_silhouette_adjusted = np.nan

# Evaluate clustering performance (Avg=weighted)
if not data_filtered.empty:
    w_accuracy = accuracy_score(data_filtered['label'], data_filtered['cluster'])
    w_precision = precision_score(data_filtered['label'], data_filtered['cluster'], average='weighted', zero_division=0)
    w_recall = recall_score(data_filtered['label'], data_filtered['cluster'], average='weighted', zero_division=0)
    w_f1 = f1_score(data_filtered['label'], data_filtered['cluster'], average='weighted', zero_division=0)
    w_jaccard = jaccard_score(data_filtered['label'], data_filtered['cluster'], average='weighted', zero_division=0)
    w_silhouette = silhouette_score(X_reduced[data['cluster'] != -1], data_filtered['cluster'])
else:
    w_accuracy = w_precision = w_recall = w_f1 = w_jaccard = w_silhouette = np.nan

# Evaluate adjusted clustering performance (Avg=weighted)
if not data_filtered_adjusted.empty:
    w_accuracy_adjusted = accuracy_score(data_filtered_adjusted['label'], data_filtered_adjusted['adjusted_cluster'])
    w_precision_adjusted = precision_score(data_filtered_adjusted['label'], data_filtered_adjusted['adjusted_cluster'], average='weighted', zero_division=0)
    w_recall_adjusted = recall_score(data_filtered_adjusted['label'], data_filtered_adjusted['adjusted_cluster'], average='weighted', zero_division=0)
    w_f1_adjusted = f1_score(data_filtered_adjusted['label'], data_filtered_adjusted['adjusted_cluster'], average='weighted', zero_division=0)
    w_jaccard_adjusted = jaccard_score(data_filtered_adjusted['label'], data_filtered_adjusted['adjusted_cluster'], average='weighted', zero_division=0)
    w_silhouette_adjusted = silhouette_score(X_reduced[data['adjusted_cluster'] != -1], data_filtered_adjusted['adjusted_cluster'])
else:
    w_accuracy_adjusted = w_precision_adjusted = w_recall_adjusted = w_f1_adjusted = w_jaccard_adjusted = w_silhouette_adjusted = np.nan

# Save results to CSV
data[['cluster', 'adjusted_cluster', 'label']].to_csv("./MiraiBotnet_DBSCAN_clustering_Compare.csv", index=False)

# Save metrics to CSV
metrics = {
    'Metric': ['Accuracy', 'Precision', 'Recall', 'F1 Score', 'Jaccard Index', 'silhouette'],
    'Macro_Original': [ma_accuracy, ma_precision, ma_recall, ma_f1, ma_jaccard, ma_silhouette],
    'Macro_Adjusted': [ma_accuracy_adjusted, ma_precision_adjusted, ma_recall_adjusted, ma_f1_adjusted, ma_jaccard_adjusted, ma_silhouette_adjusted],
    'Micro_Original': [mi_accuracy, mi_precision, mi_recall, mi_f1, mi_jaccard, mi_silhouette],
    'Micro_Adjusted': [mi_accuracy_adjusted, mi_precision_adjusted, mi_recall_adjusted, mi_f1_adjusted, mi_jaccard_adjusted, mi_silhouette_adjusted],
    'Weighted_Original': [w_accuracy, w_precision, w_recall, w_f1, w_jaccard, w_silhouette],
    'Weighted_Adjusted': [w_accuracy_adjusted, w_precision_adjusted, w_recall_adjusted, w_f1_adjusted, w_jaccard_adjusted, w_silhouette_adjusted]
}
metrics_df = pd.DataFrame(metrics)
metrics_df.to_csv("./MiraiBotnet_DBSCAN_clustering_Compare_Metrics.csv", index=False)

# Print evaluation metrics
print("Macro Clustering Performance Metrics:")
print(f"Accuracy: {ma_accuracy:.2f}")
print(f"Precision: {ma_precision:.2f}")
print(f"Recall: {ma_recall:.2f}")
print(f"F1 Score: {ma_f1:.2f}")
print(f"Jaccard Index: {ma_jaccard:.2f}")
print(f"Silhouette Score: {ma_silhouette:.2f}")

print("\nMacro Adjusted Clustering Performance Metrics:")
print(f"Adjusted Accuracy: {ma_accuracy_adjusted:.2f}")
print(f"Adjusted Precision: {ma_precision_adjusted:.2f}")
print(f"Adjusted Recall: {ma_recall_adjusted:.2f}")
print(f"Adjusted F1 Score: {ma_f1_adjusted:.2f}")
print(f"Adjusted Jaccard Index: {ma_jaccard_adjusted:.2f}")
print(f"Silhouette Score: {ma_silhouette_adjusted:.2f}")

print("\nMicro Clustering Performance Metrics:")
print(f"Accuracy: {mi_accuracy:.2f}")
print(f"Precision: {mi_precision:.2f}")
print(f"Recall: {mi_recall:.2f}")
print(f"F1 Score: {mi_f1:.2f}")
print(f"Jaccard Index: {mi_jaccard:.2f}")
print(f"Silhouette Score: {mi_silhouette:.2f}")

print("\nMicro Adjusted Clustering Performance Metrics:")
print(f"Adjusted Accuracy: {mi_accuracy_adjusted:.2f}")
print(f"Adjusted Precision: {mi_precision_adjusted:.2f}")
print(f"Adjusted Recall: {mi_recall_adjusted:.2f}")
print(f"Adjusted F1 Score: {mi_f1_adjusted:.2f}")
print(f"Adjusted Jaccard Index: {mi_jaccard_adjusted:.2f}")
print(f"Silhouette Score: {mi_silhouette_adjusted:.2f}")

print("\nWeighted Clustering Performance Metrics:")
print(f"Accuracy: {w_accuracy:.2f}")
print(f"Precision: {w_precision:.2f}")
print(f"Recall: {w_recall:.2f}")
print(f"F1 Score: {w_f1:.2f}")
print(f"Jaccard Index: {w_jaccard:.2f}")
print(f"Silhouette Score: {w_silhouette:.2f}")

print("\nWeighted Adjusted Clustering Performance Metrics:")
print(f"Adjusted Accuracy: {w_accuracy_adjusted:.2f}")
print(f"Adjusted Precision: {w_precision_adjusted:.2f}")
print(f"Adjusted Recall: {w_recall_adjusted:.2f}")
print(f"Adjusted F1 Score: {w_f1_adjusted:.2f}")
print(f"Adjusted Jaccard Index: {w_jaccard_adjusted:.2f}")
print(f"Silhouette Score: {w_silhouette_adjusted:.2f}")
