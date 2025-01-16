import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.cluster import KMeans
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
time_features = [
    'flow_iat_max', 'forward_iat_std', 'forward_iat_min', 'flow_iat_min',
    'backward_iat_min', 'backward_iat_max', 'flow_iat_total', 'forward_iat_mean', 'flow_iat_mean'
]
scaler_time = StandardScaler()
time_data = scaler_time.fit_transform(data[time_features])

# 3. Packet Length Features
packet_length_features = [
    'total_length_of_forward_packets', 'backward_packet_length_max', 'backward_packet_length_std',
    'forward_packet_length_mean', 'forward_packet_length_min', 'forward_packet_length_std',
    'backward_packet_length_mean', 'backward_packet_length_min', 'total_length_of_backward_packets'
]
scaler_packet = StandardScaler()
packet_length_data = scaler_packet.fit_transform(data[packet_length_features])

# 4. Count Features (e.g., packet counts, flow counts)
count_features = [
    'fpkts_per_second', 'bpkts_per_second', 'total_bhlen', 'total_fhlen',
    'total_backward_packets', 'total_forward_packets', 'flow_packets_per_second'
]
scaler_count = StandardScaler()
count_data = scaler_count.fit_transform(data[count_features])

# 5. Binary Features (e.g., flow flags)
binary_features = [
    'flow_psh', 'flow_syn', 'flow_urg', 'flow_fin', 'flow_ece', 'flow_ack', 'flow_rst', 'flow_cwr'
]
binary_data = data[binary_features].values

# Combine processed features
X = np.hstack([categorical_data, time_data, packet_length_data, count_data, binary_data])

# Dimensionality reduction for clustering (optional)
pca = PCA(n_components=10, random_state=42)
X_reduced = pca.fit_transform(X)

# Apply K-Means clustering
kmeans = KMeans(n_clusters=2, random_state=42)
data['cluster'] = kmeans.fit_predict(X_reduced)

# Adjust cluster labels to match ground truth if necessary
cluster_mapping = {
    0: 1,
    1: 0
}
data['adjusted_cluster'] = data['cluster'].map(cluster_mapping)

# Evaluate clustering performance
accuracy = accuracy_score(data['label'], data['cluster'])
precision = precision_score(data['label'], data['cluster'])
recall = recall_score(data['label'], data['cluster'])
f1 = f1_score(data['label'], data['cluster'])
jaccard = jaccard_score(data['label'], data['cluster'])
silhouette = silhouette_score(X, data['cluster'])
# silhouette = silhouette_score(X_reduced, data['cluster'])  # PCA ver

# Evaluate adjusted clustering performance
accuracy_adjusted = accuracy_score(data['label'], data['adjusted_cluster'])
precision_adjusted = precision_score(data['label'], data['adjusted_cluster'])
recall_adjusted = recall_score(data['label'], data['adjusted_cluster'])
f1_adjusted = f1_score(data['label'], data['adjusted_cluster'])
jaccard_adjusted = jaccard_score(data['label'], data['adjusted_cluster'])
silhouette_adjusted = silhouette_score(X, data['adjusted_cluster'])
# silhouette_adjusted = silhouette_score(X_reduced, data['adjusted_cluster'])  # PCA ver

# Save results to CSV
data[['cluster', 'adjusted_cluster', 'label']].to_csv("./MiraiBotnet_Kmeans_clustering_Compare_v.1-1.csv", index=False)

# Save metrics to CSV
metrics = {
    'Metric': ['Accuracy', 'Precision', 'Recall', 'F1 Score', 'Jaccard Index', 'silhouette'],
    'Original': [accuracy, precision, recall, f1, jaccard, silhouette],
    'Adjusted': [accuracy_adjusted, precision_adjusted, recall_adjusted, f1_adjusted, jaccard_adjusted, silhouette_adjusted]
}
metrics_df = pd.DataFrame(metrics)
metrics_df.to_csv("./MiraiBotnet_Kmeans_clustering_Compare_Metrics_v.1-1.csv", index=False)

# Print evaluation metrics
print("Clustering Performance Metrics:")
print(f"Accuracy: {accuracy:.2f}")
print(f"Precision: {precision:.2f}")
print(f"Recall: {recall:.2f}")
print(f"F1 Score: {f1:.2f}")
print(f"Jaccard Index: {jaccard:.2f}")
print(f"Silhouette Score: {silhouette:.2f}")

print("\nAdjusted Clustering Performance Metrics:")
print(f"Adjusted Accuracy: {accuracy_adjusted:.2f}")
print(f"Adjusted Precision: {precision_adjusted:.2f}")
print(f"Adjusted Recall: {recall_adjusted:.2f}")
print(f"Adjusted F1 Score: {f1_adjusted:.2f}")
print(f"Adjusted Jaccard Index: {jaccard_adjusted:.2f}")
print(f"Silhouette Score: {silhouette_adjusted:.2f}")