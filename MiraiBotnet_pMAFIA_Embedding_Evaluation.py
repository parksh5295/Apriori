import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.decomposition import PCA
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, jaccard_score, silhouette_score
from mlxtend.frequent_patterns import apriori, association_rules
import time
from tqdm import tqdm  # Import tqdm for progress bar

# Clustering Methods: pMAFIA

# Load sample data
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
    'total_length_of_forward_packets', 'total_length_of_backward_packets', 'flow_packets_per_second'
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
pca = PCA(n_components=20, random_state=42)
X_reduced = pca.fit_transform(X)

# --------------- pMAFIA Algorithm Implementation ---------------

# Step 1: Adaptive Grid Construction
num_bins = 10  # Set the number of grids (tunable)
grid_indices = np.digitize(X_reduced, bins=np.linspace(X_reduced.min(), X_reduced.max(), num_bins))

# Step 2: Frequent Pattern Mining using Apriori
df_grid = pd.DataFrame(grid_indices, columns=[f'Feature_{i}' for i in range(X_reduced.shape[1])])
df_grid = pd.get_dummies(df_grid, columns=df_grid.columns)
frequent_itemsets = apriori(df_grid, min_support=0.98, use_colnames=True)
print("Frequent_itemsets: ")
print(frequent_itemsets)
print(frequent_itemsets.columns)

# Step 3: Extract Clusters using Association Rules
rules = association_rules(frequent_itemsets, metric="confidence", min_threshold=0.7, num_itemsets=10)

# Use tqdm to show the progress bar for association rule processing
cluster_labels = np.zeros(len(X_reduced)) - 1  # Default: -1 (Noise)

for idx, row in tqdm(rules.iterrows(), total=len(rules), desc="Processing Rules"):
    matching_rows = df_grid[list(row['antecedents'])].eq(1).all(axis=1)
    cluster_labels[matching_rows] = idx

# Force the cluster labels to be 0 or 1 by mapping them
unique_clusters = np.unique(cluster_labels)
print(f"Unique Clusters Before Mapping: {unique_clusters}")

# Filter out noise (-1) and re-check the unique values
valid_clusters = unique_clusters[unique_clusters != -1]

# If there are exactly two unique clusters, map them to 0 and 1
if len(valid_clusters) == 2:
    cluster_map = {valid_clusters[0]: 0, valid_clusters[1]: 1}
    cluster_labels = np.array([cluster_map.get(label, -1) for label in cluster_labels])

# If only one valid cluster exists, map it to 0, and set the rest to -1 (noise)
elif len(valid_clusters) == 1:
    cluster_map = {valid_clusters[0]: 0}
    cluster_labels = np.array([cluster_map.get(label, -1) for label in cluster_labels])
else:
    # In case there are no valid clusters, or only noise, all will be marked as -1
    cluster_labels = np.full_like(cluster_labels, -1)

data['cluster'] = cluster_labels.astype(int)

# Check the result
print("Cluster Distribution after Mapping:")
print(data['cluster'].value_counts())

# --------------- Evaluation Metrics ---------------

# Original cluster evaluation (without any adjustment)
def compute_metrics(y_true, y_pred, X_filtered, average_type):
    # Remove NaN values from y_true and y_pred
    valid_indices = ~y_true.isna() & ~y_pred.isna()  # Find indices without NaN
    y_true_clean = y_true[valid_indices]
    y_pred_clean = y_pred[valid_indices]
    print(y_pred_clean)
    
    if not y_true_clean.empty:
        return {
            "accuracy": accuracy_score(y_true_clean, y_pred_clean),
            "precision": precision_score(y_true_clean, y_pred_clean, average=average_type, zero_division=0),
            "recall": recall_score(y_true_clean, y_pred_clean, average=average_type, zero_division=0),
            "f1": f1_score(y_true_clean, y_pred_clean, average=average_type, zero_division=0),
            "jaccard": jaccard_score(y_true_clean, y_pred_clean, average=average_type, zero_division=0),
            "silhouette": silhouette_score(X_filtered, y_pred_clean) if len(set(y_pred_clean)) > 1 else np.nan
        }
    return {metric: np.nan for metric in ["accuracy", "precision", "recall", "f1", "jaccard", "silhouette"]}

# Original evaluation
original_macro_metrics = compute_metrics(data['label'], data['cluster'], X_reduced[data['cluster'] != -1], average_type='macro')
original_micro_metrics = compute_metrics(data['label'], data['cluster'], X_reduced[data['cluster'] != -1], average_type='micro')
original_weighted_metrics = compute_metrics(data['label'], data['cluster'], X_reduced[data['cluster'] != -1], average_type='weighted')

# Adjusted cluster evaluation (flipping cluster labels)
cluster_mapping = {0: 1, 1: 0}
data['adjusted_cluster'] = data['cluster'].map(cluster_mapping)

# Adjusted evaluation
adjusted_macro_metrics = compute_metrics(data['label'], data['adjusted_cluster'], X_reduced[data['adjusted_cluster'] != -1], average_type='macro')
adjusted_micro_metrics = compute_metrics(data['label'], data['adjusted_cluster'], X_reduced[data['adjusted_cluster'] != -1], average_type='micro')
adjusted_weighted_metrics = compute_metrics(data['label'], data['adjusted_cluster'], X_reduced[data['adjusted_cluster'] != -1], average_type='weighted')

# Print results
print("\nOriginal Clustering Performance Metrics:")
for metric, value in original_macro_metrics.items():
    print(f"Original Macro {metric.capitalize()}: {value:.2f}")
for metric, value in original_micro_metrics.items():
    print(f"Original Micro {metric.capitalize()}: {value:.2f}")
for metric, value in original_weighted_metrics.items():
    print(f"Original Weighted {metric.capitalize()}: {value:.2f}")

print("\nAdjusted Clustering Performance Metrics:")
for metric, value in adjusted_macro_metrics.items():
    print(f"Adjusted Macro {metric.capitalize()}: {value:.2f}")
for metric, value in adjusted_micro_metrics.items():
    print(f"Adjusted Micro {metric.capitalize()}: {value:.2f}")
for metric, value in adjusted_weighted_metrics.items():
    print(f"Adjusted Weighted {metric.capitalize()}: {value:.2f}")

# Save results to CSV
data[['cluster', 'adjusted_cluster', 'label']].to_csv("./MiraiBotnet_pMAFIA_clustering_Compare.csv", index=False)

# Save metrics to CSV
metrics = {
    'Metric': ["Accuracy", "Precision", "Recall", "F1 Score", "Jaccard Index", "Silhouette Score"],
    'Original Macro': list(original_macro_metrics.values()),
    'Original Micro': list(original_micro_metrics.values()),
    'Original Weighted': list(original_weighted_metrics.values()),
    'Adjusted Macro': list(adjusted_macro_metrics.values()),
    'Adjusted Micro': list(adjusted_micro_metrics.values()),
    'Adjusted Weighted': list(adjusted_weighted_metrics.values())
}
metrics_df = pd.DataFrame(metrics)
metrics_df.to_csv("./MiraiBotnet_pMAFIA_clustering_Compare_Metrics.csv", index=False)
