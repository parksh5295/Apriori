import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.mixture import GaussianMixture
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, jaccard_score, silhouette_score
from sklearn.decomposition import PCA
from tqdm import tqdm

# 1. Load Kitsune (MitM) Dataset
data_file = "../Data_Resources/ARP_MitM_Kitsune/ARP_MitM_dataset.csv/ARP_MitM_dataset_final.csv"
data = pd.read_csv(data_file)

# 2. Define Label for Attack vs Benign
data['label'] = (data.iloc[:, -1] != 0).astype(int)  # 0 for normal, 1 for attack

# 3. Feature-specific embedding and preprocessing
# Categorical Features (e.g., protocol type)
categorical_features = []  # Kitsune has no explicit categorical features
if categorical_features:
    encoder = OneHotEncoder(sparse=False, handle_unknown='ignore')
    categorical_data = encoder.fit_transform(data[categorical_features])
else:
    categorical_data = np.empty((len(data), 0))  # Treating as an empty array

# Timing Features (e.g., flow_iat, forward_iat, etc...)
timing_features = [
    'SrcMAC_IP_w_100ms', 'SrcMAC_IP_mu_100ms', 'SrcMAC_IP_sigma_100ms', 'SrcMAC_IP_max_100ms', 'SrcMAC_IP_min_100ms',
    'SrcIP_w_100ms', 'SrcIP_mu_100ms', 'SrcIP_sigma_100ms', 'SrcIP_max_100ms', 'SrcIP_min_100ms',
    'Channel_w_100ms', 'Channel_mu_100ms', 'Channel_sigma_100ms', 'Channel_max_100ms', 'Channel_min_100ms'
]
scaler_time = StandardScaler()
time_data = scaler_time.fit_transform(data[timing_features])

# Packet Length Features (e.g., forward/backward packet size)
packet_length_features = [
    'Socket_w_100ms', 'Socket_mu_100ms', 'Socket_sigma_100ms', 'Socket_max_100ms', 'Socket_min_100ms'
]
scaler_packet = StandardScaler()
packet_length_data = scaler_packet.fit_transform(data[packet_length_features])

# Count Features (e.g., packet counts per second)
count_features = [
    'Jitter_mu_100ms', 'Jitter_sigma_100ms', 'Jitter_max_100ms'
]
scaler_count = StandardScaler()
count_data = scaler_count.fit_transform(data[count_features])

# Binary Features (e.g., flags)
binary_features = []  # Kitsune dataset has no clear binary features
binary_data = np.empty((len(data), 0))  # Treating as an empty array

# 4. Combine Processed Features
X = np.hstack([categorical_data, time_data, packet_length_data, count_data, binary_data])

# 5. Dimensionality Reduction using PCA
pca = PCA(n_components=10, random_state=42)
X_reduced = pca.fit_transform(X)

# 6. Apply Spherical Gaussian Mixture Model (SphericalGMM)
num_clusters = 2  # Assuming binary classification
gmm = GaussianMixture(n_components=num_clusters, covariance_type='spherical', random_state=42)

# Run clustering with progress bar
print("Clustering in Progress...")
with tqdm(total=len(X_reduced), desc="Clustering", unit="samples") as pbar:
    data['cluster'] = gmm.fit_predict(X_reduced)
    pbar.update(len(X_reduced))  # Update when clustering is complete

# 7. Adjust Cluster Labels to Match Ground Truth (if needed)
cluster_mapping = {0: 1, 1: 0}
data['adjusted_cluster'] = data['cluster'].map(cluster_mapping)

# 8. Evaluate Clustering Performance
def evaluate_clustering(y_true, y_pred, X_data):
    if not y_true.empty:
        return {
            "macro_accuracy": accuracy_score(y_true, y_pred),
            "macro_precision": precision_score(y_true, y_pred, average='macro', zero_division=0),
            "macro_recall": recall_score(y_true, y_pred, average='macro', zero_division=0),
            "macro_f1_score": f1_score(y_true, y_pred, average='macro', zero_division=0),
            "macro_jaccard": jaccard_score(y_true, y_pred, average='macro', zero_division=0),
            "macro_silhouette": silhouette_score(X_data, y_pred) if len(set(y_pred)) > 1 else np.nan,
            
            "micro_accuracy": accuracy_score(y_true, y_pred),
            "micro_precision": precision_score(y_true, y_pred, average='micro', zero_division=0),
            "micro_recall": recall_score(y_true, y_pred, average='micro', zero_division=0),
            "micro_f1_score": f1_score(y_true, y_pred, average='micro', zero_division=0),
            "micro_jaccard": jaccard_score(y_true, y_pred, average='micro', zero_division=0),
            "micro_silhouette": silhouette_score(X_data, y_pred) if len(set(y_pred)) > 1 else np.nan,
            
            "weighted_accuracy": accuracy_score(y_true, y_pred),
            "weighted_precision": precision_score(y_true, y_pred, average='weighted', zero_division=0),
            "weighted_recall": recall_score(y_true, y_pred, average='weighted', zero_division=0),
            "weighted_f1_score": f1_score(y_true, y_pred, average='weighted', zero_division=0),
            "weighted_jaccard": jaccard_score(y_true, y_pred, average='weighted', zero_division=0),
            "weighted_silhouette": silhouette_score(X_data, y_pred) if len(set(y_pred)) > 1 else np.nan,
        }
    return {}

metrics_original = evaluate_clustering(data['label'], data['cluster'], X_reduced)
metrics_adjusted = evaluate_clustering(data['label'], data['adjusted_cluster'], X_reduced)

# Save Results to CSV
data[['cluster', 'adjusted_cluster', 'label']].to_csv("./MitM_SGMM_clustering_Compare.csv", index=False)
metrics_df = pd.DataFrame([metrics_original, metrics_adjusted], index=["Original", "Adjusted"])
metrics_df.to_csv("./MitM_SGMM_clustering_Compare_Metrics.csv", index=True)

# Print Evaluation Results
print("\nClustering & Evaluation Completed!")
