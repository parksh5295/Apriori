import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, jaccard_score, silhouette_score
from sklearn.decomposition import PCA
from tqdm import tqdm
from sklearn.neighbors import KNeighborsClassifier
from sklearn.cluster import AgglomerativeClustering


# CANN with KNN Implementation
class CANN_KNN:
    def __init__(self, n_neighbors=5):
        self.n_neighbors = n_neighbors
        self.knn = KNeighborsClassifier(n_neighbors=self.n_neighbors)
        self.agglomerative = AgglomerativeClustering(n_clusters=2)

    def fit_predict(self, X):
        # 1. Create two clusters with Agglomerative Clustering
        cluster_labels = self.agglomerative.fit_predict(X)

        # 2️. Re-learning with KNN
        self.knn.fit(X, cluster_labels)

        # 3️. Return the final cluster result (consisting only of 0,1)
        return self.knn.predict(X)

# 1. Load Kitsune (MitM) Dataset
data_file = "../Data_Resources/ARP_MitM_Kitsune/ARP_MitM_dataset.csv/ARP_MitM_dataset_final.csv"
data = pd.read_csv(data_file)

# 2. Define Label for Attack vs Benign
data['Label'] = (data.iloc[:, -1] != 0).astype(int)  # 0 for normal, 1 for attack

# 3. Feature-specific embedding and preprocessing
categorical_features = []  # Kitsune has no explicit categorical features
if categorical_features:
    encoder = OneHotEncoder(sparse=False, handle_unknown='ignore')
    categorical_data = encoder.fit_transform(data[categorical_features])
else:
    categorical_data = np.empty((len(data), 0))

timing_features = [
    'SrcMAC_IP_w_100ms', 'SrcMAC_IP_mu_100ms', 'SrcMAC_IP_sigma_100ms', 'SrcMAC_IP_max_100ms', 'SrcMAC_IP_min_100ms',
    'SrcIP_w_100ms', 'SrcIP_mu_100ms', 'SrcIP_sigma_100ms', 'SrcIP_max_100ms', 'SrcIP_min_100ms',
    'Channel_w_100ms', 'Channel_mu_100ms', 'Channel_sigma_100ms', 'Channel_max_100ms', 'Channel_min_100ms'
]
scaler_time = StandardScaler()
time_data = scaler_time.fit_transform(data[timing_features])

packet_length_features = [
    'Socket_w_100ms', 'Socket_mu_100ms', 'Socket_sigma_100ms', 'Socket_max_100ms', 'Socket_min_100ms'
]
scaler_packet = StandardScaler()
packet_length_data = scaler_packet.fit_transform(data[packet_length_features])

count_features = ['Jitter_mu_100ms', 'Jitter_sigma_100ms', 'Jitter_max_100ms']
scaler_count = StandardScaler()
count_data = scaler_count.fit_transform(data[count_features])

binary_features = []  # Kitsune dataset has no clear binary features
binary_data = np.empty((len(data), 0))

# 4. Combine Processed Features
X = np.hstack([categorical_data, time_data, packet_length_data, count_data, binary_data])

# 5. Dimensionality Reduction using PCA
pca = PCA(n_components=10, random_state=42)
X_reduced = pca.fit_transform(X)

# 6. Apply CANN with KNN Clustering
cann_knn = CANN_KNN(n_neighbors=5)

print("Clustering in Progress...")
with tqdm(total=len(X_reduced), desc="Clustering", unit="samples") as pbar:
    cluster_labels = cann_knn.fit_predict(X_reduced)
    data['cluster'] = cluster_labels
    pbar.update(len(X_reduced))
print("\nClustering Done!")

# 7. Adjust Cluster Labels to Match Ground Truth (if needed)
cluster_mapping = {0: 1, 1: 0}
data['adjusted_cluster'] = data['cluster'].map(cluster_mapping)
print("\nMapping Done!")

# 8. Evaluate Clustering Performance
def evaluate_clustering(y_true, y_pred, X_data, sample_size=3000):
    if y_true.empty:
        return {}

    metrics = {}

    metric_functions = {
        "precision": precision_score,
        "recall": recall_score,
        "f1_score": f1_score,
        "jaccard": jaccard_score
    }
    average_types = ["macro", "micro", "weighted"]

    print("\n[INFO] Evaluating clustering metrics...")

    with tqdm(total=1 + len(metric_functions) * len(average_types) + 1, desc="Computing Metrics") as pbar:
        # Accuracy (Calculated separately, no average required)
        metrics["accuracy"] = accuracy_score(y_true, y_pred)
        pbar.update(1)

        # Precision, Recall, F1, Jaccard (macro/micro/weighted)
        for avg in average_types:
            for key, func in metric_functions.items():
                metrics[f"{avg}_{key}"] = func(y_true, y_pred, average=avg, zero_division=0)
                pbar.update(1)

        # Silhouette Score (Calculate after some samples)
        if len(set(y_pred)) > 1:
            sample_indices = np.random.choice(len(X_data), min(sample_size, len(X_data)), replace=False)
            X_sample = X_data[sample_indices]
            y_sample = np.array(y_pred)[sample_indices]
            metrics["silhouette"] = silhouette_score(X_sample, y_sample)
        else:
            metrics["silhouette"] = np.nan
        
        pbar.update(1)

    return metrics

metrics_original = evaluate_clustering(data['Label'], data['cluster'], X_reduced)
metrics_adjusted = evaluate_clustering(data['Label'], data['adjusted_cluster'], X_reduced)
print("\nScore Compute Done!")

'''
# Save Results to CSV
data[['cluster', 'adjusted_cluster', 'Label']].to_csv("./MitM_CANNwKNN_clustering_Compare.csv", index=False)
metrics_df = pd.DataFrame([metrics_original, metrics_adjusted], index=["Original", "Adjusted"])
metrics_df.to_csv("./MitM_CANNwKNN_clustering_Compare_Metrics.csv", index=True)
'''

batch_size = 10000  # Number of rows to save at once
# Save Results to CSV with Progress Bar
save_path = "./MitM_CANNwKNN_clustering_Compare_v.2.csv"
print(f"Saving CSV to {save_path}...")
with open(save_path, "w") as f:
    data[['cluster', 'adjusted_cluster', 'Label']].iloc[:0].to_csv(f, index=False)  # Write header
    with tqdm(total=len(data), desc="Saving CSV", unit="rows") as pbar:
        for i in range(0, len(data), batch_size):
            data.iloc[i:i+batch_size][['cluster', 'adjusted_cluster', 'Label']].to_csv(f, header=False, index=False)
            pbar.update(min(batch_size, len(data) - i))

# Save Metrics CSV with Progress Bar
metrics_df = pd.DataFrame([metrics_original, metrics_adjusted], index=["Original", "Adjusted"])
metrics_save_path = "./MitM_CANNwKNN_clustering_Compare_Metrics_v.2.csv"
metrics_df.to_csv(metrics_save_path, index=True)

# Print Evaluation Results
print("\nClustering & Evaluation Completed!")
