import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.mixture import GaussianMixture
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, jaccard_score, silhouette_score
from sklearn.decomposition import PCA
from tqdm import tqdm

# Spherical GMM Algorithm Implementation (with GaussianMixture and spherical covariance)
class SphericalGMMCluster:
    def __init__(self, max_k=10, max_iter=300, tol=1e-4, random_state=42):
        self.max_k = max_k  # Try different k values (e.g., 2 to max_k)
        self.max_iter = max_iter
        self.tol = tol
        self.random_state = random_state

    def fit_predict(self, X):
        best_k = self.select_optimal_k(X)
        gmm = GaussianMixture(n_components=best_k, covariance_type='spherical', max_iter=self.max_iter, tol=self.tol, random_state=self.random_state)
        return gmm.fit_predict(X), best_k
    
    def select_optimal_k(self, X):
        silhouette_scores = []
        k_range = range(2, self.max_k+1)  # Try different cluster counts (e.g., 2 to max_k)

        for k in k_range:
            gmm = GaussianMixture(n_components=k, covariance_type='spherical', max_iter=self.max_iter, tol=self.tol, random_state=self.random_state)
            gmm.fit(X)
            labels = gmm.predict(X)
            score = silhouette_score(X, labels)
            silhouette_scores.append(score)

        best_k = np.argmax(silhouette_scores) + 2  # Add 2 because range starts at 2
        print(f"Best number of clusters (k): {best_k}")
        return best_k

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

# Apply Spherical GMM clustering
spherical_gmm_cluster = SphericalGMMCluster(max_k=10)  # Try up to 10 clusters
print("Clustering in Progress...")

with tqdm(total=len(X_reduced), desc="Clustering", unit="samples") as pbar:
    cluster_labels, best_k = spherical_gmm_cluster.fit_predict(X_reduced)
    data['cluster'] = cluster_labels
    pbar.update(len(X_reduced))

# Adjust Cluster Labels to Match Ground Truth (if needed)
cluster_mapping = {0: 1, 1: 0}
data['adjusted_cluster'] = data['cluster'].map(cluster_mapping)

# Evaluate Clustering Performance
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
data[['cluster', 'adjusted_cluster', 'label']].to_csv("./netML_SGMM_clustering_Compare.csv", index=False)
metrics_df = pd.DataFrame([metrics_original, metrics_adjusted], index=["Original", "Adjusted"])
metrics_df.to_csv("./netML_SGMM_clustering_Compare_Metrics.csv", index=True)

# Print Evaluation Results
print("\nClustering & Evaluation Completed!")
