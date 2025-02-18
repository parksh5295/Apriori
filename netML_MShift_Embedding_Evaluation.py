from sklearn.cluster import MeanShift
from sklearn.preprocessing import StandardScaler, OneHotEncoder
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, jaccard_score, silhouette_score
from tqdm import tqdm

# Load the dataset
data = pd.read_csv("../Data_Resources/netML/netML_dataset.csv")

# Define label (assumed 'Label' column contains ground truth)
data['label'] = data['Label'].apply(lambda x: 1 if x == "Attack" else 0)
features = data.drop(columns=["label"])

# Feature preprocessing
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

# Dimensionality reduction using PCA
pca = PCA(n_components=10, random_state=42)
X_reduced = pca.fit_transform(X)

# Apply MeanShift clustering
mean_shift = MeanShift(bandwidth=2)  # Set bandwidth based on your data (larger values group points into fewer clusters)
print("Clustering in Progress...")

with tqdm(total=len(X_reduced), desc="Clustering", unit="samples") as pbar:
    cluster_labels = mean_shift.fit_predict(X_reduced)
    data['cluster'] = cluster_labels
    pbar.update(len(X_reduced))

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

# Save Results to CSV
data[['cluster', 'label']].to_csv("./netML_MeanShift_clustering_Compare.csv", index=False)
metrics_df = pd.DataFrame([metrics_original], index=["Original"])
metrics_df.to_csv("./netML_MeanShift_clustering_Compare_Metrics.csv", index=True)

# Print Evaluation Results
print("\nClustering & Evaluation Completed!")
