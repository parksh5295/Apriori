import skfuzzy as fuzz
from sklearn.preprocessing import StandardScaler, OneHotEncoder
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, jaccard_score, silhouette_score
from tqdm import tqdm
from CICIDS2017_CSV_Selector import select_csv_file

# Load sample data
select_csv = select_csv_file()
data = pd.read_csv(select_csv)

# Define label (assumed 'Label' column contains ground truth)
data['label'] = data['Label'].apply(lambda x: 0 if x == "BENIGN" else 1)
features = data.drop(columns=["label"])

# Feature preprocessing
# 1. Categorical Data Embedding
categorical_features = ['Destination Port']
encoder = OneHotEncoder(sparse=False, handle_unknown='ignore')
categorical_data = encoder.fit_transform(data[categorical_features])
categorical_df = pd.DataFrame(categorical_data, columns=encoder.get_feature_names_out(categorical_features))

# 2. Timing Features
time_features = [
    'Flow Duration', 'Flow IAT Mean', 'Flow IAT Std', 'Flow IAT Max', 'Flow IAT Min',
    'Fwd IAT Total', 'Fwd IAT Mean', 'Fwd IAT Std', 'Fwd IAT Max', 'Fwd IAT Min',
    'Bwd IAT Total', 'Bwd IAT Mean', 'Bwd IAT Std', 'Bwd IAT Max', 'Bwd IAT Min',
    'Active Mean', 'Active Std', 'Active Max', 'Active Min',
    'Idle Mean', 'Idle Std', 'Idle Max', 'Idle Min'
]
scaler_time = StandardScaler()
time_data = scaler_time.fit_transform(data[time_features])
time_df = pd.DataFrame(time_data, columns=time_features)

# 3. Packet Length Features
packet_length_features = [
    'Total Length of Fwd Packets', 'Total Length of Bwd Packets',
    'Fwd Packet Length Max', 'Fwd Packet Length Min', 'Fwd Packet Length Mean', 'Fwd Packet Length Std',
    'Bwd Packet Length Max', 'Bwd Packet Length Min', 'Bwd Packet Length Mean', 'Bwd Packet Length Std',
    'Packet Length Mean', 'Packet Length Std', 'Packet Length Variance',
    'Avg Fwd Segment Size', 'Avg Bwd Segment Size'
]
scaler_packet = StandardScaler()
packet_data = scaler_packet.fit_transform(data[packet_length_features])
packet_df = pd.DataFrame(packet_data, columns=packet_length_features)

# 4. Count Features
count_features = [
    'Total Fwd Packets', 'Total Backward Packets',
    'Flow Bytes/s', 'Flow Packets/s',
    'Fwd Packets/s', 'Bwd Packets/s',
    'Fwd Header Length', 'Bwd Header Length',
    'Down/Up Ratio', 'Average Packet Size',
    'Fwd Avg Bytes/Bulk', 'Fwd Avg Packets/Bulk', 'Fwd Avg Bulk Rate',
    'Bwd Avg Bytes/Bulk', 'Bwd Avg Packets/Bulk', 'Bwd Avg Bulk Rate',
    'Subflow Fwd Packets', 'Subflow Fwd Bytes',
    'Subflow Bwd Packets', 'Subflow Bwd Bytes',
    'Init_Win_bytes_forward', 'Init_Win_bytes_backward',
    'act_data_pkt_fwd', 'min_seg_size_forward'
]
scaler_count = StandardScaler()
count_data = scaler_count.fit_transform(data[count_features])
count_df = pd.DataFrame(count_data, columns=count_features)

# 5. Binary Features
binary_features = [
    'Fwd PSH Flags', 'Bwd PSH Flags',
    'Fwd URG Flags', 'Bwd URG Flags',
    'FIN Flag Count', 'SYN Flag Count', 'RST Flag Count',
    'PSH Flag Count', 'ACK Flag Count', 'URG Flag Count',
    'CWE Flag Count', 'ECE Flag Count'
]
binary_df = data[binary_features].astype(int)

# Combine processed features
X = np.hstack([categorical_data, time_data, packet_data, count_data, binary_df.to_numpy()])

# Dimensionality reduction using PCA
pca = PCA(n_components=10, random_state=42)
X_reduced = pca.fit_transform(X)

# Apply Fuzzy C-Means clustering with progress bar
n_clusters = 2  # Number of clusters
fcm_model = fuzz.cluster.cmeans(X_reduced.T, c=n_clusters, m=2, error=0.005, maxiter=1000, init=None)

# Use tqdm to track the progress of the iterations
with tqdm(total=1000, desc="Fuzzy C-Means Clustering", unit="iteration") as pbar:
    for i in range(fcm_model[0].shape[1]):  # Loop through all iterations
        fcm_model = fuzz.cluster.cmeans(X_reduced.T, c=n_clusters, m=2, error=0.005, maxiter=1, init=fcm_model[0])
        pbar.update(1)

# Fuzzy C-means returns fuzzy partition matrix (U), where each row is the membership of a sample for each cluster
# We assign the cluster with the highest membership to each data point
cluster_labels = np.argmax(fcm_model[0], axis=0)

data['cluster'] = cluster_labels

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
data[['cluster', 'label']].to_csv("./CICIDS2017_FCM_clustering_Compare.csv", index=False)
metrics_df = pd.DataFrame([metrics_original], index=["Original"])
metrics_df.to_csv("./CICIDS2017_FCM_clustering_Compare_Metrics.csv", index=True)

# Print Evaluation Results
print("\nClustering & Evaluation Completed!")
