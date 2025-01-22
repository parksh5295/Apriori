import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, jaccard_score, silhouette_score
from keras.layers import Dense, Flatten, Attention
import tensorflow as tf

# CANN with KNN

# Load sample data
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
print(X.shape)

# Defining the CANN model
class CANN(tf.keras.Model):
    def __init__(self, input_shape):
        super(CANN, self).__init__()
        self.dense1 = Dense(64, activation='relu', input_shape=input_shape)
        self.attention = Attention()
        self.flatten = Flatten()
        self.dense2 = Dense(32, activation='relu')
        self.dense3 = Dense(1, activation='sigmoid')
    
    def call(self, inputs):
        x = self.dense1(inputs)
        x = self.attention([x, x])  # Applying the Attention mechanism
        x = self.flatten(x)
        x = self.dense2(x)
        return self.dense3(x)

# Define model input shapes
input_shape = (X.shape[1],)  # 35 features

# Create CANN Model
model = CANN(input_shape)
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Learning CANN Model
model.fit(X, data['label'], epochs=5, batch_size=32)

# Feature Extraction with CANN Models
features = model.predict(X)

# Apply K-NN clustering
knn = KNeighborsClassifier(n_neighbors=3)
knn.fit(features, data['label'])

# Predict
predictions = knn.predict(features)
predictions_reversed = 1 - predictions

# avg = macro

# Evaluation
ma_accuracy = accuracy_score(data['label'], predictions)
ma_precision = precision_score(data['label'], predictions, average='macro', zero_division=0)
ma_recall = recall_score(data['label'], predictions, average='macro', zero_division=0)
ma_f1 = f1_score(data['label'], predictions, average='macro', zero_division=0)
ma_jaccard = jaccard_score(data['label'], predictions, average='macro', zero_division=0)

# adjust
ma_accuracy_adj = accuracy_score(data['label'], predictions_reversed)
ma_precision_adj = precision_score(data['label'], predictions_reversed, average='macro', zero_division=0)
ma_recall_adj = recall_score(data['label'], predictions_reversed, average='macro', zero_division=0)
ma_f1_adj = f1_score(data['label'], predictions_reversed, average='macro', zero_division=0)
ma_jaccard_adj = jaccard_score(data['label'], predictions_reversed, average='macro', zero_division=0)

# avg = micro

# Evaluation
mi_accuracy = accuracy_score(data['label'], predictions)
mi_precision = precision_score(data['label'], predictions, average='micro', zero_division=0)
mi_recall = recall_score(data['label'], predictions, average='micro', zero_division=0)
mi_f1 = f1_score(data['label'], predictions, average='micro', zero_division=0)
mi_jaccard = jaccard_score(data['label'], predictions, average='micro', zero_division=0)

# adjust
mi_accuracy_adj = accuracy_score(data['label'], predictions_reversed)
mi_precision_adj = precision_score(data['label'], predictions_reversed, average='micro', zero_division=0)
mi_recall_adj = recall_score(data['label'], predictions_reversed, average='micro', zero_division=0)
mi_f1_adj = f1_score(data['label'], predictions_reversed, average='micro', zero_division=0)
mi_jaccard_adj = jaccard_score(data['label'], predictions_reversed, average='micro', zero_division=0)

# avg = weighted

# Evaluation
w_accuracy = accuracy_score(data['label'], predictions)
w_precision = precision_score(data['label'], predictions, average='weighted', zero_division=0)
w_recall = recall_score(data['label'], predictions, average='weighted', zero_division=0)
w_f1 = f1_score(data['label'], predictions, average='weighted', zero_division=0)
w_jaccard = jaccard_score(data['label'], predictions, average='weighted', zero_division=0)

# adjust
w_accuracy_adj = accuracy_score(data['label'], predictions_reversed)
w_precision_adj = precision_score(data['label'], predictions_reversed, average='weighted', zero_division=0)
w_recall_adj = recall_score(data['label'], predictions_reversed, average='weighted', zero_division=0)
w_f1_adj = f1_score(data['label'], predictions_reversed, average='weighted', zero_division=0)
w_jaccard_adj = jaccard_score(data['label'], predictions_reversed, average='weighted', zero_division=0)

# Evaluate clustering performance
silhouette = silhouette_score(features, predictions)
silhouette_adj = silhouette_score(features, predictions_reversed)

# Save metrics to CSV
metrics = {
    'Metric': ['Accuracy', 'Precision', 'Recall', 'F1 Score', 'Jaccard Index', 'silhouette'],
    'Macro_Original': [ma_accuracy, ma_precision, ma_recall, ma_f1, ma_jaccard, silhouette],
    'Macro_Adjusted': [ma_accuracy_adj, ma_precision_adj, ma_recall_adj, ma_f1_adj, ma_jaccard_adj, silhouette_adj],
    'Micro_Original': [mi_accuracy, mi_precision, mi_recall, mi_f1, mi_jaccard, silhouette],
    'Micro_Adjusted': [mi_accuracy_adj, mi_precision_adj, mi_recall_adj, mi_f1_adj, mi_jaccard_adj, silhouette_adj],
    'Weighted_Original': [w_accuracy, w_precision, w_recall, w_f1, w_jaccard, silhouette],
    'Weighted_Adjusted': [w_accuracy_adj, w_precision_adj, w_recall_adj, w_f1_adj, w_jaccard_adj, silhouette_adj]
}
metrics_df = pd.DataFrame(metrics)
metrics_df.to_csv("./MiraiBotnet_CANNwKNN_clustering_Compare_Metrics_v.1-1.csv", index=False)

# printing Results
print("Macro Clustering Performance Metrics:")
print(f"Accuracy: {ma_accuracy:.2f}")
print(f"Precision: {ma_precision:.2f}")
print(f"Recall: {ma_recall:.2f}")
print(f"F1 Score: {ma_f1:.2f}")
print(f"Jaccard Index: {ma_jaccard:.2f}")
print(f"Silhouette Score: {silhouette:.2f}")

print("\nMacro Adjusted Clustering Performance Metrics:")
print(f"Adjusted Accuracy: {ma_accuracy_adj:.2f}")
print(f"Adjusted Precision: {ma_precision_adj:.2f}")
print(f"Adjusted Recall: {ma_recall_adj:.2f}")
print(f"Adjusted F1 Score: {ma_f1_adj:.2f}")
print(f"Adjusted Jaccard Index: {ma_jaccard_adj:.2f}")
print(f"Silhouette Score: {silhouette_adj:.2f}")

print("\nMicro Clustering Performance Metrics:")
print(f"Accuracy: {mi_accuracy:.2f}")
print(f"Precision: {mi_precision:.2f}")
print(f"Recall: {mi_recall:.2f}")
print(f"F1 Score: {mi_f1:.2f}")
print(f"Jaccard Index: {mi_jaccard:.2f}")
print(f"Silhouette Score: {silhouette:.2f}")

print("\nMicro Adjusted Clustering Performance Metrics:")
print(f"Adjusted Accuracy: {mi_accuracy_adj:.2f}")
print(f"Adjusted Precision: {mi_precision_adj:.2f}")
print(f"Adjusted Recall: {mi_recall_adj:.2f}")
print(f"Adjusted F1 Score: {mi_f1_adj:.2f}")
print(f"Adjusted Jaccard Index: {mi_jaccard_adj:.2f}")
print(f"Silhouette Score: {silhouette_adj:.2f}")

print("\nWeighted Clustering Performance Metrics:")
print(f"Accuracy: {w_accuracy:.2f}")
print(f"Precision: {w_precision:.2f}")
print(f"Recall: {w_recall:.2f}")
print(f"F1 Score: {w_f1:.2f}")
print(f"Jaccard Index: {w_jaccard:.2f}")
print(f"Silhouette Score: {silhouette:.2f}")

print("\nWeighted Adjusted Clustering Performance Metrics:")
print(f"Adjusted Accuracy: {w_accuracy_adj:.2f}")
print(f"Adjusted Precision: {w_precision_adj:.2f}")
print(f"Adjusted Recall: {w_recall_adj:.2f}")
print(f"Adjusted F1 Score: {w_f1_adj:.2f}")
print(f"Adjusted Jaccard Index: {w_jaccard_adj:.2f}")
print(f"Silhouette Score: {silhouette_adj:.2f}")
