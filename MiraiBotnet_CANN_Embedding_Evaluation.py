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

# Evaluation
accuracy = accuracy_score(data['label'], predictions)
precision = precision_score(data['label'], predictions, average='macro', zero_division=0)
recall = recall_score(data['label'], predictions, average='macro', zero_division=0)
f1 = f1_score(data['label'], predictions, average='macro', zero_division=0)
jaccard = jaccard_score(data['label'], predictions, average='macro', zero_division=0)

# adjust
predictions_reversed = 1 - predictions
accuracy_2 = accuracy_score(data['label'], predictions_reversed)
precision_2 = precision_score(data['label'], predictions_reversed, average='macro', zero_division=0)
recall_2 = recall_score(data['label'], predictions_reversed, average='macro', zero_division=0)
f1_2 = f1_score(data['label'], predictions_reversed, average='macro', zero_division=0)
jaccard_2 = jaccard_score(data['label'], predictions_reversed, average='macro', zero_division=0)

# Evaluate clustering performance
silhouette = silhouette_score(features, predictions)
silhouette_adj = silhouette_score(features, predictions_reversed)

# printing Results
print("Orthodox:")
print(f"Accuracy: {accuracy:.2f}")
print(f"Precision: {precision:.2f}")
print(f"Recall: {recall:.2f}")
print(f"F1 Score: {f1:.2f}")
print(f"Jaccard Index: {jaccard:.2f}")
print(f"Silhouette Score: {silhouette:.2f}")

print("/n Adj:")
print(f"Accuracy: {accuracy:.2f}")
print(f"Precision: {precision:.2f}")
print(f"Recall: {recall:.2f}")
print(f"F1 Score: {f1:.2f}")
print(f"Jaccard Index: {jaccard:.2f}")
print(f"Silhouette Score: {silhouette_adj:.2f}")

# save to csv
metrics = {
    'Metric': ['Accuracy', 'Precision', 'Recall', 'F1 Score', 'Jaccard Index', 'Silhouette'],
    'Value': [accuracy, precision, recall, f1, jaccard, silhouette],
    'Adj-Value': [accuracy_2, precision_2, recall_2, f1_2, jaccard_2, silhouette_adj]
}
metrics_df = pd.DataFrame(metrics)
metrics_df.to_csv("./MiraiBotnet_CANNwKNN_clustering_Compare_Metrics.csv", index=False)
