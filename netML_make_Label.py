import pandas as pd
import numpy as np
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.cluster import KMeans
from sklearn.feature_extraction.text import TfidfVectorizer
import matplotlib.pyplot as plt
from keras.layers import Input, Dense, LSTM, RepeatVector, TimeDistributed
from keras.models import Model
import networkx as nx
from node2vec import Node2Vec

# Load the data
data = pd.read_csv("./netML/2_training_set.json/2_training_set.csv")

# Helper functions
def parse_array_column(column):
    return column.apply(lambda x: np.array(eval(x)) if isinstance(x, str) else x)

def encode_onehot_dense(column):
    encoder = OneHotEncoder(sparse=False)
    onehot = encoder.fit_transform(column.values.reshape(-1, 1))
    dense_layer = Dense(16, activation='relu')  # Compress to 16 dimensions
    return dense_layer(onehot)

def custom_tokenize(text):
    # Example tokenization: split by `/` and `?`
    tokens = text.split('/')
    tokens = [subtoken.split('?') for token in tokens for subtoken in token]
    return [item for sublist in tokens for item in sublist]

# Embedding specific features
# 1. One-Hot Encoding + Dense Layer
for col in ["http_method", "dns_query_type", "dns_query_class"]:
    if col in data.columns:
        data[col] = encode_onehot_dense(data[col])

# 2. Custom Tokenization (http_uri)
if "http_uri" in data.columns:
    vectorizer = TfidfVectorizer(tokenizer=custom_tokenize)
    data["http_uri_embedded"] = list(vectorizer.fit_transform(data["http_uri"]).toarray())

# 3. GNN for src_port, dst_port, sa, da
# Assuming src_port and dst_port define edges and sa/da define nodes
if all(col in data.columns for col in ["src_port", "dst_port", "sa", "da"]):
    graph = nx.Graph()
    for idx, row in data.iterrows():
        graph.add_edge(row["sa"], row["da"], src_port=row["src_port"], dst_port=row["dst_port"])
    node2vec = Node2Vec(graph, dimensions=16, walk_length=10, num_walks=100, workers=4)
    model = node2vec.fit(window=5, min_count=1)
    data["gnn_embedded"] = [model.wv[node] for node in graph.nodes()]

# 4. Manual Feature Engineering (tls_cs)
if "tls_cs" in data.columns:
    data["tls_cs_len"] = data["tls_cs"].apply(lambda x: len(str(x)))  # Example feature
    data["tls_cs_unique"] = data["tls_cs"].apply(lambda x: len(set(str(x))))

# 5. Protocol-Specific Parsing (dns_query_name, tls_ext_types)
if "dns_query_name" in data.columns:
    data["dns_tld"] = data["dns_query_name"].apply(lambda x: x.split('.')[-1] if isinstance(x, str) else x)
    vectorizer = TfidfVectorizer()
    data["dns_query_embedded"] = list(vectorizer.fit_transform(data["dns_query_name"]).toarray())

if "tls_ext_types" in data.columns:
    data["tls_ext_count"] = data["tls_ext_types"].apply(lambda x: len(eval(x)) if isinstance(x, str) else 0)
    data["tls_ext_mean"] = data["tls_ext_types"].apply(lambda x: np.mean(eval(x)) if isinstance(x, str) and len(eval(x)) > 0 else 0)

# 6. Sequence Autoencoders for Remaining Strings
string_features = [col for col in data.columns if data[col].dtype == object]
for col in string_features:
    unique_strings = data[col].dropna().unique()
    max_length = max(len(str(x)) for x in unique_strings)

    # Prepare the data
    sequences = np.array([list(str(x).ljust(max_length)) for x in unique_strings])
    char_set = sorted(set(sequences.flatten()))
    char_to_int = {c: i for i, c in enumerate(char_set)}
    int_sequences = np.array([[char_to_int[c] for c in seq] for seq in sequences])

    # Autoencoder
    input_dim = len(char_set)
    input_seq = Input(shape=(max_length, input_dim))
    encoded = LSTM(64)(input_seq)
    decoded = RepeatVector(max_length)(encoded)
    decoded = LSTM(input_dim, return_sequences=True)(decoded)

    autoencoder = Model(input_seq, decoded)
    encoder = Model(input_seq, encoded)

    autoencoder.compile(optimizer='adam', loss='mse')
    autoencoder.fit(int_sequences, int_sequences, epochs=10, batch_size=32)

    data[f"{col}_embedded"] = encoder.predict(int_sequences)

# 숫자형 피처 처리 함수
def process_numeric_features(data):
    # 문자열 피처 제외 (예: http_method, dns_query_name 등 제외)
    string_features = [
        "index", "sa", "da", "http_method", "http_uri", "http_host",
        "http_code", "http_content_len", "http_content_type", "dns_query_type",
        "dns_answer_ip", "dns_answer_ttl", "dns_query_name"
    ]
    numeric_data = data.drop(columns=string_features, errors="ignore")
    
    # NaN 값 처리
    numeric_data = numeric_data.fillna(0)
    
    # StandardScaler로 표준화
    scaler = StandardScaler()
    scaled_numeric_data = scaler.fit_transform(numeric_data)
    
    return scaled_numeric_data

# 숫자형 피처 표준화
scaled_numeric_features = process_numeric_features(data)

# 모든 피처 결합
if embedded_features:  # 문자열 임베딩 결과가 있는 경우
    embedded_features = np.hstack(embedded_features)  # 임베딩 결과 결합
    full_data = np.hstack([scaled_numeric_features, embedded_features])  # 숫자형 데이터와 결합
else:
    full_data = scaled_numeric_features  # 숫자형 데이터만 사용

# NaN 또는 무한대 값 확인
if np.isnan(full_data).any() or np.isinf(full_data).any():
    print("Warning: NaN or Inf values detected in full_data.")
else:
    print("Data prepared successfully for PCA and clustering.")


# Prepare numeric data for PCA
numeric_data = np.hstack([
    np.vstack(data[col].values) if isinstance(data[col].iloc[0], np.ndarray) else data[col].values.reshape(-1, 1)
    for col in data.columns if col not in string_features
])

# Standardize data
scaler = StandardScaler()
scaled_data = scaler.fit_transform(numeric_data)

# PCA
pca = PCA(n_components=2)
pca_data = pca.fit_transform(scaled_data)

# Clustering
kmeans = KMeans(n_clusters=2, random_state=42)
labels = kmeans.fit_predict(pca_data)

# Visualization
plt.figure(figsize=(8, 6))
plt.scatter(pca_data[:, 0], pca_data[:, 1], c=labels, cmap="viridis", alpha=0.6)
plt.title("PCA 결과: 정상/공격 클러스터링")
plt.xlabel("PCA Component 1")
plt.ylabel("PCA Component 2")
plt.colorbar(label="Cluster")
plt.show()