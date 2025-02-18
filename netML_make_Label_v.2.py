import pandas as pd
import json
import numpy as np
import networkx as nx
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
from keras.layers import Input, Dense, LSTM, RepeatVector, TimeDistributed
from keras.models import Model
from node2vec import Node2Vec

# 0. Label Mapping
# JSON file reading
with open('../Data_Resources/netML-Competition2020/2_training_anno_fine.json/2_training_anno_fine.json', 'r') as f:
    fine_annotations = json.load(f)

with open('../Data_Resources/netML-Competition2020/2_training_anno_top.json/2_training_anno_top.json', 'r') as f:
    top_annotations = json.load(f)

# Change JSON data to dictionary
fine_dict = fine_annotations
top_dict = top_annotations

# 1. Loading data
data = pd.read_csv("../Data_Resources/netML-Competition2020/2_training_set.json/2_training_set.csv")

# Add 'id' label to columns
data['label_fine'] = data['id'].map(fine_dict).fillna('unknown')
data['label_top'] = data['id'].map(top_dict).fillna('unknown')

data = data.drop(['sa', 'da'], axis=1) # because of IP_masked

def remove_mixed_dtype_columns(df):
    mixed_columns = []

    for column in df.columns:
        # Save each column's data type as a set
        unique_types = set(df[column].map(type))
        
        # More than one data type is considered a mixed column
        if len(unique_types) > 1:
            mixed_columns.append(column)
    
    # Remove mixed heat
    cleaned_df = df.drop(columns=mixed_columns)
    
    return cleaned_df

data = remove_mixed_dtype_columns(data)

# 2. Define a String Feature Embedding Function
def embed_one_hot_dense(data, column):
    unique_vals = data[column].dropna().unique()
    one_hot_map = {val: i for i, val in enumerate(unique_vals)}
    data[f"{column}_encoded"] = data[column].map(one_hot_map)
    one_hot_encoded = pd.get_dummies(data[f"{column}_encoded"])
    return one_hot_encoded.values

def embed_sequence_autoencoder(data, column, latent_dim=10):
    sequences = data[column].dropna().apply(eval).tolist()
    max_len = max(len(seq) for seq in sequences)
    padded_sequences = np.array([np.pad(seq, (0, max_len - len(seq))) for seq in sequences])

    # Autoencoder Model
    input_seq = Input(shape=(max_len, 1))
    encoded = LSTM(latent_dim)(input_seq)
    decoded = RepeatVector(max_len)(encoded)
    decoded = LSTM(1, return_sequences=True)(decoded)
    autoencoder = Model(input_seq, decoded)
    autoencoder.compile(optimizer="adam", loss="mse")

    # Learning
    autoencoder.fit(padded_sequences[..., np.newaxis], padded_sequences[..., np.newaxis], epochs=10, verbose=0)
    encoder = Model(input_seq, encoded)

    # Embedding
    return encoder.predict(padded_sequences[..., np.newaxis])

# 3. Apply String Feature Embedding
embedded_features = []

# 1. One-Hot Encoding + Dense Layer
for col in ["http_method", "dns_query_type", "dns_query_class"]:
    if col in data.columns:
        embedded_features.append(embed_one_hot_dense(data, col))

# 2. Custom Tokenization and TF-IDF Vectorization
if "http_uri" in data.columns:
    from sklearn.feature_extraction.text import TfidfVectorizer
    def custom_tokenize(text):
        tokens = text.split('/')
        tokens = [subtoken.split('?') for token in tokens for subtoken in token]
        return [item for sublist in tokens for item in sublist]
    vectorizer = TfidfVectorizer(tokenizer=custom_tokenize)
    tfidf_embedded = vectorizer.fit_transform(data["http_uri"].fillna(""))
    embedded_features.append(tfidf_embedded.toarray())

# 3. GNN Embedding
if all(col in data.columns for col in ["src_port", "dst_port", "bytes_in", "bytes_out"]):
    import networkx as nx
    from node2vec import Node2Vec
    graph = nx.Graph()
    
    for idx, row in data.iterrows():
        # Graph Trunk: bytes_in, bytes_out
        graph.add_edge(row["src_port"], row["dst_port"], 
                       weight_in=row["bytes_in"], weight_out=row["bytes_out"])
    
    # Node2Vec Model Learning
    node2vec = Node2Vec(graph, dimensions=16, walk_length=10, num_walks=100, workers=4)
    model = node2vec.fit(window=5, min_count=1)
    
    # Embedding Extraction of Node
    gnn_embedded = [model.wv[node] for node in graph.nodes() if node in model.wv]
    embedded_features.append(np.array(gnn_embedded))  # Add Embedding

# 4. Manual Feature Engineering (tls_cs)
if "tls_cs" in data.columns:
    data["tls_cs_len"] = data["tls_cs"].apply(lambda x: len(str(x)))  # Example feature
    data["tls_cs_unique"] = data["tls_cs"].apply(lambda x: len(set(str(x))))
    embedded_features.append(data[["tls_cs_len", "tls_cs_unique"]].fillna(0).values)

# 5. Protocol-Specific Parsing
if "dns_query_name" in data.columns:
    data["dns_tld"] = data["dns_query_name"].apply(lambda x: x.split('.')[-1] if isinstance(x, str) else x)
    vectorizer = TfidfVectorizer()
    dns_query_embedded = vectorizer.fit_transform(data["dns_query_name"].fillna(""))
    embedded_features.append(dns_query_embedded.toarray())

if "tls_ext_types" in data.columns:
    data["tls_ext_count"] = data["tls_ext_types"].apply(lambda x: len(eval(x)) if isinstance(x, str) else 0)
    data["tls_ext_mean"] = data["tls_ext_types"].apply(lambda x: np.mean(eval(x)) if isinstance(x, str) and len(eval(x)) > 0 else 0)
    embedded_features.append(data[["tls_ext_count", "tls_ext_mean"]].fillna(0).values)

# 6. Sequence Autoencoders for Remaining Strings
string_features = [col for col in data.columns if data[col].dtype == object]
for col in string_features:
    try:
        embedded_features.append(embed_sequence_autoencoder(data, col))
    except Exception as e:
        print(f"Error embedding column {col}: {e}")

# 4. Scaling Numeric Features
def parse_array_column(column):
    return column.apply(lambda x: np.array(eval(x)) if isinstance(x, str) else x)

numeric_columns = data.select_dtypes(include=[np.number]).columns
numeric_data = data[numeric_columns].fillna(0)
scaler = StandardScaler()
scaled_numeric_data = scaler.fit_transform(numeric_data)

# 5. Integrate all features
if embedded_features:
    embedded_features = np.hstack(embedded_features)
    full_data = np.hstack([scaled_numeric_data, embedded_features])
else:
    full_data = scaled_numeric_data

# 6. PCA
pca = PCA(n_components=2)
pca_data = pca.fit_transform(full_data)

# 7. K-Means Clusturing
kmeans = KMeans(n_clusters=2, random_state=42)
labels = kmeans.fit_predict(pca_data)

# 8. Visualize results
plt.scatter(pca_data[:, 0], pca_data[:, 1], c=labels, cmap="viridis", alpha=0.6)
plt.title("PCA + Clustering")
plt.xlabel("PCA Component 1")
plt.ylabel("PCA Component 2")
plt.colorbar(label="Cluster")
plt.show()