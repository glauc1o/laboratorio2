import numpy as np

def create_embedding_table(vocab_size, d_model=64):
    return np.random.randn(vocab_size, d_model)

def get_embeddings(ids, embedding_table):
    return embedding_table[ids]

def init_attention_weights(d_model):
    
    WQ = np.random.randn(d_model, d_model)
    WK = np.random.randn(d_model, d_model)
    WV = np.random.randn(d_model, d_model)

    return WQ, WK, WV

def compute_qkv(X, WQ, WK, WV):

    Q = X @ WQ
    K = X @ WK
    V = X @ WV

    return Q, K, V