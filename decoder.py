import numpy as np
from attention import softmax

def create_causal_mask(seq_len):
    mask = np.triu(np.ones((seq_len, seq_len)), k=1)
    mask = np.where(mask == 1, -np.inf, 0)
    return mask 

def cross_attention(encoder_out, decoder_state):
    d_model = encoder_out.shape[-1]
    d_k = d_model
    
    WQ = np.random.randn(d_model, d_model)
    WK = np.random.randn(d_model, d_model)
    WV = np.random.randn(d_model, d_model)
    
    Q = decoder_state @ WQ
    K = encoder_out @ WK
    V = encoder_out @ WV
    
    scores = np.matmul(Q, np.swapaxes(K, -1, -2)) / np.sqrt(d_k)
    weights = softmax(scores)
    return np.matmul(weights, V)

def generate_next_token(current_sequence, encoder_out):
    vocab_size = 10000
    probs = np.random.dirichlet(np.ones(vocab_size), size=1).flatten()
    return probs