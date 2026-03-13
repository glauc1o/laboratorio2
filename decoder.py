import numpy as np
from attention import softmax

def create_causal_mask(seq_len):
    mask = np.triu(np.ones((seq_len, seq_len)), k=1)
    mask = np.where(mask == 1, -np.inf, 0)
    return mask

seq_len = 4
d_k = 64
Q = np.random.randn(seq_len, d_k)
K = np.random.randn(seq_len, d_k)
M = create_causal_mask(seq_len)

scores = np.matmul(Q, K.T) / np.sqrt(d_k)
masked_scores = scores + M
attention_weights = softmax(masked_scores)

print("Matriz de Atenção com Máscara Causal:")
print(attention_weights) 

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

encoder_output = np.random.randn(1, 10, 512)
decoder_state = np.random.randn(1, 4, 512)  

output_cross = cross_attention(encoder_output, decoder_state)
print(f"\nShape da Cross-Attention: {output_cross.shape}")

def generate_next_token(current_sequence, encoder_out):
    vocab_size = 10000
    probs = np.random.dirichlet(np.ones(vocab_size), size=1).flatten()
    return probs

vocab_mock = {i: f"palavra_{i}" for i in range(10000)}
vocab_mock[999] = "<EOS>" 

contexto = ["<START>", "O", "rato"]
enc_out_ficticio = np.random.randn(1, 10, 512)

print("\n--- Iniciando Loop de Inferência ---")
while True:
    probs = generate_next_token(contexto, enc_out_ficticio)
    
    next_token_id = np.argmax(probs)
    proxima_palavra = vocab_mock[next_token_id]
    
    contexto.append(proxima_palavra)
    print(f"Gerado: {proxima_palavra}")
    
    if proxima_palavra == "<EOS>" or len(contexto) > 10:
        break

print(f"Frase final: {' '.join(contexto)}")