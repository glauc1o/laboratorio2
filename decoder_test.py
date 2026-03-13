import numpy as np
from decoder import create_causal_mask, cross_attention, generate_next_token
from attention import softmax

#Teste para a função do create_causal_mask
seq_len = 5
d_k = 64
Q = np.random.randn(seq_len, d_k)
K = np.random.randn(seq_len, d_k)
M = create_causal_mask(seq_len)

scores = np.matmul(Q, K.T) / np.sqrt(d_k)
masked_scores = scores + M
attention_weights = softmax(masked_scores)

print("Matriz de Atenção com Máscara Causal:")
print(attention_weights)

#Teste para a função de cross_attention
encoder_output = np.random.randn(1, 10, 512)
decoder_state = np.random.randn(1, 4, 512)  

output_cross = cross_attention(encoder_output, decoder_state)
print(f"\nShape da Cross-Attention: {output_cross.shape}")

#teste para a função generate_next_token
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