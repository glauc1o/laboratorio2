import numpy as np
from embeddings import create_embedding_table, get_embeddings
from dados import create_vocab, sentence_to_ids
from encoder import encoder_block
from attention import scaled_dot_product_attention

vocab, df = create_vocab()

sentence = "o banco bloqueou cartao"

ids = sentence_to_ids(sentence, vocab)

embedding_table = create_embedding_table(len(vocab))

X = get_embeddings(ids, embedding_table)

d_model = 64

WQ = np.random.randn(d_model, d_model)
WK = np.random.randn(d_model, d_model)
WV = np.random.randn(d_model, d_model)

output = encoder_block(X, WQ, WK, WV)

print("Output:")
print(output.shape)