from attention import scaled_dot_product_attention
from layernorm import layer_norm

def encoder_block(X, WQ, WK, WV):

    Q = X @ WQ
    K = X @ WK
    V = X @ WV

    Z = scaled_dot_product_attention(Q, K, V)

    X_res = X + Z

    return layer_norm(X_res)