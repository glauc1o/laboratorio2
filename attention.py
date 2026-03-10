import numpy as np

def softmax(x: np.ndarray) -> np.ndarray:

    x_stable = x - np.max(x, axis=1, keepdims=True)
    
    exp_x = np.exp(x_stable)
    sum_exp_x = np.sum(exp_x, axis=1, keepdims=True)
    
    return exp_x / sum_exp_x


def scaled_dot_product_attention(Q: np.ndarray,
                                 K: np.ndarray,
                                 V: np.ndarray) -> np.ndarray:
    d_k = K.shape[1]

    scores = np.matmul(Q, K.T)

    scaled_scores = scores / np.sqrt(d_k)

    attention_weights = softmax(scaled_scores)

    output = np.matmul(attention_weights, V)

    return output