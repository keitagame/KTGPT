import numpy as np

# 語彙数と埋め込み次元
vocab_size = 10000
embedding_dim = 64

# ランダムな埋め込み行列を作成
embedding_matrix = np.random.randn(vocab_size, embedding_dim)

# 例：トークンIDのシーケンス
token_ids = np.array([12, 456, 789, 22])

# ベクトル化された入力（埋め込み）
embedded = embedding_matrix[token_ids]
def self_attention(x):
    d_k = x.shape[-1]
    
    # クエリ・キー・バリューの重み行列（同じ次元で簡略化）
    W_q = np.random.randn(d_k, d_k)
    W_k = np.random.randn(d_k, d_k)
    W_v = np.random.randn(d_k, d_k)
    
    Q = x @ W_q
    K = x @ W_k
    V = x @ W_v
    
    scores = Q @ K.T / np.sqrt(d_k)
    attention_weights = softmax(scores)
    output = attention_weights @ V
    return output

def softmax(x):
    e_x = np.exp(x - np.max(x, axis=-1, keepdims=True))
    return e_x / np.sum(e_x, axis=-1, keepdims=True)
def feed_forward(x, hidden_dim):
    input_dim = x.shape[-1]
    
    # 重みとバイアスの初期化（シンプルにランダム）
    W1 = np.random.randn(input_dim, hidden_dim)
    b1 = np.random.randn(hidden_dim)
    W2 = np.random.randn(hidden_dim, input_dim)
    b2 = np.random.randn(input_dim)

    # 活性化関数：ReLU
    def relu(x):
        return np.maximum(0, x)

    # 順伝播
    hidden = relu(x @ W1 + b1)
    output = hidden @ W2 + b2
    return output

# セルフアテンションを適用
output = self_attention(embedded)
def layer_norm_with_params(x, gamma, beta, eps=1e-5):
    mean = x.mean(axis=-1, keepdims=True)
    std = x.std(axis=-1, keepdims=True)
    return gamma * (x - mean) / (std + eps) + beta
# セルフアテンションの出力
attn_out = self_attention(embedded)

# 残差とLayerNorm（パラメータなしの場合）
residual1 = embedded + attn_out


gamma = np.ones_like(residual1)
beta = np.zeros_like(residual1)
normed1 = layer_norm_with_params(residual1, gamma, beta)
# FFNを通す
ffn_out = feed_forward(normed1, hidden_dim=256)

# 残差接続 + LayerNorm（第2段）
residual2 = normed1 + ffn_out
normed2 = layer_norm_with_params(residual2,gamma,beta)
