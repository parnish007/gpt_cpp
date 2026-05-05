#pragma once
#include "tensor.h"
#include "ops.h"
#include "config.h"
#include <vector>
#include <string>

namespace gpt {

// ─────────────────────────────────────────────────────────────────────────────
//  Base layer interface
// ─────────────────────────────────────────────────────────────────────────────
struct Layer {
    bool training = true;
    virtual ~Layer() = default;

    // Returns list of (weight, grad) pairs for optimizer
    virtual std::vector<std::pair<float*, float*>> parameters() { return {}; }
    virtual std::vector<size_t> param_sizes() { return {}; }
    virtual void set_training(bool mode) { training = mode; }
    virtual std::string name() const { return "Layer"; }
};

// ─────────────────────────────────────────────────────────────────────────────
//  Linear  y = xW + b
//  W: (in_dim, out_dim)    b: (out_dim,)
// ─────────────────────────────────────────────────────────────────────────────
struct Linear : Layer {
    size_t in_dim, out_dim;
    bool   use_bias;

    Tensor W, b;         // weights
    Tensor dW, db;       // grads
    Tensor x_cache;      // saved input for backward

    Linear(size_t in_dim, size_t out_dim, bool bias = true);

    // forward: x (B,T,in_dim) → out (B,T,out_dim)
    Tensor forward(const Tensor& x);

    // backward: dout (B,T,out_dim) → dx (B,T,in_dim)
    Tensor backward(const Tensor& dout);

    std::vector<std::pair<float*, float*>> parameters() override;
    std::vector<size_t> param_sizes() override;
    std::string name() const override { return "Linear"; }
};

// ─────────────────────────────────────────────────────────────────────────────
//  LayerNorm
// ─────────────────────────────────────────────────────────────────────────────
struct LayerNorm : Layer {
    size_t dim;
    float  eps;

    Tensor gamma, beta;
    Tensor d_gamma, d_beta;
    // cache
    Tensor x_cache, mean_cache, var_cache, xhat_cache;

    LayerNorm(size_t dim, float eps = 1e-5f);

    Tensor forward(const Tensor& x);
    Tensor backward(const Tensor& dout);

    std::vector<std::pair<float*, float*>> parameters() override;
    std::vector<size_t> param_sizes() override;
    std::string name() const override { return "LayerNorm"; }
};

// ─────────────────────────────────────────────────────────────────────────────
//  Embedding  (token + positional)
// ─────────────────────────────────────────────────────────────────────────────
struct Embedding : Layer {
    size_t vocab_size, embed_dim, max_seq_len;

    Tensor token_emb;     // (vocab_size, embed_dim)
    Tensor pos_emb;       // (max_seq_len, embed_dim)
    Tensor d_token_emb, d_pos_emb;

    // cache
    std::vector<int> idx_cache;
    size_t B_cache{0}, T_cache{0};

    Embedding(size_t vocab_size, size_t embed_dim, size_t max_seq_len);

    // idx: flat (B*T) int array   → out: (B, T, embed_dim)
    Tensor forward(const int* idx, size_t B, size_t T);
    void   backward(const Tensor& dout);

    std::vector<std::pair<float*, float*>> parameters() override;
    std::vector<size_t> param_sizes() override;
    std::string name() const override { return "Embedding"; }
};

// ─────────────────────────────────────────────────────────────────────────────
//  Dropout
// ─────────────────────────────────────────────────────────────────────────────
struct Dropout : Layer {
    float              p;
    std::vector<float> mask;

    Dropout(float p = 0.1f) : p(p) {}

    Tensor forward(const Tensor& x);
    Tensor backward(const Tensor& dout);
    std::string name() const override { return "Dropout"; }
};

// ─────────────────────────────────────────────────────────────────────────────
//  MultiHeadAttention
// ─────────────────────────────────────────────────────────────────────────────
struct MultiHeadAttention : Layer {
    size_t embed_dim, n_heads, head_dim;

    Linear qkv_proj;      // (embed_dim → 3*embed_dim)
    Linear out_proj;      // (embed_dim → embed_dim)
    Dropout attn_drop, resid_drop;

    // cache for backward
    Tensor Q_cache, K_cache, V_cache;
    Tensor attn_cache;          // post-softmax attention weights
    Tensor attn_drop_cache;
    Tensor x_in_cache;
    size_t B_cache{0}, T_cache{0};

    MultiHeadAttention(size_t embed_dim, size_t n_heads, float dropout = 0.1f);

    Tensor forward(const Tensor& x);
    Tensor backward(const Tensor& dout);

    std::vector<std::pair<float*, float*>> parameters() override;
    std::vector<size_t> param_sizes() override;
    void set_training(bool mode) override;
    std::string name() const override { return "MultiHeadAttention"; }
};

// ─────────────────────────────────────────────────────────────────────────────
//  FeedForward  (2-layer MLP with GELU)
// ─────────────────────────────────────────────────────────────────────────────
struct FeedForward : Layer {
    Linear  fc1, fc2;
    Dropout drop;
    Tensor  gelu_cache;   // pre-activation cache

    FeedForward(size_t embed_dim, float dropout = 0.1f);

    Tensor forward(const Tensor& x);
    Tensor backward(const Tensor& dout);

    std::vector<std::pair<float*, float*>> parameters() override;
    std::vector<size_t> param_sizes() override;
    void set_training(bool mode) override;
    std::string name() const override { return "FeedForward"; }
};

// ─────────────────────────────────────────────────────────────────────────────
//  TransformerBlock  =  LayerNorm + Attention + LayerNorm + FFN
// ─────────────────────────────────────────────────────────────────────────────
struct TransformerBlock : Layer {
    LayerNorm        ln1, ln2;
    MultiHeadAttention attn;
    FeedForward      ffn;

    Tensor x_in_cache, x_mid_cache;

    TransformerBlock(size_t embed_dim, size_t n_heads, float dropout = 0.1f);

    Tensor forward(const Tensor& x);
    Tensor backward(const Tensor& dout);

    std::vector<std::pair<float*, float*>> parameters() override;
    std::vector<size_t> param_sizes() override;
    void set_training(bool mode) override;
    std::string name() const override { return "TransformerBlock"; }
};

} // namespace gpt
