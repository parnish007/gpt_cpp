#include "core/layers.h"
#include <stdexcept>
#include <cstring>
#include <random>
#include <cmath>
#include <algorithm>

namespace gpt {

// ─────────────────────────────────────────────────────────────────────────────
//  Linear
// ─────────────────────────────────────────────────────────────────────────────
Linear::Linear(size_t in_dim_, size_t out_dim_, bool bias)
    : in_dim(in_dim_), out_dim(out_dim_), use_bias(bias),
      W({in_dim_, out_dim_}), b({out_dim_}),
      dW({in_dim_, out_dim_}), db({out_dim_})
{
    W.xavier_uniform(in_dim_, out_dim_);
    b.fill(0.0f);
    dW.fill(0.0f); db.fill(0.0f);
}

Tensor Linear::forward(const Tensor& x) {
    // x: (B, T, in_dim)  →  out: (B, T, out_dim)
    size_t B  = x.dim(0), T = x.dim(1);
    x_cache   = x;
    Tensor out({B, T, out_dim});
    // batched matmul: for each (b,t) row, multiply by W
    size_t M = B * T;
    ops::matmul(x.raw(), W.raw(), out.raw(), M, in_dim, out_dim);
    if (use_bias) {
        for (size_t i = 0; i < M; ++i)
            ops::add_inplace(out.raw() + i * out_dim, b.raw(), out_dim);
    }
    return out;
}

Tensor Linear::backward(const Tensor& dout) {
    size_t B = x_cache.dim(0), T = x_cache.dim(1);
    size_t M = B * T;

    // dx = dout @ W^T
    Tensor dx({B, T, in_dim});
    std::vector<float> W_T(in_dim * out_dim);
    ops::transpose2d(W.raw(), W_T.data(), in_dim, out_dim);
    ops::matmul(dout.raw(), W_T.data(), dx.raw(), M, out_dim, in_dim);

    // dW += x^T @ dout
    std::vector<float> x_T(in_dim * M);
    ops::transpose2d(x_cache.raw(), x_T.data(), M, in_dim);
    ops::matmul(x_T.data(), dout.raw(), dW.raw(), in_dim, M, out_dim, /*accumulate=*/true);

    // db += sum over (B,T) dim
    if (use_bias) {
        for (size_t i = 0; i < M; ++i)
            ops::add_inplace(db.raw(), dout.raw() + i * out_dim, out_dim);
    }
    return dx;
}

std::vector<std::pair<float*, float*>> Linear::parameters() {
    std::vector<std::pair<float*, float*>> p;
    p.push_back({W.raw(), dW.raw()});
    if (use_bias) p.push_back({b.raw(), db.raw()});
    return p;
}

std::vector<size_t> Linear::param_sizes() {
    std::vector<size_t> s;
    s.push_back(in_dim * out_dim);
    if (use_bias) s.push_back(out_dim);
    return s;
}

// ─────────────────────────────────────────────────────────────────────────────
//  LayerNorm
// ─────────────────────────────────────────────────────────────────────────────
LayerNorm::LayerNorm(size_t dim_, float eps_)
    : dim(dim_), eps(eps_),
      gamma({dim_}), beta({dim_}),
      d_gamma({dim_}), d_beta({dim_})
{
    gamma.fill(1.0f);
    beta.fill(0.0f);
    d_gamma.fill(0.0f); d_beta.fill(0.0f);
}

Tensor LayerNorm::forward(const Tensor& x) {
    size_t B = x.dim(0), T = x.dim(1);
    x_cache   = x;
    mean_cache  = Tensor({B, T});
    var_cache   = Tensor({B, T});
    xhat_cache  = Tensor({B, T, dim});
    Tensor out({B, T, dim});

    for (size_t b = 0; b < B; ++b) {
        for (size_t t = 0; t < T; ++t) {
            const float* xrow = x.raw() + (b * T + t) * dim;
            float* orow       = out.raw() + (b * T + t) * dim;
            float* xhat_row   = xhat_cache.raw() + (b * T + t) * dim;
            float mean = 0.0f, var = 0.0f;
            for (size_t i = 0; i < dim; ++i) mean += xrow[i];
            mean /= (float)dim;
            for (size_t i = 0; i < dim; ++i) { float d = xrow[i]-mean; var += d*d; }
            var /= (float)dim;
            float inv_std = 1.0f / std::sqrt(var + eps);
            mean_cache.at(b * T + t) = mean;
            var_cache.at(b * T + t)  = var;
            for (size_t i = 0; i < dim; ++i) {
                xhat_row[i] = (xrow[i] - mean) * inv_std;
                orow[i]     = gamma.at(i) * xhat_row[i] + beta.at(i);
            }
        }
    }
    return out;
}

Tensor LayerNorm::backward(const Tensor& dout) {
    size_t B = x_cache.dim(0), T = x_cache.dim(1);
    Tensor dx({B, T, dim});

    for (size_t b = 0; b < B; ++b) {
        for (size_t t = 0; t < T; ++t) {
            size_t idx        = b * T + t;
            const float* drow = dout.raw() + idx * dim;
            const float* xrow = x_cache.raw() + idx * dim;
            const float* xhat = xhat_cache.raw() + idx * dim;
            float* dxrow      = dx.raw() + idx * dim;
            float  var = var_cache.at(idx);
            float  inv_std = 1.0f / std::sqrt(var + eps);
            float  N = (float)dim;

            // dxhat = dout * gamma
            std::vector<float> dxhat(dim);
            for (size_t i = 0; i < dim; ++i) {
                dxhat[i] = drow[i] * gamma.at(i);
                d_gamma.raw()[i] += drow[i] * xhat[i];
                d_beta.raw()[i]  += drow[i];
            }
            // dx = (1/N/std) * (N*dxhat - sum(dxhat) - xhat*sum(dxhat*xhat))
            float s1 = 0.0f, s2 = 0.0f;
            for (size_t i = 0; i < dim; ++i) { s1 += dxhat[i]; s2 += dxhat[i]*xhat[i]; }
            for (size_t i = 0; i < dim; ++i)
                dxrow[i] = inv_std / N * (N * dxhat[i] - s1 - xhat[i] * s2);
        }
    }
    return dx;
}

std::vector<std::pair<float*, float*>> LayerNorm::parameters() {
    return {{gamma.raw(), d_gamma.raw()}, {beta.raw(), d_beta.raw()}};
}

std::vector<size_t> LayerNorm::param_sizes() { return {dim, dim}; }

// ─────────────────────────────────────────────────────────────────────────────
//  Embedding
// ─────────────────────────────────────────────────────────────────────────────
Embedding::Embedding(size_t vs, size_t ed, size_t msl)
    : vocab_size(vs), embed_dim(ed), max_seq_len(msl),
      token_emb({vs, ed}), pos_emb({msl, ed}),
      d_token_emb({vs, ed}), d_pos_emb({msl, ed})
{
    token_emb.randn(0.0f, 0.02f);
    pos_emb.randn(0.0f, 0.02f);
    d_token_emb.fill(0.0f);
    d_pos_emb.fill(0.0f);
}

Tensor Embedding::forward(const int* idx, size_t B, size_t T) {
    idx_cache.assign(idx, idx + B * T);
    B_cache = B; T_cache = T;
    Tensor out({B, T, embed_dim});
    for (size_t b = 0; b < B; ++b) {
        for (size_t t = 0; t < T; ++t) {
            int tok = idx[b * T + t];
            float* dst = out.raw() + (b * T + t) * embed_dim;
            const float* tok_src = token_emb.raw() + tok * embed_dim;
            const float* pos_src = pos_emb.raw()   + t   * embed_dim;
            for (size_t e = 0; e < embed_dim; ++e)
                dst[e] = tok_src[e] + pos_src[e];
        }
    }
    return out;
}

void Embedding::backward(const Tensor& dout) {
    size_t B = B_cache, T = T_cache;
    for (size_t b = 0; b < B; ++b) {
        for (size_t t = 0; t < T; ++t) {
            int tok = idx_cache[b * T + t];
            const float* d = dout.raw() + (b * T + t) * embed_dim;
            float* dtok    = d_token_emb.raw() + tok * embed_dim;
            float* dpos    = d_pos_emb.raw()   + t   * embed_dim;
            for (size_t e = 0; e < embed_dim; ++e) {
                dtok[e] += d[e];
                dpos[e] += d[e];
            }
        }
    }
}

std::vector<std::pair<float*, float*>> Embedding::parameters() {
    return {{token_emb.raw(), d_token_emb.raw()},
            {pos_emb.raw(),   d_pos_emb.raw()}};
}

std::vector<size_t> Embedding::param_sizes() {
    return {vocab_size * embed_dim, max_seq_len * embed_dim};
}

// ─────────────────────────────────────────────────────────────────────────────
//  Dropout
// ─────────────────────────────────────────────────────────────────────────────
Tensor Dropout::forward(const Tensor& x) {
    if (!training || p == 0.0f) return x;
    Tensor out = x;
    mask.resize(x.numel());
    std::mt19937 rng(std::random_device{}());
    std::bernoulli_distribution dist(1.0f - p);
    float scale = 1.0f / (1.0f - p);
    for (size_t i = 0; i < x.numel(); ++i) {
        mask[i]      = dist(rng) ? scale : 0.0f;
        out.raw()[i] = x.at(i) * mask[i];
    }
    return out;
}

Tensor Dropout::backward(const Tensor& dout) {
    if (!training || p == 0.0f) return dout;
    Tensor dx = dout;
    for (size_t i = 0; i < dout.numel(); ++i)
        dx.raw()[i] = dout.at(i) * mask[i];
    return dx;
}

} // namespace gpt
