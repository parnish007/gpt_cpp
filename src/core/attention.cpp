#include "core/layers.h"
#include <cstring>
#include <cmath>
#include <algorithm>
#include <numeric>

namespace gpt {

// ─────────────────────────────────────────────────────────────────────────────
//  MultiHeadAttention
// ─────────────────────────────────────────────────────────────────────────────
MultiHeadAttention::MultiHeadAttention(size_t ed, size_t nh, float dp)
    : embed_dim(ed), n_heads(nh), head_dim(ed / nh),
      qkv_proj(ed, 3 * ed), out_proj(ed, ed),
      attn_drop(dp), resid_drop(dp)
{
    if (ed % nh != 0)
        throw std::runtime_error("embed_dim must be divisible by n_heads");
}

Tensor MultiHeadAttention::forward(const Tensor& x) {
    size_t B = x.dim(0), T = x.dim(1), C = embed_dim;
    size_t H = n_heads, D = head_dim;
    B_cache = B; T_cache = T;
    x_in_cache = x;

    // QKV projection: (B, T, 3C)
    Tensor qkv = qkv_proj.forward(x);

    // Split into Q, K, V each (B, H, T, D)
    Q_cache = Tensor({B, H, T, D});
    K_cache = Tensor({B, H, T, D});
    V_cache = Tensor({B, H, T, D});

    const float* qkv_ptr = qkv.raw();
    for (size_t b = 0; b < B; ++b) {
        for (size_t t = 0; t < T; ++t) {
            const float* row = qkv_ptr + (b * T + t) * 3 * C;
            for (size_t h = 0; h < H; ++h) {
                float* Qrow = Q_cache.raw() + (b*H*T + h*T + t) * D;
                float* Krow = K_cache.raw() + (b*H*T + h*T + t) * D;
                float* Vrow = V_cache.raw() + (b*H*T + h*T + t) * D;
                for (size_t d = 0; d < D; ++d) {
                    Qrow[d] = row[h * D + d];
                    Krow[d] = row[C + h * D + d];
                    Vrow[d] = row[2*C + h * D + d];
                }
            }
        }
    }

    // Scaled dot-product attention: (B, H, T, T)
    float scale = 1.0f / std::sqrt((float)D);
    Tensor attn_scores({B, H, T, T});

    for (size_t b = 0; b < B; ++b) {
        for (size_t h = 0; h < H; ++h) {
            for (size_t i = 0; i < T; ++i) {
                const float* Qi = Q_cache.raw() + (b*H*T + h*T + i) * D;
                for (size_t j = 0; j < T; ++j) {
                    float val = 0.0f;
                    if (j <= i) { // causal mask
                        const float* Kj = K_cache.raw() + (b*H*T + h*T + j) * D;
                        for (size_t d = 0; d < D; ++d) val += Qi[d] * Kj[d];
                        val *= scale;
                    } else {
                        val = -1e9f;
                    }
                    attn_scores.raw()[(b*H + h)*T*T + i*T + j] = val;
                }
                // softmax over T
                ops::softmax_inplace(
                    attn_scores.raw() + (b*H + h)*T*T + i*T, T);
            }
        }
    }

    attn_cache      = attn_drop.forward(attn_scores);
    attn_drop_cache = attn_cache;

    // Weighted sum of V: (B, H, T, D)
    Tensor ctx({B, H, T, D});
    for (size_t b = 0; b < B; ++b) {
        for (size_t h = 0; h < H; ++h) {
            for (size_t i = 0; i < T; ++i) {
                float* out_row = ctx.raw() + (b*H*T + h*T + i) * D;
                const float* a_row = attn_cache.raw() + (b*H + h)*T*T + i*T;
                std::fill(out_row, out_row + D, 0.0f);
                for (size_t j = 0; j < T; ++j) {
                    const float* Vj = V_cache.raw() + (b*H*T + h*T + j) * D;
                    for (size_t d = 0; d < D; ++d)
                        out_row[d] += a_row[j] * Vj[d];
                }
            }
        }
    }

    // Reshape ctx (B, H, T, D) → (B, T, C)
    Tensor ctx_merged({B, T, C});
    for (size_t b = 0; b < B; ++b) {
        for (size_t t = 0; t < T; ++t) {
            for (size_t h = 0; h < H; ++h) {
                const float* src = ctx.raw() + (b*H*T + h*T + t) * D;
                float* dst = ctx_merged.raw() + (b*T + t)*C + h*D;
                std::copy(src, src + D, dst);
            }
        }
    }

    return resid_drop.forward(out_proj.forward(ctx_merged));
}

Tensor MultiHeadAttention::backward(const Tensor& dout) {
    size_t B = B_cache, T = T_cache, C = embed_dim;
    size_t H = n_heads, D = head_dim;

    // backward: resid_drop → out_proj
    Tensor d = resid_drop.backward(dout);
    d = out_proj.backward(d);   // (B, T, C)

    // Rearrange (B, T, C) → (B, H, T, D)
    Tensor d_ctx({B, H, T, D});
    for (size_t b = 0; b < B; ++b)
        for (size_t t = 0; t < T; ++t)
            for (size_t h = 0; h < H; ++h)
                for (size_t dd = 0; dd < D; ++dd)
                    d_ctx.raw()[(b*H*T + h*T + t)*D + dd] =
                        d.raw()[(b*T + t)*C + h*D + dd];

    // dV and d_attn_drop
    Tensor dV({B, H, T, D}); dV.fill(0.0f);
    Tensor d_attn_scores({B, H, T, T}); d_attn_scores.fill(0.0f);

    for (size_t b = 0; b < B; ++b) {
        for (size_t h = 0; h < H; ++h) {
            const float* a = attn_cache.raw() + (b*H+h)*T*T;
            for (size_t i = 0; i < T; ++i) {
                const float* d_ctx_i = d_ctx.raw() + (b*H*T+h*T+i)*D;
                // dV[j] += a[i,j] * d_ctx[i]
                for (size_t j = 0; j < T; ++j) {
                    float* dVj = dV.raw() + (b*H*T+h*T+j)*D;
                    for (size_t dd = 0; dd < D; ++dd)
                        dVj[dd] += a[i*T+j] * d_ctx_i[dd];
                }
                // d_attn[i,j] += d_ctx[i] · V[j]
                for (size_t j = 0; j < T; ++j) {
                    const float* Vj = V_cache.raw() + (b*H*T+h*T+j)*D;
                    for (size_t dd = 0; dd < D; ++dd)
                        d_attn_scores.raw()[(b*H+h)*T*T+i*T+j] +=
                            d_ctx_i[dd] * Vj[dd];
                }
            }
        }
    }

    // backward attn_drop
    Tensor d_attn = attn_drop.backward(d_attn_scores);

    // backward softmax (causal)
    float scale = 1.0f / std::sqrt((float)D);
    Tensor dQ({B, H, T, D}); dQ.fill(0.0f);
    Tensor dK({B, H, T, D}); dK.fill(0.0f);

    for (size_t b = 0; b < B; ++b) {
        for (size_t h = 0; h < H; ++h) {
            float* a_row_base  = attn_cache.raw() + (b*H+h)*T*T;
            float* da_row_base = d_attn.raw()     + (b*H+h)*T*T;
            for (size_t i = 0; i < T; ++i) {
                float* s  = a_row_base  + i*T;
                float* ds = da_row_base + i*T;
                // softmax backward: ds_i = s_i * (ds_i - sum_j(s_j * ds_j))
                float dot = 0.0f;
                for (size_t j = 0; j <= i; ++j) dot += s[j] * ds[j];
                for (size_t j = 0; j <= i; ++j) ds[j] = s[j] * (ds[j] - dot);
                // scale
                for (size_t j = 0; j <= i; ++j) ds[j] *= scale;

                // dQ[i] += ds @ K
                float* dQi = dQ.raw() + (b*H*T+h*T+i)*D;
                for (size_t j = 0; j <= i; ++j) {
                    const float* Kj = K_cache.raw() + (b*H*T+h*T+j)*D;
                    for (size_t dd = 0; dd < D; ++dd)
                        dQi[dd] += ds[j] * Kj[dd];
                }
                // dK[j] += ds[i,j] * Q[i]
                const float* Qi = Q_cache.raw() + (b*H*T+h*T+i)*D;
                for (size_t j = 0; j <= i; ++j) {
                    float* dKj = dK.raw() + (b*H*T+h*T+j)*D;
                    for (size_t dd = 0; dd < D; ++dd)
                        dKj[dd] += ds[j] * Qi[dd];
                }
            }
        }
    }

    // Pack dQ, dK, dV → dqkv (B, T, 3C)
    Tensor dqkv({B, T, 3*C}); dqkv.fill(0.0f);
    for (size_t b = 0; b < B; ++b) {
        for (size_t t = 0; t < T; ++t) {
            float* row = dqkv.raw() + (b*T+t)*3*C;
            for (size_t h = 0; h < H; ++h) {
                const float* dQrow = dQ.raw() + (b*H*T+h*T+t)*D;
                const float* dKrow = dK.raw() + (b*H*T+h*T+t)*D;
                const float* dVrow = dV.raw() + (b*H*T+h*T+t)*D;
                for (size_t dd = 0; dd < D; ++dd) {
                    row[h*D + dd]     += dQrow[dd];
                    row[C + h*D + dd] += dKrow[dd];
                    row[2*C+h*D + dd] += dVrow[dd];
                }
            }
        }
    }

    return qkv_proj.backward(dqkv);
}

std::vector<std::pair<float*, float*>> MultiHeadAttention::parameters() {
    auto p = qkv_proj.parameters();
    auto p2 = out_proj.parameters();
    p.insert(p.end(), p2.begin(), p2.end());
    return p;
}
std::vector<size_t> MultiHeadAttention::param_sizes() {
    auto s = qkv_proj.param_sizes(); auto s2 = out_proj.param_sizes();
    s.insert(s.end(), s2.begin(), s2.end()); return s;
}
void MultiHeadAttention::set_training(bool mode) {
    training = mode;
    attn_drop.set_training(mode);
    resid_drop.set_training(mode);
}

// ─────────────────────────────────────────────────────────────────────────────
//  FeedForward
// ─────────────────────────────────────────────────────────────────────────────
FeedForward::FeedForward(size_t ed, float dp)
    : fc1(ed, 4*ed), fc2(4*ed, ed), drop(dp) {}

Tensor FeedForward::forward(const Tensor& x) {
    Tensor h = fc1.forward(x);
    gelu_cache = h;
    // apply GELU in-place
    for (size_t i = 0; i < h.numel(); ++i) h.raw()[i] = ops::gelu(h.raw()[i]);
    return drop.forward(fc2.forward(h));
}

Tensor FeedForward::backward(const Tensor& dout) {
    Tensor d = fc2.backward(drop.backward(dout));
    for (size_t i = 0; i < d.numel(); ++i)
        d.raw()[i] *= ops::gelu_grad(gelu_cache.raw()[i]);
    return fc1.backward(d);
}

std::vector<std::pair<float*, float*>> FeedForward::parameters() {
    auto p = fc1.parameters(); auto p2 = fc2.parameters();
    p.insert(p.end(), p2.begin(), p2.end()); return p;
}
std::vector<size_t> FeedForward::param_sizes() {
    auto s = fc1.param_sizes(); auto s2 = fc2.param_sizes();
    s.insert(s.end(), s2.begin(), s2.end()); return s;
}
void FeedForward::set_training(bool mode) {
    training = mode; drop.set_training(mode);
}

// ─────────────────────────────────────────────────────────────────────────────
//  TransformerBlock
// ─────────────────────────────────────────────────────────────────────────────
TransformerBlock::TransformerBlock(size_t ed, size_t nh, float dp)
    : ln1(ed), ln2(ed), attn(ed, nh, dp), ffn(ed, dp) {}

Tensor TransformerBlock::forward(const Tensor& x) {
    x_in_cache  = x;
    Tensor x1   = ln1.forward(x);
    Tensor a    = attn.forward(x1);
    // residual 1
    x_mid_cache = Tensor({x.dim(0), x.dim(1), x.dim(2)});
    for (size_t i = 0; i < x.numel(); ++i)
        x_mid_cache.raw()[i] = x.at(i) + a.at(i);

    Tensor x2 = ln2.forward(x_mid_cache);
    Tensor f  = ffn.forward(x2);
    // residual 2
    Tensor out(x_mid_cache.shape);
    for (size_t i = 0; i < out.numel(); ++i)
        out.raw()[i] = x_mid_cache.at(i) + f.at(i);
    return out;
}

Tensor TransformerBlock::backward(const Tensor& dout) {
    // backward FFN sub-layer
    Tensor df    = ffn.backward(dout);
    Tensor d_ln2 = ln2.backward(df);
    // residual: d_mid = dout + d_ln2
    Tensor d_mid(dout.shape);
    for (size_t i = 0; i < dout.numel(); ++i)
        d_mid.raw()[i] = dout.at(i) + d_ln2.at(i);

    // backward Attention sub-layer
    Tensor da    = attn.backward(d_mid);
    Tensor d_ln1 = ln1.backward(da);
    // residual: dx = d_mid + d_ln1
    Tensor dx(d_mid.shape);
    for (size_t i = 0; i < d_mid.numel(); ++i)
        dx.raw()[i] = d_mid.at(i) + d_ln1.at(i);
    return dx;
}

std::vector<std::pair<float*, float*>> TransformerBlock::parameters() {
    auto p = ln1.parameters();
    auto a = attn.parameters(); p.insert(p.end(), a.begin(), a.end());
    auto l = ln2.parameters();  p.insert(p.end(), l.begin(), l.end());
    auto f = ffn.parameters();  p.insert(p.end(), f.begin(), f.end());
    return p;
}
std::vector<size_t> TransformerBlock::param_sizes() {
    auto s = ln1.param_sizes();
    auto a = attn.param_sizes(); s.insert(s.end(), a.begin(), a.end());
    auto l = ln2.param_sizes();  s.insert(s.end(), l.begin(), l.end());
    auto f = ffn.param_sizes();  s.insert(s.end(), f.begin(), f.end());
    return s;
}
void TransformerBlock::set_training(bool mode) {
    training = mode; attn.set_training(mode); ffn.set_training(mode);
}

} // namespace gpt
