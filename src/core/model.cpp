#include "core/model.h"
#include <fstream>
#include <stdexcept>
#include <iostream>
#include <cstring>
#include <cmath>
#include <numeric>
#include <algorithm>

namespace gpt {

GPTModel::GPTModel(const ModelConfig& c)
    : cfg(c),
      embed(c.vocab_size, c.embed_dim, c.max_seq_len),
      drop(c.dropout),
      ln_final(c.embed_dim, c.eps),
      lm_head(c.embed_dim, c.vocab_size, /*bias=*/false)
{
    blocks.reserve(c.n_layers);
    for (size_t i = 0; i < c.n_layers; ++i)
        blocks.emplace_back(c.embed_dim, c.n_heads, c.dropout);
    tie_weights();
}

void GPTModel::tie_weights() {
    // Share token embedding weight with lm_head weight (classic GPT trick)
    // They point to the same underlying data — gradients must be merged
    std::copy(embed.token_emb.raw(),
              embed.token_emb.raw() + embed.token_emb.numel(),
              lm_head.W.raw());
}

// ─────────────────────────────────────────────────────────────────────────────
//  Forward
// ─────────────────────────────────────────────────────────────────────────────
Tensor GPTModel::forward(const int* idx, size_t B, size_t T) {
    Tensor x = embed.forward(idx, B, T);
    x = drop.forward(x);
    for (auto& block : blocks) x = block.forward(x);
    x = ln_final.forward(x);
    logits_cache = lm_head.forward(x);
    return logits_cache;
}

// ─────────────────────────────────────────────────────────────────────────────
//  Loss  (cross-entropy, numerically stable)
// ─────────────────────────────────────────────────────────────────────────────
float GPTModel::loss(const Tensor& logits, const int* targets, size_t BT) {
    size_t V = cfg.vocab_size;
    BT_cache = BT;
    targets_cache.assign(targets, targets + BT);

    probs_cache = Tensor({BT, V});
    float total_loss = 0.0f;

    for (size_t i = 0; i < BT; ++i) {
        const float* row = logits.raw() + i * V;
        float* p         = probs_cache.raw() + i * V;
        // stable softmax
        float mx = *std::max_element(row, row + V);
        float sum = 0.0f;
        for (size_t v = 0; v < V; ++v) { p[v] = std::exp(row[v] - mx); sum += p[v]; }
        for (size_t v = 0; v < V; ++v) p[v] /= sum;
        total_loss += -std::log(p[targets[i]] + 1e-8f);
    }
    return total_loss / (float)BT;
}

// ─────────────────────────────────────────────────────────────────────────────
//  Backward
// ─────────────────────────────────────────────────────────────────────────────
void GPTModel::backward() {
    size_t BT = BT_cache, V = cfg.vocab_size;
    size_t B  = logits_cache.dim(0), T = logits_cache.dim(1);

    // d_logits = (probs - one_hot) / BT
    Tensor d_logits({B, T, V});
    std::copy(probs_cache.raw(), probs_cache.raw() + BT * V, d_logits.raw());
    for (size_t i = 0; i < BT; ++i)
        d_logits.raw()[i * V + targets_cache[i]] -= 1.0f;
    ops::scale_inplace(d_logits.raw(), 1.0f / (float)BT, BT * V);

    // backward: lm_head → ln_final → blocks → drop → embed
    Tensor dx = lm_head.backward(d_logits);
    dx = ln_final.backward(dx);
    for (int i = (int)blocks.size() - 1; i >= 0; --i)
        dx = blocks[i].backward(dx);
    dx = drop.backward(dx);
    embed.backward(dx);

    // Merge lm_head.dW into d_token_emb (weight tying)
    ops::add_inplace(embed.d_token_emb.raw(), lm_head.dW.raw(),
                     cfg.vocab_size * cfg.embed_dim);
}

// ─────────────────────────────────────────────────────────────────────────────
//  Parameters
// ─────────────────────────────────────────────────────────────────────────────
std::vector<std::pair<float*, float*>> GPTModel::parameters() {
    std::vector<std::pair<float*, float*>> p;
    auto add = [&](auto& layer) {
        auto lp = layer.parameters();
        p.insert(p.end(), lp.begin(), lp.end());
    };
    add(embed);
    for (auto& b : blocks) add(b);
    add(ln_final);
    // lm_head shares W with embed — only add bias if present
    if (lm_head.use_bias) p.push_back({lm_head.b.raw(), lm_head.db.raw()});
    return p;
}

std::vector<size_t> GPTModel::param_sizes() {
    std::vector<size_t> s;
    auto add = [&](auto& layer) {
        auto ls = layer.param_sizes();
        s.insert(s.end(), ls.begin(), ls.end());
    };
    add(embed);
    for (auto& b : blocks) add(b);
    add(ln_final);
    return s;
}

size_t GPTModel::num_params() {
    auto sizes = param_sizes();
    return std::accumulate(sizes.begin(), sizes.end(), size_t(0));
}

// ─────────────────────────────────────────────────────────────────────────────
//  Zero gradients
// ─────────────────────────────────────────────────────────────────────────────
void GPTModel::zero_grad() {
    for (auto [w, g] : parameters()) {
        if (g) std::fill(g, g + 1, 0.0f); // placeholder — sizes handled below
    }
    // Proper zero using sizes
    auto params = parameters();
    auto sizes  = param_sizes();
    for (size_t i = 0; i < params.size() && i < sizes.size(); ++i)
        if (params[i].second)
            std::fill(params[i].second, params[i].second + sizes[i], 0.0f);

    // also zero lm_head.dW
    lm_head.dW.fill(0.0f);
    embed.d_token_emb.fill(0.0f);
    embed.d_pos_emb.fill(0.0f);
}

// ─────────────────────────────────────────────────────────────────────────────
//  Train / Eval mode
// ─────────────────────────────────────────────────────────────────────────────
void GPTModel::set_training(bool mode) {
    drop.set_training(mode);
    for (auto& b : blocks) b.set_training(mode);
}

// ─────────────────────────────────────────────────────────────────────────────
//  Serialization  (simple binary format)
// ─────────────────────────────────────────────────────────────────────────────
void GPTModel::save(const std::string& path) const {
    std::ofstream f(path, std::ios::binary);
    if (!f) throw std::runtime_error("Cannot open file for writing: " + path);

    // Write config
    f.write(reinterpret_cast<const char*>(&cfg), sizeof(ModelConfig));

    // Write all weight tensors
    auto write_tensor = [&](const Tensor& t) {
        size_t n = t.numel();
        f.write(reinterpret_cast<const char*>(&n), sizeof(n));
        f.write(reinterpret_cast<const char*>(t.raw()), n * sizeof(float));
    };

    write_tensor(embed.token_emb);
    write_tensor(embed.pos_emb);
    for (const auto& block : blocks) {
        write_tensor(block.ln1.gamma); write_tensor(block.ln1.beta);
        write_tensor(block.attn.qkv_proj.W); write_tensor(block.attn.qkv_proj.b);
        write_tensor(block.attn.out_proj.W); write_tensor(block.attn.out_proj.b);
        write_tensor(block.ln2.gamma); write_tensor(block.ln2.beta);
        write_tensor(block.ffn.fc1.W); write_tensor(block.ffn.fc1.b);
        write_tensor(block.ffn.fc2.W); write_tensor(block.ffn.fc2.b);
    }
    write_tensor(ln_final.gamma); write_tensor(ln_final.beta);

    std::cout << "Model saved to: " << path << "\n";
}

void GPTModel::load(const std::string& path) {
    std::ifstream f(path, std::ios::binary);
    if (!f) throw std::runtime_error("Cannot open model file: " + path);

    f.read(reinterpret_cast<char*>(&cfg), sizeof(ModelConfig));

    auto read_tensor = [&](Tensor& t) {
        size_t n; f.read(reinterpret_cast<char*>(&n), sizeof(n));
        t.data.resize(n);
        f.read(reinterpret_cast<char*>(t.raw()), n * sizeof(float));
    };

    read_tensor(embed.token_emb);
    read_tensor(embed.pos_emb);
    for (auto& block : blocks) {
        read_tensor(block.ln1.gamma); read_tensor(block.ln1.beta);
        read_tensor(block.attn.qkv_proj.W); read_tensor(block.attn.qkv_proj.b);
        read_tensor(block.attn.out_proj.W); read_tensor(block.attn.out_proj.b);
        read_tensor(block.ln2.gamma); read_tensor(block.ln2.beta);
        read_tensor(block.ffn.fc1.W); read_tensor(block.ffn.fc1.b);
        read_tensor(block.ffn.fc2.W); read_tensor(block.ffn.fc2.b);
    }
    read_tensor(ln_final.gamma); read_tensor(ln_final.beta);
    tie_weights();
    std::cout << "Model loaded from: " << path << "\n";
}

} // namespace gpt
