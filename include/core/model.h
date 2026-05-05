#pragma once
#include "layers.h"
#include "config.h"
#include <vector>
#include <memory>
#include <string>

namespace gpt {

// ─────────────────────────────────────────────────────────────────────────────
//  GPT — decoder-only transformer
// ─────────────────────────────────────────────────────────────────────────────
class GPTModel {
public:
    ModelConfig cfg;

    Embedding                       embed;
    Dropout                         drop;
    std::vector<TransformerBlock>   blocks;
    LayerNorm                       ln_final;
    Linear                          lm_head;

    // cache for backward
    Tensor logits_cache;
    Tensor probs_cache;
    std::vector<int> targets_cache;
    size_t BT_cache{0};

    explicit GPTModel(const ModelConfig& cfg);

    // ── Forward pass ─────────────────────────────────────────────────────────
    // idx: flat int array of shape (B, T)
    // returns logits: Tensor (B, T, vocab_size)
    Tensor forward(const int* idx, size_t B, size_t T);

    // ── Loss (cross-entropy) ─────────────────────────────────────────────────
    // logits: (B,T,V)   targets: flat int (B*T)
    float loss(const Tensor& logits, const int* targets, size_t BT);

    // ── Backward ─────────────────────────────────────────────────────────────
    void backward();

    // ── Parameter collection ─────────────────────────────────────────────────
    std::vector<std::pair<float*, float*>> parameters();
    std::vector<size_t> param_sizes();
    size_t num_params();

    // ── Gradient management ──────────────────────────────────────────────────
    void zero_grad();

    // ── Train / eval mode ────────────────────────────────────────────────────
    void set_training(bool mode);

    // ── Serialization ────────────────────────────────────────────────────────
    void save(const std::string& path) const;
    void load(const std::string& path);

private:
    void tie_weights();   // share token_emb <-> lm_head.W
};

} // namespace gpt
