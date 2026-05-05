#pragma once
#include <cstddef>
#include <string>

namespace gpt {

// ─────────────────────────────────────────────────────────────────────────────
//  ModelConfig — architecture hyperparameters
// ─────────────────────────────────────────────────────────────────────────────
struct ModelConfig {
    size_t vocab_size  = 256;    // character-level default
    size_t embed_dim   = 256;
    size_t n_heads     = 8;
    size_t n_layers    = 6;
    size_t max_seq_len = 256;
    float  dropout     = 0.1f;
    float  eps         = 1e-5f;  // layer norm epsilon
};

// ─────────────────────────────────────────────────────────────────────────────
//  TrainConfig — training hyperparameters
// ─────────────────────────────────────────────────────────────────────────────
struct TrainConfig {
    size_t batch_size    = 32;
    size_t total_steps   = 5000;
    size_t eval_every    = 100;
    size_t gen_every     = 500;
    size_t warmup_steps  = 200;
    float  lr            = 3e-4f;
    float  weight_decay  = 0.1f;
    float  beta1         = 0.9f;
    float  beta2         = 0.999f;
    float  eps           = 1e-8f;
    float  grad_clip     = 1.0f;
    std::string data_path   = "data/input.txt";
    std::string ckpt_dir    = "checkpoints";
    std::string seed_text   = "The ";
};

// ─────────────────────────────────────────────────────────────────────────────
//  InferenceConfig
// ─────────────────────────────────────────────────────────────────────────────
struct InferenceConfig {
    std::string model_path  = "checkpoints/model.bin";
    std::string vocab_path  = "checkpoints/vocab.bin";
    std::string prompt      = "Once upon a time";
    size_t      max_tokens  = 200;
    float       temperature = 0.8f;
    float       top_p       = 0.9f;   // nucleus sampling
    int         top_k       = 40;
};

} // namespace gpt
