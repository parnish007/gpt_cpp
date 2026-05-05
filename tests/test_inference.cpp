#include "inference/engine.h"
#include "core/model.h"
#include "training/dataset.h"
#include <cassert>
#include <cmath>
#include <iostream>
#include <numeric>

using namespace gpt;

// Minimal model config for fast tests
static ModelConfig small_cfg() {
    ModelConfig cfg;
    cfg.vocab_size  = 30;
    cfg.embed_dim   = 32;
    cfg.n_heads     = 4;
    cfg.n_layers    = 2;
    cfg.max_seq_len = 16;
    cfg.dropout     = 0.0f;
    return cfg;
}

void test_greedy_is_deterministic() {
    auto cfg = small_cfg();
    GPTModel model(cfg);
    CharTokenizer tok;
    tok.build("abcdefghijklmnopqrstuvwxyz ");

    InferenceEngine engine(model, tok, /*seed=*/42);
    SamplingParams sp; sp.strategy = SamplingStrategy::Greedy;

    std::string out1 = engine.generate("abc", 20, sp);
    std::string out2 = engine.generate("abc", 20, sp);
    assert(out1 == out2);
    std::cout << "[PASS] greedy_is_deterministic\n";
}

void test_generate_length() {
    auto cfg = small_cfg();
    GPTModel model(cfg);
    CharTokenizer tok;
    tok.build("abcdefghijklmnopqrstuvwxyz ");

    InferenceEngine engine(model, tok, 42);
    SamplingParams sp; sp.strategy = SamplingStrategy::Temperature; sp.temperature = 1.0f;

    size_t new_tokens = 15;
    std::string prompt = "hello";
    std::string out = engine.generate(prompt, new_tokens, sp);
    // output should be prompt + new_tokens characters
    assert(out.size() == prompt.size() + new_tokens);
    std::cout << "[PASS] generate_length (len=" << out.size() << ")\n";
}

void test_topk_stays_in_vocab() {
    auto cfg = small_cfg();
    GPTModel model(cfg);
    CharTokenizer tok;
    tok.build("abcdefghijklmnopqrstuvwxyz ");

    InferenceEngine engine(model, tok, 99);
    SamplingParams sp;
    sp.strategy    = SamplingStrategy::TopK;
    sp.top_k       = 5;
    sp.temperature = 1.0f;

    std::string out = engine.generate("test", 30, sp);
    for (char c : out) {
        assert(tok.ch2id.count(c) > 0);
    }
    std::cout << "[PASS] topk_stays_in_vocab\n";
}

void test_topp_stays_in_vocab() {
    auto cfg = small_cfg();
    GPTModel model(cfg);
    CharTokenizer tok;
    tok.build("abcdefghijklmnopqrstuvwxyz ");

    InferenceEngine engine(model, tok, 7);
    SamplingParams sp;
    sp.strategy    = SamplingStrategy::TopP;
    sp.top_p       = 0.9f;
    sp.temperature = 1.0f;

    std::string out = engine.generate("ab", 20, sp);
    for (char c : out) {
        assert(tok.ch2id.count(c) > 0);
    }
    std::cout << "[PASS] topp_stays_in_vocab\n";
}

void test_context_window_truncation() {
    // Generate more tokens than max_seq_len to exercise context window clipping
    auto cfg = small_cfg();  // max_seq_len = 16
    GPTModel model(cfg);
    CharTokenizer tok;
    tok.build("abcdefghijklmnopqrstuvwxyz ");

    InferenceEngine engine(model, tok, 1);
    SamplingParams sp; sp.strategy = SamplingStrategy::Greedy;

    // 30 new tokens > max_seq_len=16, should not crash
    std::string out = engine.generate("a", 30, sp);
    assert(out.size() == 31);
    std::cout << "[PASS] context_window_truncation\n";
}

void test_model_save_load() {
    auto cfg = small_cfg();
    GPTModel model(cfg);

    // Save
    const char* path = "/tmp/test_model.bin";
    model.save(path);

    // Load into new model
    GPTModel model2(cfg);
    model2.load(path);

    // Forward should give identical outputs
    int idx[4] = {0, 1, 2, 3};
    model.set_training(false);
    model2.set_training(false);
    Tensor out1 = model.forward(idx, 1, 4);
    Tensor out2 = model2.forward(idx, 1, 4);

    float max_diff = 0.0f;
    for (size_t i = 0; i < out1.numel(); ++i)
        max_diff = std::max(max_diff, std::fabs(out1.raw()[i] - out2.raw()[i]));
    assert(max_diff < 1e-5f);

    std::remove(path);
    std::cout << "[PASS] model_save_load (max_diff=" << max_diff << ")\n";
}

int main() {
    std::cout << "=== Inference Engine Tests ===\n";
    test_greedy_is_deterministic();
    test_generate_length();
    test_topk_stays_in_vocab();
    test_topp_stays_in_vocab();
    test_context_window_truncation();
    test_model_save_load();
    std::cout << "\nAll inference tests passed.\n";
    return 0;
}
