#include "core/layers.h"
#include "core/model.h"
#include <cassert>
#include <cmath>
#include <iostream>
#include <vector>
#include <random>

using namespace gpt;

static bool approx(float a, float b, float tol = 1e-3f) {
    return std::fabs(a - b) / (std::fabs(b) + 1e-8f) < tol;
}

// ── Numerical gradient checker ─────────────────────────────────────────────
// Perturbs each input element by eps and computes finite-difference gradient.
// Returns max relative error between analytical and numerical grads.
float grad_check_linear(size_t in_dim, size_t out_dim,
                         size_t B, size_t T)
{
    Linear layer(in_dim, out_dim, true);
    layer.set_training(false);

    // Random input
    Tensor x({B, T, in_dim});
    std::mt19937 rng(1);
    std::normal_distribution<float> dist(0.f, 1.f);
    for (size_t i = 0; i < x.numel(); ++i) x.raw()[i] = dist(rng);

    // Random upstream gradient
    Tensor dout({B, T, out_dim});
    for (size_t i = 0; i < dout.numel(); ++i) dout.raw()[i] = dist(rng);

    // Analytical gradient
    Tensor out = layer.forward(x);
    Tensor dx  = layer.backward(dout);

    // Numerical gradient via finite differences
    float eps = 1e-3f, max_err = 0.f;
    for (size_t i = 0; i < std::min(x.numel(), size_t(20)); ++i) {
        float orig = x.raw()[i];

        x.raw()[i] = orig + eps;
        layer.dW.fill(0); layer.db.fill(0);
        Tensor op = layer.forward(x);
        float fp = 0.f;
        for (size_t j = 0; j < op.numel(); ++j) fp += op.raw()[j] * dout.raw()[j];

        x.raw()[i] = orig - eps;
        layer.dW.fill(0); layer.db.fill(0);
        Tensor om = layer.forward(x);
        float fm = 0.f;
        for (size_t j = 0; j < om.numel(); ++j) fm += om.raw()[j] * dout.raw()[j];

        float num_grad = (fp - fm) / (2.f * eps);
        float ana_grad = dx.raw()[i];
        float err = std::fabs(num_grad - ana_grad) / (std::fabs(num_grad) + 1e-6f);
        if (err > max_err) max_err = err;
        x.raw()[i] = orig;
    }
    return max_err;
}

void test_linear_shapes() {
    Linear layer(32, 64);
    Tensor x({2, 8, 32});
    x.randn();
    Tensor out = layer.forward(x);
    assert(out.dim(0) == 2);
    assert(out.dim(1) == 8);
    assert(out.dim(2) == 64);
    std::cout << "[PASS] linear_shapes\n";
}

void test_linear_grad_check() {
    float err = grad_check_linear(16, 32, 2, 4);
    assert(err < 0.01f);
    std::cout << "[PASS] linear_grad_check (max_err=" << err << ")\n";
}

void test_layernorm_shapes() {
    LayerNorm ln(64);
    Tensor x({2, 8, 64});
    x.randn();
    Tensor out = ln.forward(x);
    assert(out.numel() == x.numel());

    Tensor dout({2, 8, 64});
    dout.randn();
    Tensor dx = ln.backward(dout);
    assert(dx.numel() == x.numel());
    std::cout << "[PASS] layernorm_shapes\n";
}

void test_layernorm_output() {
    // output of LN should have ~zero mean and ~unit variance per row
    LayerNorm ln(128);
    Tensor x({1, 1, 128});
    std::mt19937 rng(42);
    std::normal_distribution<float> d(5.f, 3.f);
    for (size_t i = 0; i < x.numel(); ++i) x.raw()[i] = d(rng);
    Tensor out = ln.forward(x);

    float mean = 0.f, var = 0.f;
    for (size_t i = 0; i < 128; ++i) mean += out.raw()[i];
    mean /= 128.f;
    for (size_t i = 0; i < 128; ++i) var += (out.raw()[i]-mean)*(out.raw()[i]-mean);
    var /= 128.f;
    assert(std::fabs(mean) < 1e-4f);
    assert(std::fabs(var - 1.f) < 1e-3f);
    std::cout << "[PASS] layernorm_output (mean=" << mean << " var=" << var << ")\n";
}

void test_embedding_shapes() {
    Embedding emb(50, 32, 64);
    int idx[8] = {0, 1, 2, 3, 4, 5, 6, 7};
    Tensor out = emb.forward(idx, 2, 4);
    assert(out.dim(0) == 2);
    assert(out.dim(1) == 4);
    assert(out.dim(2) == 32);
    std::cout << "[PASS] embedding_shapes\n";
}

void test_dropout_training() {
    Dropout drop(0.5f);
    drop.set_training(true);
    Tensor x({100});
    x.fill(1.0f);
    Tensor out = drop.forward(x);
    // Roughly half should be zero
    int zeros = 0;
    for (size_t i = 0; i < out.numel(); ++i)
        if (out.raw()[i] == 0.f) zeros++;
    assert(zeros > 20 && zeros < 80);
    std::cout << "[PASS] dropout_training (zeros=" << zeros << "/100)\n";
}

void test_dropout_eval() {
    Dropout drop(0.5f);
    drop.set_training(false);
    Tensor x({10}); x.fill(1.0f);
    Tensor out = drop.forward(x);
    for (size_t i = 0; i < out.numel(); ++i)
        assert(out.raw()[i] == 1.0f);
    std::cout << "[PASS] dropout_eval\n";
}

void test_attention_shapes() {
    MultiHeadAttention attn(64, 8, 0.0f);
    attn.set_training(false);
    Tensor x({2, 16, 64}); x.randn();
    Tensor out = attn.forward(x);
    assert(out.dim(0) == 2);
    assert(out.dim(1) == 16);
    assert(out.dim(2) == 64);
    std::cout << "[PASS] attention_shapes\n";
}

void test_ffn_shapes() {
    FeedForward ffn(64, 0.0f);
    ffn.set_training(false);
    Tensor x({2, 8, 64}); x.randn();
    Tensor out = ffn.forward(x);
    assert(out.numel() == x.numel());
    std::cout << "[PASS] ffn_shapes\n";
}

void test_block_shapes() {
    TransformerBlock block(64, 8, 0.0f);
    block.set_training(false);
    Tensor x({1, 32, 64}); x.randn();
    Tensor out = block.forward(x);
    assert(out.numel() == x.numel());
    std::cout << "[PASS] transformer_block_shapes\n";
}

void test_gpt_forward() {
    ModelConfig cfg;
    cfg.vocab_size  = 50;
    cfg.embed_dim   = 32;
    cfg.n_heads     = 4;
    cfg.n_layers    = 2;
    cfg.max_seq_len = 16;
    cfg.dropout     = 0.0f;

    GPTModel model(cfg);
    model.set_training(false);

    int idx[16] = {0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15};
    Tensor logits = model.forward(idx, 1, 16);
    assert(logits.dim(0) == 1);
    assert(logits.dim(1) == 16);
    assert(logits.dim(2) == 50);
    std::cout << "[PASS] gpt_forward\n";
}

void test_gpt_loss_backward() {
    ModelConfig cfg;
    cfg.vocab_size  = 20;
    cfg.embed_dim   = 16;
    cfg.n_heads     = 2;
    cfg.n_layers    = 1;
    cfg.max_seq_len = 8;
    cfg.dropout     = 0.0f;

    GPTModel model(cfg);
    model.set_training(true);
    model.zero_grad();

    int idx[8]  = {0,1,2,3,4,5,6,7};
    int tgt[8]  = {1,2,3,4,5,6,7,0};
    Tensor logits = model.forward(idx, 1, 8);
    float loss    = model.loss(logits, tgt, 8);
    assert(loss > 0.f && loss < 20.f);
    model.backward();

    // Check at least one gradient is non-zero
    bool has_nonzero = false;
    for (float v : model.embed.d_token_emb.data)
        if (std::fabs(v) > 1e-8f) { has_nonzero = true; break; }
    assert(has_nonzero);
    std::cout << "[PASS] gpt_loss_backward (loss=" << loss << ")\n";
}

int main() {
    std::cout << "=== Layer Tests ===\n";
    test_linear_shapes();
    test_linear_grad_check();
    test_layernorm_shapes();
    test_layernorm_output();
    test_embedding_shapes();
    test_dropout_training();
    test_dropout_eval();
    test_attention_shapes();
    test_ffn_shapes();
    test_block_shapes();
    test_gpt_forward();
    test_gpt_loss_backward();
    std::cout << "\nAll layer tests passed.\n";
    return 0;
}
