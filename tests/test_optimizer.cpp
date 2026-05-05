#include "training/optimizer.h"
#include "core/model.h"
#include <cassert>
#include <cmath>
#include <iostream>
#include <vector>

using namespace gpt;

static bool approx(float a, float b, float tol = 1e-4f) {
    return std::fabs(a - b) < tol;
}

void test_adamw_step_reduces_loss() {
    // Simple 1-param quadratic: loss = w^2, grad = 2w
    // After enough steps w should approach 0
    std::vector<float> w = {5.0f};
    std::vector<float> g = {0.0f};

    std::vector<std::pair<float*, float*>> params = {{w.data(), g.data()}};
    std::vector<size_t> sizes = {1};

    AdamW opt(params, sizes, /*lr=*/0.1f, 0.9f, 0.999f, 1e-8f, /*wd=*/0.0f);

    for (int i = 0; i < 500; ++i) {
        g[0] = 2.0f * w[0];  // gradient of w^2
        opt.step();
    }
    assert(std::fabs(w[0]) < 0.1f);
    std::cout << "[PASS] adamw_converges (w=" << w[0] << ")\n";
}

void test_adamw_weight_decay() {
    // With weight decay and zero gradient, weights should shrink toward 0
    std::vector<float> w = {1.0f};
    std::vector<float> g = {0.0f};
    std::vector<std::pair<float*, float*>> params = {{w.data(), g.data()}};
    std::vector<size_t> sizes = {1};

    AdamW opt(params, sizes, /*lr=*/0.01f, 0.9f, 0.999f, 1e-8f, /*wd=*/0.1f);
    float prev = w[0];
    opt.step();
    // weight should have decayed
    assert(w[0] < prev);
    std::cout << "[PASS] adamw_weight_decay\n";
}

void test_cosine_scheduler_warmup() {
    std::vector<float> w = {1.0f};
    std::vector<float> g = {0.0f};
    std::vector<std::pair<float*, float*>> params = {{w.data(), g.data()}};
    std::vector<size_t> sizes = {1};

    AdamW opt(params, sizes, /*lr=*/1.0f);
    CosineScheduler sched(opt, /*warmup=*/10, /*total=*/100);

    // During warmup, lr should increase linearly
    float prev_lr = 0.0f;
    for (int i = 1; i <= 10; ++i) {
        float lr = sched.step();
        assert(lr >= prev_lr);
        prev_lr = lr;
    }
    assert(approx(prev_lr, 1.0f, 0.01f));  // at end of warmup, lr ≈ base_lr

    // After warmup, lr should decrease
    float mid_lr = sched.step();
    for (int i = 12; i <= 100; ++i) {
        float lr = sched.step();
        (void)lr;
    }
    assert(opt.lr < mid_lr);
    std::cout << "[PASS] cosine_scheduler_warmup\n";
}

void test_grad_clip() {
    // Create large gradients and check they get clipped
    std::vector<float> w(10, 1.0f);
    std::vector<float> g(10, 10.0f);  // large gradient
    std::vector<std::pair<float*, float*>> params = {{w.data(), g.data()}};
    std::vector<size_t> sizes = {10};

    clip_grad_norm(params, sizes, 1.0f);

    // After clipping, L2 norm of gradients should be ≤ max_norm
    float norm = 0.0f;
    for (float v : g) norm += v * v;
    norm = std::sqrt(norm);
    assert(norm <= 1.01f);
    std::cout << "[PASS] grad_clip (norm=" << norm << ")\n";
}

void test_grad_clip_small_grad() {
    // Gradient already small — should not be changed
    std::vector<float> w(4, 1.0f);
    std::vector<float> g = {0.1f, 0.1f, 0.1f, 0.1f};
    std::vector<std::pair<float*, float*>> params = {{w.data(), g.data()}};
    std::vector<size_t> sizes = {4};

    float before = 0.0f;
    for (float v : g) before += v * v;
    before = std::sqrt(before);

    clip_grad_norm(params, sizes, 10.0f);  // max_norm > actual norm

    float after = 0.0f;
    for (float v : g) after += v * v;
    after = std::sqrt(after);

    assert(approx(before, after));
    std::cout << "[PASS] grad_clip_small_grad\n";
}

int main() {
    std::cout << "=== Optimizer Tests ===\n";
    test_adamw_step_reduces_loss();
    test_adamw_weight_decay();
    test_cosine_scheduler_warmup();
    test_grad_clip();
    test_grad_clip_small_grad();
    std::cout << "\nAll optimizer tests passed.\n";
    return 0;
}
