#include "core/tensor.h"
#include "core/ops.h"
#include <cassert>
#include <cmath>
#include <iostream>
#include <vector>

using namespace gpt;

// ── helpers ──────────────────────────────────────────────────────────────────
static bool approx(float a, float b, float tol = 1e-4f) {
    return std::fabs(a - b) < tol;
}

// ── tests ─────────────────────────────────────────────────────────────────────
void test_tensor_creation() {
    Tensor t({2, 3, 4});
    assert(t.numel() == 24);
    assert(t.ndim()  == 3);
    assert(t.dim(0)  == 2);
    assert(t.dim(-1) == 4);
    std::cout << "[PASS] tensor_creation\n";
}

void test_tensor_fill() {
    Tensor t({4});
    t.fill(3.14f);
    for (size_t i = 0; i < t.numel(); ++i)
        assert(approx(t.at(i), 3.14f));
    std::cout << "[PASS] tensor_fill\n";
}

void test_matmul() {
    // 2×3 @ 3×2 = 2×2
    float A[6] = {1,2,3, 4,5,6};
    float B[6] = {7,8, 9,10, 11,12};
    float C[4] = {};
    ops::matmul(A, B, C, 2, 3, 2);
    // row0: 1*7+2*9+3*11=58, 1*8+2*10+3*12=64
    // row1: 4*7+5*9+6*11=139, 4*8+5*10+6*12=154
    assert(approx(C[0], 58.f));
    assert(approx(C[1], 64.f));
    assert(approx(C[2], 139.f));
    assert(approx(C[3], 154.f));
    std::cout << "[PASS] matmul\n";
}

void test_softmax() {
    float x[4] = {1.0f, 2.0f, 3.0f, 4.0f};
    ops::softmax_inplace(x, 4);
    float sum = 0.0f;
    for (int i = 0; i < 4; ++i) sum += x[i];
    assert(approx(sum, 1.0f));
    // monotonically increasing
    for (int i = 0; i < 3; ++i) assert(x[i] < x[i+1]);
    std::cout << "[PASS] softmax\n";
}

void test_gelu() {
    // GELU(0) = 0
    assert(approx(ops::gelu(0.0f), 0.0f));
    // GELU is roughly linear for large positive x
    assert(ops::gelu(5.0f) > 4.9f);
    // GELU(-inf) ≈ 0
    assert(ops::gelu(-10.0f) < 1e-4f);
    std::cout << "[PASS] gelu\n";
}

void test_layer_norm() {
    const size_t N = 4;
    float x[N]     = {1.f, 2.f, 3.f, 4.f};
    float gamma[N] = {1.f, 1.f, 1.f, 1.f};
    float beta[N]  = {0.f, 0.f, 0.f, 0.f};
    float out[N];
    ops::layer_norm(x, gamma, beta, out, N, 1e-5f);
    // output should have mean≈0, var≈1
    float mean = 0.f, var = 0.f;
    for (int i = 0; i < (int)N; ++i) mean += out[i];
    mean /= N;
    for (int i = 0; i < (int)N; ++i) var += (out[i]-mean)*(out[i]-mean);
    var /= N;
    assert(approx(mean, 0.0f, 1e-3f));
    assert(approx(var,  1.0f, 1e-3f));
    std::cout << "[PASS] layer_norm\n";
}

void test_transpose2d() {
    float A[6] = {1,2,3, 4,5,6}; // 2×3
    float B[6];
    ops::transpose2d(A, B, 2, 3);
    // B should be 3×2: [1,4, 2,5, 3,6]
    assert(approx(B[0], 1.f)); assert(approx(B[1], 4.f));
    assert(approx(B[2], 2.f)); assert(approx(B[3], 5.f));
    assert(approx(B[4], 3.f)); assert(approx(B[5], 6.f));
    std::cout << "[PASS] transpose2d\n";
}

void test_xavier_init() {
    Tensor t({100, 100});
    t.xavier_uniform(100, 100);
    float limit = std::sqrt(6.0f / 200.0f);
    for (size_t i = 0; i < t.numel(); ++i) {
        assert(t.at(i) >= -limit - 1e-4f);
        assert(t.at(i) <=  limit + 1e-4f);
    }
    std::cout << "[PASS] xavier_init\n";
}

int main() {
    std::cout << "=== Tensor & Ops Tests ===\n";
    test_tensor_creation();
    test_tensor_fill();
    test_matmul();
    test_softmax();
    test_gelu();
    test_layer_norm();
    test_transpose2d();
    test_xavier_init();
    std::cout << "\nAll tensor tests passed.\n";
    return 0;
}
