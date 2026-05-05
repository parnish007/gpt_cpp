#pragma once
#include "tensor.h"
#include <algorithm>
#include <cmath>

#ifdef USE_OPENMP
#include <omp.h>
#endif

namespace gpt {
namespace ops {

// ─────────────────────────────────────────────────────────────────────────────
//  Matrix multiply  C = A @ B
//  A: (M, K)   B: (K, N)   C: (M, N)
// ─────────────────────────────────────────────────────────────────────────────
inline void matmul(const float* A, const float* B, float* C,
                   size_t M, size_t K, size_t N,
                   bool accumulate = false)
{
    if (!accumulate) std::fill(C, C + M * N, 0.0f);

    #ifdef USE_OPENMP
    #pragma omp parallel for schedule(static)
    #endif
    for (size_t i = 0; i < M; ++i) {
        for (size_t k = 0; k < K; ++k) {
            float a = A[i * K + k];
            for (size_t j = 0; j < N; ++j) {
                C[i * N + j] += a * B[k * N + j];
            }
        }
    }
}

// ─────────────────────────────────────────────────────────────────────────────
//  Batched matmul  C[b] = A[b] @ B[b]
//  A: (B, M, K)  B: (B, K, N)  C: (B, M, N)
// ─────────────────────────────────────────────────────────────────────────────
inline void batched_matmul(const float* A, const float* B, float* C,
                            size_t batch, size_t M, size_t K, size_t N,
                            bool accumulate = false)
{
    for (size_t b = 0; b < batch; ++b) {
        matmul(A + b * M * K, B + b * K * N, C + b * M * N, M, K, N, accumulate);
    }
}

// ─────────────────────────────────────────────────────────────────────────────
//  Softmax (in-place, over last axis of a 2-D slice)
//  x: pointer to row of length N
// ─────────────────────────────────────────────────────────────────────────────
inline void softmax_inplace(float* x, size_t N) {
    float max_val = *std::max_element(x, x + N);
    float sum = 0.0f;
    for (size_t i = 0; i < N; ++i) { x[i] = std::exp(x[i] - max_val); sum += x[i]; }
    for (size_t i = 0; i < N; ++i) x[i] /= sum;
}

// ─────────────────────────────────────────────────────────────────────────────
//  GELU activation  (tanh approximation)
// ─────────────────────────────────────────────────────────────────────────────
inline float gelu(float x) {
    constexpr float c = 0.7978845608f; // sqrt(2/pi)
    return 0.5f * x * (1.0f + std::tanh(c * (x + 0.044715f * x * x * x)));
}

inline float gelu_grad(float x) {
    constexpr float c = 0.7978845608f;
    float t  = std::tanh(c * (x + 0.044715f * x * x * x));
    float dt = (1.0f - t * t) * c * (1.0f + 3.0f * 0.044715f * x * x);
    return 0.5f * (1.0f + t) + 0.5f * x * dt;
}

// ─────────────────────────────────────────────────────────────────────────────
//  Element-wise ops
// ─────────────────────────────────────────────────────────────────────────────
inline void add_inplace(float* dst, const float* src, size_t n) {
    for (size_t i = 0; i < n; ++i) dst[i] += src[i];
}

inline void scale_inplace(float* x, float s, size_t n) {
    for (size_t i = 0; i < n; ++i) x[i] *= s;
}

inline float dot(const float* a, const float* b, size_t n) {
    float s = 0.0f;
    for (size_t i = 0; i < n; ++i) s += a[i] * b[i];
    return s;
}

// ─────────────────────────────────────────────────────────────────────────────
//  Transpose  (2-D)   out[j][i] = in[i][j]
// ─────────────────────────────────────────────────────────────────────────────
inline void transpose2d(const float* in, float* out, size_t rows, size_t cols) {
    for (size_t i = 0; i < rows; ++i)
        for (size_t j = 0; j < cols; ++j)
            out[j * rows + i] = in[i * cols + j];
}

// ─────────────────────────────────────────────────────────────────────────────
//  Layer Norm forward
//  x: (N,)  gamma: (N,)  beta: (N,)  → out: (N,)
// ─────────────────────────────────────────────────────────────────────────────
inline void layer_norm(const float* x, const float* gamma, const float* beta,
                       float* out, size_t N, float eps,
                       float* mean_out = nullptr, float* var_out = nullptr)
{
    float mean = 0.0f, var = 0.0f;
    for (size_t i = 0; i < N; ++i) mean += x[i];
    mean /= (float)N;
    for (size_t i = 0; i < N; ++i) { float d = x[i] - mean; var += d * d; }
    var /= (float)N;
    float inv_std = 1.0f / std::sqrt(var + eps);
    for (size_t i = 0; i < N; ++i)
        out[i] = gamma[i] * ((x[i] - mean) * inv_std) + beta[i];
    if (mean_out) *mean_out = mean;
    if (var_out)  *var_out  = var;
}

} // namespace ops
} // namespace gpt
