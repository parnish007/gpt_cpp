#pragma once
#include <vector>
#include <memory>
#include <cassert>
#include <cstring>
#include <stdexcept>
#include <functional>
#include <string>
#include <numeric>
#include <iostream>
#include <random>
#include <cmath>

namespace gpt {

// ─────────────────────────────────────────────────────────────────────────────
//  Tensor — owns a flat float buffer, tracks shape and optional gradient
// ─────────────────────────────────────────────────────────────────────────────
class Tensor {
public:
    std::vector<size_t>        shape;
    std::vector<float>         data;
    std::vector<float>         grad;
    bool                       requires_grad{false};

    // ── Constructors ─────────────────────────────────────────────────────────
    Tensor() = default;

    explicit Tensor(std::vector<size_t> shape_, bool req_grad = false)
        : shape(std::move(shape_)), requires_grad(req_grad)
    {
        data.resize(numel(), 0.0f);
        if (requires_grad) grad.resize(numel(), 0.0f);
    }

    // ── Shape helpers ────────────────────────────────────────────────────────
    size_t numel() const {
        if (shape.empty()) return 0;
        return std::accumulate(shape.begin(), shape.end(),
                               size_t(1), std::multiplies<size_t>());
    }

    size_t ndim()  const { return shape.size(); }
    size_t dim(int d) const {
        if (d < 0) d += (int)ndim();
        return shape[d];
    }

    // ── Data access ──────────────────────────────────────────────────────────
    float& at(size_t i)              { return data[i]; }
    float  at(size_t i) const        { return data[i]; }

    float* raw()                     { return data.data(); }
    const float* raw() const         { return data.data(); }
    float* grad_raw()                { return grad.data(); }
    const float* grad_raw() const    { return grad.data(); }

    // ── Gradient utilities ───────────────────────────────────────────────────
    void zero_grad() {
        if (!grad.empty()) std::fill(grad.begin(), grad.end(), 0.0f);
    }

    void enable_grad() {
        requires_grad = true;
        grad.assign(numel(), 0.0f);
    }

    // ── Reshape (returns view — no copy) ────────────────────────────────────
    Tensor reshape(std::vector<size_t> new_shape) const {
        Tensor out(new_shape, requires_grad);
        out.data = data;
        return out;
    }

    // ── Fill / init helpers ──────────────────────────────────────────────────
    void fill(float val) { std::fill(data.begin(), data.end(), val); }

    void randn(float mean = 0.0f, float std = 1.0f, uint32_t seed = 42) {
        std::mt19937 rng(seed);
        std::normal_distribution<float> dist(mean, std);
        for (auto& v : data) v = dist(rng);
    }

    void xavier_uniform(size_t fan_in, size_t fan_out) {
        float limit = std::sqrt(6.0f / (float)(fan_in + fan_out));
        std::mt19937 rng(std::random_device{}());
        std::uniform_real_distribution<float> dist(-limit, limit);
        for (auto& v : data) v = dist(rng);
    }

    // ── Debug ────────────────────────────────────────────────────────────────
    std::string shape_str() const {
        std::string s = "(";
        for (size_t i = 0; i < shape.size(); ++i) {
            s += std::to_string(shape[i]);
            if (i + 1 < shape.size()) s += ", ";
        }
        return s + ")";
    }

    void print_summary(const std::string& name = "") const {
        std::cout << (name.empty() ? "Tensor" : name)
                  << " shape=" << shape_str()
                  << " numel=" << numel() << "\n";
    }
};

using TensorPtr = std::shared_ptr<Tensor>;

inline TensorPtr make_tensor(std::vector<size_t> shape, bool req_grad = false) {
    return std::make_shared<Tensor>(std::move(shape), req_grad);
}

} // namespace gpt
