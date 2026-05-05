#pragma once
#include "core/model.h"
#include <vector>
#include <cmath>

namespace gpt {

// ─────────────────────────────────────────────────────────────────────────────
//  AdamW optimizer  (decoupled weight decay)
// ─────────────────────────────────────────────────────────────────────────────
class AdamW {
public:
    float lr, beta1, beta2, eps, weight_decay;
    int   step_count{0};

    using ParamList = std::vector<std::pair<float*, float*>>;
    ParamList          params;
    std::vector<size_t> sizes;

    std::vector<std::vector<float>> m;  // first moment
    std::vector<std::vector<float>> v;  // second moment

    AdamW(ParamList params_, std::vector<size_t> sizes_,
          float lr = 3e-4f, float beta1 = 0.9f, float beta2 = 0.999f,
          float eps = 1e-8f, float weight_decay = 0.1f)
        : lr(lr), beta1(beta1), beta2(beta2), eps(eps), weight_decay(weight_decay),
          params(std::move(params_)), sizes(std::move(sizes_))
    {
        m.resize(params.size());
        v.resize(params.size());
        for (size_t i = 0; i < params.size(); ++i) {
            m[i].assign(sizes[i], 0.0f);
            v[i].assign(sizes[i], 0.0f);
        }
    }

    void step() {
        ++step_count;
        float bc1 = 1.0f - std::pow(beta1, step_count);
        float bc2 = 1.0f - std::pow(beta2, step_count);
        float lr_t = lr * std::sqrt(bc2) / bc1;

        for (size_t p = 0; p < params.size(); ++p) {
            float* w = params[p].first;
            float* g = params[p].second;
            if (!w || !g) continue;
            size_t n = sizes[p];

            for (size_t i = 0; i < n; ++i) {
                // decoupled weight decay
                w[i] *= (1.0f - lr * weight_decay);
                // moment update
                m[p][i] = beta1 * m[p][i] + (1.0f - beta1) * g[i];
                v[p][i] = beta2 * v[p][i] + (1.0f - beta2) * g[i] * g[i];
                // parameter update
                w[i] -= lr_t * m[p][i] / (std::sqrt(v[p][i]) + eps);
            }
        }
    }
};

// ─────────────────────────────────────────────────────────────────────────────
//  Cosine LR Scheduler with linear warmup
// ─────────────────────────────────────────────────────────────────────────────
class CosineScheduler {
public:
    AdamW&  optimizer;
    size_t  warmup_steps, total_steps;
    float   base_lr, min_lr;
    size_t  step_count{0};

    CosineScheduler(AdamW& opt, size_t warmup, size_t total, float min_lr = 1e-5f)
        : optimizer(opt), warmup_steps(warmup), total_steps(total),
          base_lr(opt.lr), min_lr(min_lr) {}

    float step() {
        ++step_count;
        float lr;
        if (step_count <= warmup_steps) {
            lr = base_lr * (float)step_count / (float)warmup_steps;
        } else {
            float progress = (float)(step_count - warmup_steps)
                           / (float)(total_steps - warmup_steps);
            lr = min_lr + 0.5f * (base_lr - min_lr)
                        * (1.0f + std::cos(3.14159265f * progress));
        }
        optimizer.lr = lr;
        return lr;
    }
};

// ─────────────────────────────────────────────────────────────────────────────
//  Gradient clipping  (global L2 norm)
// ─────────────────────────────────────────────────────────────────────────────
inline void clip_grad_norm(const std::vector<std::pair<float*, float*>>& params,
                            const std::vector<size_t>& sizes,
                            float max_norm)
{
    float total_sq = 0.0f;
    for (size_t p = 0; p < params.size(); ++p) {
        float* g = params[p].second;
        if (!g) continue;
        for (size_t i = 0; i < sizes[p]; ++i) total_sq += g[i] * g[i];
    }
    float norm = std::sqrt(total_sq);
    if (norm > max_norm) {
        float scale = max_norm / (norm + 1e-6f);
        for (size_t p = 0; p < params.size(); ++p) {
            float* g = params[p].second;
            if (!g) continue;
            for (size_t i = 0; i < sizes[p]; ++i) g[i] *= scale;
        }
    }
}

} // namespace gpt
