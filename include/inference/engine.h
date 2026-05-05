#pragma once
#include "core/model.h"
#include "core/config.h"
#include "training/dataset.h"
#include <vector>
#include <string>
#include <random>
#include <algorithm>
#include <numeric>
#include <stdexcept>
#include <iostream>
#include <chrono>

namespace gpt {

// ─────────────────────────────────────────────────────────────────────────────
//  SamplingStrategy — how to pick the next token from logits
// ─────────────────────────────────────────────────────────────────────────────
enum class SamplingStrategy { Greedy, Temperature, TopK, TopP, TopKP };

struct SamplingParams {
    SamplingStrategy strategy   = SamplingStrategy::TopKP;
    float            temperature = 0.8f;
    int              top_k       = 40;
    float            top_p       = 0.9f;
};

// ─────────────────────────────────────────────────────────────────────────────
//  InferenceEngine — wraps a trained GPTModel for fast text generation
// ─────────────────────────────────────────────────────────────────────────────
class InferenceEngine {
public:
    GPTModel&       model;
    CharTokenizer&  tokenizer;
    std::mt19937    rng;

    InferenceEngine(GPTModel& model_, CharTokenizer& tok_, uint32_t seed = 42)
        : model(model_), tokenizer(tok_), rng(seed)
    {
        model.set_training(false);
    }

    // ── Generate text from a prompt ──────────────────────────────────────────
    std::string generate(const std::string& prompt,
                         size_t max_new_tokens,
                         const SamplingParams& sp = {})
    {
        auto ids = tokenizer.encode(prompt);
        if (ids.empty())
            throw std::runtime_error("Prompt encoded to empty token sequence");

        size_t max_len = model.cfg.max_seq_len;
        auto t0 = std::chrono::steady_clock::now();

        for (size_t i = 0; i < max_new_tokens; ++i) {
            // Context window
            size_t T  = std::min(ids.size(), max_len);
            auto   ctx = std::vector<int>(ids.end() - T, ids.end());

            Tensor logits = model.forward(ctx.data(), 1, T);

            // Extract last-timestep logits
            size_t V = model.cfg.vocab_size;
            std::vector<float> logit_vec(logits.raw() + (T - 1) * V,
                                         logits.raw() + T * V);

            int next_id = sample(logit_vec, sp);
            ids.push_back(next_id);
        }

        auto t1 = std::chrono::steady_clock::now();
        double secs = std::chrono::duration<double>(t1 - t0).count();
        std::cout << "Generated " << max_new_tokens << " tokens in "
                  << secs << "s ("
                  << (float)max_new_tokens / secs << " tok/s)\n";

        return tokenizer.decode(ids);
    }

    // ── Interactive REPL ─────────────────────────────────────────────────────
    void repl(const SamplingParams& sp = {}) {
        std::cout << "=== GPT Inference Engine ===\n"
                  << "Type a prompt and press Enter. Ctrl+C to quit.\n\n";
        std::string prompt;
        while (true) {
            std::cout << ">>> ";
            if (!std::getline(std::cin, prompt)) break;
            if (prompt.empty()) continue;
            std::string out = generate(prompt, 200, sp);
            std::cout << "\n" << out << "\n\n";
        }
    }

private:
    // ── Sampling implementations ─────────────────────────────────────────────
    int sample(std::vector<float>& logits, const SamplingParams& sp) {
        switch (sp.strategy) {
            case SamplingStrategy::Greedy:
                return (int)(std::max_element(logits.begin(), logits.end()) - logits.begin());

            case SamplingStrategy::Temperature:
                return sample_temperature(logits, sp.temperature);

            case SamplingStrategy::TopK:
                return sample_topk(logits, sp.top_k, sp.temperature);

            case SamplingStrategy::TopP:
                return sample_topp(logits, sp.top_p, sp.temperature);

            case SamplingStrategy::TopKP:
                return sample_topkp(logits, sp.top_k, sp.top_p, sp.temperature);

            default:
                return sample_temperature(logits, sp.temperature);
        }
    }

    // temperature sampling
    int sample_temperature(std::vector<float>& logits, float temp) {
        apply_temperature(logits, temp);
        softmax_inplace(logits);
        return categorical(logits);
    }

    // top-k sampling
    int sample_topk(std::vector<float>& logits, int k, float temp) {
        apply_temperature(logits, temp);
        topk_mask(logits, k);
        softmax_inplace(logits);
        return categorical(logits);
    }

    // nucleus (top-p) sampling
    int sample_topp(std::vector<float>& logits, float p, float temp) {
        apply_temperature(logits, temp);
        topp_mask(logits, p);
        softmax_inplace(logits);
        return categorical(logits);
    }

    // top-k then nucleus
    int sample_topkp(std::vector<float>& logits, int k, float p, float temp) {
        apply_temperature(logits, temp);
        topk_mask(logits, k);
        topp_mask(logits, p);
        softmax_inplace(logits);
        return categorical(logits);
    }

    // ── Helpers ──────────────────────────────────────────────────────────────
    void apply_temperature(std::vector<float>& logits, float temp) {
        if (temp != 1.0f && temp > 0.0f)
            for (auto& l : logits) l /= temp;
    }

    void softmax_inplace(std::vector<float>& v) {
        float mx = *std::max_element(v.begin(), v.end());
        float sum = 0.0f;
        for (auto& x : v) { x = std::exp(x - mx); sum += x; }
        for (auto& x : v) x /= sum;
    }

    void topk_mask(std::vector<float>& logits, int k) {
        size_t V = logits.size();
        if (k <= 0 || (size_t)k >= V) return;
        std::vector<size_t> idx(V);
        std::iota(idx.begin(), idx.end(), 0);
        std::partial_sort(idx.begin(), idx.begin() + k, idx.end(),
            [&](size_t a, size_t b){ return logits[a] > logits[b]; });
        float kth = logits[idx[k - 1]];
        for (auto& l : logits) if (l < kth) l = -1e9f;
    }

    void topp_mask(std::vector<float>& logits, float p) {
        // Operate on a sorted copy to find cutoff
        std::vector<std::pair<float, size_t>> sv;
        sv.reserve(logits.size());
        float mx = *std::max_element(logits.begin(), logits.end());
        float sum = 0.0f;
        for (size_t i = 0; i < logits.size(); ++i) {
            float e = std::exp(logits[i] - mx); sv.push_back({e, i}); sum += e;
        }
        std::sort(sv.begin(), sv.end(), std::greater<>());
        float cumsum = 0.0f;
        float threshold = -1e9f;
        for (auto& [val, idx] : sv) {
            cumsum += val / sum;
            if (cumsum >= p) { threshold = logits[idx]; break; }
        }
        for (auto& l : logits) if (l < threshold) l = -1e9f;
    }

    int categorical(const std::vector<float>& probs) {
        std::discrete_distribution<int> dist(probs.begin(), probs.end());
        return dist(rng);
    }
};

} // namespace gpt
