#include "core/model.h"
#include "core/config.h"
#include "utils/timer.h"
#include <iostream>
#include <iomanip>
#include <vector>
#include <random>
#include <string>
#include <chrono>

using namespace gpt;

struct BenchResult {
    std::string label;
    double      ms_mean;
    double      ms_std;
    double      tokens_per_sec;
};

BenchResult bench_forward(GPTModel& model,
                           size_t B, size_t T, size_t runs = 20)
{
    model.set_training(false);
    std::mt19937 rng(42);
    std::uniform_int_distribution<int> dist(0, (int)model.cfg.vocab_size - 1);

    std::vector<int> idx(B * T);
    for (auto& v : idx) v = dist(rng);

    // warmup
    for (int i = 0; i < 3; ++i) model.forward(idx.data(), B, T);

    std::vector<double> times;
    times.reserve(runs);
    for (size_t r = 0; r < runs; ++r) {
        auto t0 = std::chrono::steady_clock::now();
        model.forward(idx.data(), B, T);
        double ms = std::chrono::duration<double, std::milli>(
            std::chrono::steady_clock::now() - t0).count();
        times.push_back(ms);
    }

    double mean = 0.0;
    for (double t : times) mean += t;
    mean /= runs;

    double var = 0.0;
    for (double t : times) var += (t - mean) * (t - mean);
    double std_dev = std::sqrt(var / runs);

    double tps = (double)(B * T) / (mean / 1000.0);
    return {"forward B=" + std::to_string(B) + " T=" + std::to_string(T),
            mean, std_dev, tps};
}

BenchResult bench_forward_backward(GPTModel& model,
                                    size_t B, size_t T, size_t runs = 10)
{
    model.set_training(true);
    std::mt19937 rng(7);
    std::uniform_int_distribution<int> dist(0, (int)model.cfg.vocab_size - 1);

    std::vector<int> idx(B * T), tgt(B * T);
    for (auto& v : idx) v = dist(rng);
    for (auto& v : tgt) v = dist(rng);

    // warmup
    for (int i = 0; i < 2; ++i) {
        model.zero_grad();
        auto logits = model.forward(idx.data(), B, T);
        model.loss(logits, tgt.data(), B * T);
        model.backward();
    }

    std::vector<double> times;
    times.reserve(runs);
    for (size_t r = 0; r < runs; ++r) {
        auto t0 = std::chrono::steady_clock::now();
        model.zero_grad();
        auto logits = model.forward(idx.data(), B, T);
        model.loss(logits, tgt.data(), B * T);
        model.backward();
        double ms = std::chrono::duration<double, std::milli>(
            std::chrono::steady_clock::now() - t0).count();
        times.push_back(ms);
    }

    double mean = 0.0;
    for (double t : times) mean += t;
    mean /= runs;

    double var = 0.0;
    for (double t : times) var += (t - mean) * (t - mean);
    double std_dev = std::sqrt(var / runs);

    double tps = (double)(B * T) / (mean / 1000.0);
    return {"fwd+bwd B=" + std::to_string(B) + " T=" + std::to_string(T),
            mean, std_dev, tps};
}

void print_result(const BenchResult& r) {
    std::cout << std::left  << std::setw(32) << r.label
              << std::right << std::fixed << std::setprecision(2)
              << std::setw(10) << r.ms_mean  << " ms"
              << std::setw(10) << r.ms_std   << " ms std"
              << std::setw(14) << (size_t)r.tokens_per_sec << " tok/s\n";
}

int main(int argc, char** argv) {
    std::cout << "=== GPT C++ Benchmark ===\n\n";

    // ── Model configs to benchmark ────────────────────────────────────────────
    struct Cfg { std::string name; ModelConfig mc; };
    std::vector<Cfg> cfgs = {
        {"tiny  (embed=128, layers=4, heads=4)", []{
            ModelConfig c; c.vocab_size=256; c.embed_dim=128;
            c.n_heads=4; c.n_layers=4; c.max_seq_len=128; c.dropout=0; return c;}()},
        {"small (embed=256, layers=6, heads=8)", []{
            ModelConfig c; c.vocab_size=256; c.embed_dim=256;
            c.n_heads=8; c.n_layers=6; c.max_seq_len=256; c.dropout=0; return c;}()},
    };

    std::cout << std::left  << std::setw(32) << "Config"
              << std::right << std::setw(12) << "Mean"
              << std::setw(14) << "Std"
              << std::setw(14) << "Throughput\n"
              << std::string(72, '-') << "\n";

    for (auto& [name, mc] : cfgs) {
        std::cout << "\n[" << name << "]\n";
        GPTModel model(mc);
        size_t params = model.num_params();
        std::cout << "  Parameters: " << params / 1000 << "K\n";

        print_result(bench_forward(model, 1, mc.max_seq_len));
        print_result(bench_forward(model, 4, mc.max_seq_len));
        print_result(bench_forward_backward(model, 4, mc.max_seq_len));
    }

    std::cout << "\nDone.\n";
    return 0;
}
