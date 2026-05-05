#pragma once
#include "core/model.h"
#include "core/config.h"
#include "training/optimizer.h"
#include "training/dataset.h"
#include <chrono>
#include <iostream>
#include <filesystem>
#include <iomanip>

namespace gpt {

class Trainer {
public:
    GPTModel&      model;
    TextDataset&   dataset;
    TrainConfig    cfg;

    Trainer(GPTModel& model_, TextDataset& dataset_, const TrainConfig& cfg_)
        : model(model_), dataset(dataset_), cfg(cfg_) {}

    void train() {
        namespace fs = std::filesystem;
        fs::create_directories(cfg.ckpt_dir);

        auto params = model.parameters();
        auto sizes  = model.param_sizes();
        AdamW optimizer(params, sizes, cfg.lr, cfg.beta1, cfg.beta2,
                        cfg.eps, cfg.weight_decay);
        CosineScheduler scheduler(optimizer, cfg.warmup_steps, cfg.total_steps);

        std::vector<int> x_batch, y_batch;
        std::vector<float> loss_history;
        loss_history.reserve(cfg.total_steps);

        auto t0 = std::chrono::steady_clock::now();

        for (size_t step = 1; step <= cfg.total_steps; ++step) {
            model.set_training(true);
            model.zero_grad();

            // ── Forward ──────────────────────────────────────────────────────
            dataset.get_batch(x_batch, y_batch, cfg.batch_size, model.cfg.max_seq_len);

            Tensor logits = model.forward(x_batch.data(),
                                          cfg.batch_size,
                                          model.cfg.max_seq_len);
            float  loss   = model.loss(logits, y_batch.data(),
                                       cfg.batch_size * model.cfg.max_seq_len);

            // ── Backward ─────────────────────────────────────────────────────
            model.backward();

            // ── Gradient clip ─────────────────────────────────────────────────
            if (cfg.grad_clip > 0.0f)
                clip_grad_norm(params, sizes, cfg.grad_clip);

            // ── Optimizer step ────────────────────────────────────────────────
            optimizer.step();
            float lr = scheduler.step();

            loss_history.push_back(loss);

            // ── Logging ───────────────────────────────────────────────────────
            if (step % cfg.eval_every == 0) {
                auto now     = std::chrono::steady_clock::now();
                double elapsed = std::chrono::duration<double>(now - t0).count();
                float avg_loss = 0.0f;
                size_t window  = std::min(cfg.eval_every, step);
                for (size_t i = loss_history.size() - window; i < loss_history.size(); ++i)
                    avg_loss += loss_history[i];
                avg_loss /= (float)window;

                std::cout << std::fixed << std::setprecision(4)
                          << "step " << std::setw(6) << step
                          << " | loss " << avg_loss
                          << " | lr "   << std::scientific << lr
                          << std::fixed
                          << " | " << std::setprecision(1) << elapsed << "s\n";
            }

            // ── Generation preview ────────────────────────────────────────────
            if (cfg.gen_every > 0 && step % cfg.gen_every == 0 && !cfg.seed_text.empty()) {
                generate_sample(cfg.seed_text, 100, 0.8f);
            }

            // ── Checkpoint ───────────────────────────────────────────────────
            if (step % (cfg.eval_every * 10) == 0) {
                std::string ckpt = cfg.ckpt_dir + "/model_step" + std::to_string(step) + ".bin";
                model.save(ckpt);
            }
        }

        // Final save
        model.save(cfg.ckpt_dir + "/model.bin");
        dataset.tokenizer.save(cfg.ckpt_dir + "/vocab.bin");
        std::cout << "\nTraining complete. Final loss: "
                  << loss_history.back() << "\n";
    }

private:
    void generate_sample(const std::string& seed, size_t max_tokens, float temp) {
        model.set_training(false);
        auto ids = dataset.tokenizer.encode(seed);
        size_t max_len = model.cfg.max_seq_len;

        std::mt19937 rng(42);
        for (size_t i = 0; i < max_tokens; ++i) {
            size_t T   = std::min(ids.size(), max_len);
            auto   ctx = std::vector<int>(ids.end() - T, ids.end());
            Tensor logits = model.forward(ctx.data(), 1, T);

            // last timestep logits
            size_t V = model.cfg.vocab_size;
            std::vector<float> prob(logits.raw() + (T-1)*V,
                                    logits.raw() + T*V);
            // temperature + softmax
            float mx = *std::max_element(prob.begin(), prob.end());
            float sum = 0.0f;
            for (auto& p : prob) { p = std::exp((p - mx) / temp); sum += p; }
            for (auto& p : prob) p /= sum;
            // sample
            std::discrete_distribution<int> dist(prob.begin(), prob.end());
            ids.push_back(dist(rng));
        }

        std::cout << "\n--- Sample ---\n"
                  << dataset.tokenizer.decode(ids)
                  << "\n--------------\n\n";
        model.set_training(true);
    }
};

} // namespace gpt
