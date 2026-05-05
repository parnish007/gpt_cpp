#pragma once
#include "core/model.h"
#include "training/dataset.h"
#include "utils/logger.h"
#include <string>
#include <filesystem>
#include <vector>
#include <algorithm>
#include <regex>

namespace gpt {

// ─────────────────────────────────────────────────────────────────────────────
//  CheckpointManager
//  Handles saving / loading / rotating model checkpoints
// ─────────────────────────────────────────────────────────────────────────────
class CheckpointManager {
public:
    std::string ckpt_dir;
    size_t      keep_last;   // how many step-checkpoints to retain

    CheckpointManager(const std::string& dir, size_t keep = 3)
        : ckpt_dir(dir), keep_last(keep)
    {
        std::filesystem::create_directories(dir);
    }

    // ── Save step checkpoint ──────────────────────────────────────────────────
    void save_step(GPTModel& model, CharTokenizer& tok,
                   size_t step, float loss)
    {
        std::string model_path = ckpt_dir + "/model_step" +
                                 std::to_string(step) + ".bin";
        std::string vocab_path = ckpt_dir + "/vocab.bin";
        std::string meta_path  = ckpt_dir + "/meta.txt";

        model.save(model_path);
        tok.save(vocab_path);

        // Write metadata
        std::ofstream f(meta_path);
        f << "step=" << step << "\n"
          << "loss=" << loss << "\n"
          << "model=" << model_path << "\n"
          << "vocab=" << vocab_path << "\n";

        LOG_INFO("Checkpoint saved: step=", step, " loss=", loss);
        rotate_checkpoints();
    }

    // ── Save best model (by loss) ─────────────────────────────────────────────
    void save_best(GPTModel& model, CharTokenizer& tok,
                   size_t step, float loss)
    {
        std::string model_path = ckpt_dir + "/model_best.bin";
        std::string info_path  = ckpt_dir + "/best_info.txt";

        // Check if this is actually better
        float prev_best = read_best_loss();
        if (loss >= prev_best) return;

        model.save(model_path);
        tok.save(ckpt_dir + "/vocab.bin");

        std::ofstream f(info_path);
        f << "step=" << step << "\nloss=" << loss << "\n";
        LOG_INFO("New best model: step=", step, " loss=", loss,
                 " (prev=", prev_best, ")");
    }

    // ── Load latest checkpoint ────────────────────────────────────────────────
    bool load_latest(GPTModel& model, CharTokenizer& tok) {
        auto checkpoints = list_checkpoints();
        if (checkpoints.empty()) return false;
        // Sort by step number (highest last)
        std::string latest = checkpoints.back();
        model.load(latest);
        tok.load(ckpt_dir + "/vocab.bin");
        LOG_INFO("Resumed from: ", latest);
        return true;
    }

    // ── Load best model ───────────────────────────────────────────────────────
    bool load_best(GPTModel& model, CharTokenizer& tok) {
        std::string path = ckpt_dir + "/model_best.bin";
        if (!std::filesystem::exists(path)) return false;
        model.load(path);
        tok.load(ckpt_dir + "/vocab.bin");
        LOG_INFO("Loaded best model from: ", path);
        return true;
    }

private:
    std::vector<std::string> list_checkpoints() {
        std::vector<std::string> result;
        std::regex  pattern("model_step(\\d+)\\.bin");
        for (auto& entry : std::filesystem::directory_iterator(ckpt_dir)) {
            std::string fname = entry.path().filename().string();
            std::smatch m;
            if (std::regex_match(fname, m, pattern))
                result.push_back(entry.path().string());
        }
        // Sort lexicographically (step numbers are zero-padded-ish)
        std::sort(result.begin(), result.end());
        return result;
    }

    void rotate_checkpoints() {
        auto checkpoints = list_checkpoints();
        while (checkpoints.size() > keep_last) {
            std::filesystem::remove(checkpoints.front());
            LOG_INFO("Removed old checkpoint: ", checkpoints.front());
            checkpoints.erase(checkpoints.begin());
        }
    }

    float read_best_loss() {
        std::string path = ckpt_dir + "/best_info.txt";
        if (!std::filesystem::exists(path)) return 1e9f;
        std::ifstream f(path);
        std::string line;
        while (std::getline(f, line)) {
            if (line.rfind("loss=", 0) == 0)
                return std::stof(line.substr(5));
        }
        return 1e9f;
    }
};

} // namespace gpt
