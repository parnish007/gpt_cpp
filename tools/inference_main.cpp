#include "core/model.h"
#include "core/config.h"
#include "training/dataset.h"
#include "inference/engine.h"
#include <iostream>
#include <string>
#include <stdexcept>

using namespace gpt;

void print_usage(const char* prog) {
    std::cerr << "Usage:\n"
              << "  " << prog << " [options]\n\n"
              << "Options:\n"
              << "  --model  <path>   Path to model .bin  (default: checkpoints/model.bin)\n"
              << "  --vocab  <path>   Path to vocab .bin  (default: checkpoints/vocab.bin)\n"
              << "  --prompt <text>   Seed prompt         (default: interactive mode)\n"
              << "  --tokens <n>      Max new tokens      (default: 200)\n"
              << "  --temp   <f>      Temperature         (default: 0.8)\n"
              << "  --top_k  <n>      Top-K               (default: 40)\n"
              << "  --top_p  <f>      Top-P               (default: 0.9)\n"
              << "  --greedy          Use greedy decoding\n"
              << "  --repl            Interactive REPL mode\n\n";
}

int main(int argc, char** argv) {
    InferenceConfig cfg;
    bool repl_mode = false;

    // ── Argument parsing ──────────────────────────────────────────────────────
    for (int i = 1; i < argc; ++i) {
        std::string arg = argv[i];
        if      (arg == "--model"  && i+1 < argc) cfg.model_path  = argv[++i];
        else if (arg == "--vocab"  && i+1 < argc) cfg.vocab_path  = argv[++i];
        else if (arg == "--prompt" && i+1 < argc) cfg.prompt      = argv[++i];
        else if (arg == "--tokens" && i+1 < argc) cfg.max_tokens  = std::stoul(argv[++i]);
        else if (arg == "--temp"   && i+1 < argc) cfg.temperature = std::stof(argv[++i]);
        else if (arg == "--top_k"  && i+1 < argc) cfg.top_k       = std::stoi(argv[++i]);
        else if (arg == "--top_p"  && i+1 < argc) cfg.top_p       = std::stof(argv[++i]);
        else if (arg == "--greedy") cfg.temperature = 0.0f;  // signals greedy
        else if (arg == "--repl")   repl_mode = true;
        else if (arg == "--help")   { print_usage(argv[0]); return 0; }
        else { std::cerr << "Unknown argument: " << arg << "\n"; print_usage(argv[0]); return 1; }
    }

    // ── Load tokenizer ────────────────────────────────────────────────────────
    CharTokenizer tokenizer;
    tokenizer.load(cfg.vocab_path);
    std::cout << "Loaded vocab: " << tokenizer.vocab_size << " tokens\n";

    // ── Load model ────────────────────────────────────────────────────────────
    ModelConfig model_cfg;
    GPTModel model(model_cfg);  // config will be overwritten by load()
    model.load(cfg.model_path);
    model.set_training(false);
    std::cout << "Loaded model: " << model.num_params() << " params\n";

    // ── Sampling params ───────────────────────────────────────────────────────
    SamplingParams sp;
    if (cfg.temperature == 0.0f) {
        sp.strategy = SamplingStrategy::Greedy;
    } else {
        sp.strategy    = SamplingStrategy::TopKP;
        sp.temperature = cfg.temperature;
        sp.top_k       = cfg.top_k;
        sp.top_p       = cfg.top_p;
    }

    // ── Build inference engine ────────────────────────────────────────────────
    InferenceEngine engine(model, tokenizer);

    if (repl_mode) {
        engine.repl(sp);
    } else {
        std::string output = engine.generate(cfg.prompt, cfg.max_tokens, sp);
        std::cout << "\n" << output << "\n";
    }

    return 0;
}
