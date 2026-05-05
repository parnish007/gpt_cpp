#include "core/model.h"
#include "core/config.h"
#include "training/dataset.h"
#include "training/trainer.h"
#include <iostream>
#include <string>

using namespace gpt;

int main(int argc, char** argv) {
    // ── Config ────────────────────────────────────────────────────────────────
    TrainConfig train_cfg;
    train_cfg.data_path    = (argc > 1) ? argv[1] : "data/input.txt";
    train_cfg.ckpt_dir     = "checkpoints";
    train_cfg.batch_size   = 32;
    train_cfg.total_steps  = 5000;
    train_cfg.eval_every   = 100;
    train_cfg.gen_every    = 500;
    train_cfg.lr           = 3e-4f;
    train_cfg.weight_decay = 0.1f;
    train_cfg.warmup_steps = 200;
    train_cfg.grad_clip    = 1.0f;
    train_cfg.seed_text    = "The ";

    // ── Dataset ───────────────────────────────────────────────────────────────
    TextDataset dataset(train_cfg.data_path, 256);

    // ── Model config ──────────────────────────────────────────────────────────
    ModelConfig model_cfg;
    model_cfg.vocab_size  = dataset.tokenizer.vocab_size;
    model_cfg.embed_dim   = 256;
    model_cfg.n_heads     = 8;
    model_cfg.n_layers    = 6;
    model_cfg.max_seq_len = 256;
    model_cfg.dropout     = 0.1f;

    // ── Build model ───────────────────────────────────────────────────────────
    GPTModel model(model_cfg);
    std::cout << "Parameters: " << model.num_params() << "\n";

    // ── Train ─────────────────────────────────────────────────────────────────
    Trainer trainer(model, dataset, train_cfg);
    trainer.train();

    return 0;
}
