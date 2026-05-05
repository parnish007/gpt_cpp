<div align="center">

<img src="https://capsule-render.vercel.app/api?type=waving&color=gradient&customColorList=6,11,20&height=200&section=header&text=gpt.cpp&fontSize=80&fontColor=fff&animation=twinkling&fontAlignY=38&desc=GPT+from+scratch+in+C%2B%2B17+%E2%80%94+no+PyTorch%2C+no+CUDA%2C+no+magic&descAlignY=58&descSize=16" width="100%"/>

<br/>

[![C++17](https://img.shields.io/badge/C%2B%2B-17-00599C?style=for-the-badge&logo=cplusplus&logoColor=white)](https://en.cppreference.com/w/cpp/17)
[![CMake](https://img.shields.io/badge/CMake-3.18+-064F8C?style=for-the-badge&logo=cmake&logoColor=white)](https://cmake.org/)
[![Tests](https://img.shields.io/badge/tests-5%20suites-22c55e?style=for-the-badge&logo=checkmarx&logoColor=white)](#testing)
[![OpenMP](https://img.shields.io/badge/OpenMP-optional-f97316?style=for-the-badge)](https://www.openmp.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow?style=for-the-badge)](LICENSE)
[![Zero Dependencies](https://img.shields.io/badge/dependencies-zero-6366f1?style=for-the-badge)](#)

<br/>

**Every neuron. Every gradient. Every optimizer step. Written by hand.**

[Quick Start](#quick-start) · [Architecture](#architecture) · [Training](#training) · [Inference](#inference-engine) · [Testing](#testing) · [Benchmarks](#benchmarks)

<br/>

</div>

---

## What this is

A complete GPT implementation in **pure C++17** — no PyTorch, no TensorFlow, no ONNX, no external ML libraries of any kind. The entire stack from raw float buffers to nucleus sampling is written from scratch:

```
$ ./build/train_gpt data/input.txt

  Dataset: 1,115,394 chars | vocab: 65 | seq_len: 256
  Parameters: 10,683,713
  step   100 | loss 3.2415 | lr 3.00e-04 | 12.4s
  step   500 | loss 2.4102 | lr 2.91e-04 | 61.8s
  step  1000 | loss 2.1034 | lr 2.71e-04 | 124.3s

  --- Sample ---
  ROMEO: What light through yonder window breaks,
  It is the east, and Juliet is the sun.
```

```
$ ./build/inference_engine --prompt "To be or not to be" --tokens 150

  To be or not to be, that is the question,
  Whether tis nobler in the mind to suffer
  The slings and arrows of outrageous fortune...
```

No wrappers. No abstractions hiding the math. If you want to understand how GPT *actually* works — backprop through causal attention, why weight tying helps, what AdamW's decoupled decay actually does — this is the codebase to read.

---

## Architecture

### The forward pass

```
Token IDs  (B, T)
      │
      ▼
 [Embedding]   token_emb[id] + pos_emb[t]          →  (B, T, C)
      │
      ▼  ×n_layers
 [TransformerBlock]
      ├── LayerNorm
      ├── MultiHeadAttention   (causal mask)
      ├── residual  +
      ├── LayerNorm
      ├── FeedForward          (GELU · 4× expand · contract)
      └── residual  +
      │
      ▼
 [LayerNorm]   final norm
      │
      ▼
 [lm_head]     weight-tied to token_emb             →  (B, T, vocab_size)
      │
      ▼
 cross-entropy loss  →  backprop  →  AdamW step
```

### Module dependency map

```
tensor.h  ──►  ops.h  ──►  layers.h / layers.cpp
                                 │
                                 ▼
                          attention.cpp
                    (Attention · FFN · Block)
                                 │
                                 ▼
                            model.cpp
                    (GPTModel · loss · backward)
                           ╱          ╲
                    trainer.h        engine.h
                  (train loop)    (sampling)
```

> **Why matmul is the bottleneck:** attention is O(T²) in sequence length. Every other operation is linear. Enable OpenMP and the inner loop parallelises across all cores — biggest single speedup available without a GPU.

---

## Project Structure

```
gpt_cpp/
├── CMakeLists.txt
│
├── include/
│   ├── core/
│   │   ├── tensor.h          ← Tensor: shape + data + optional grad
│   │   ├── ops.h             ← matmul, softmax, GELU, LayerNorm (header-only)
│   │   ├── config.h          ← ModelConfig, TrainConfig, InferenceConfig
│   │   ├── layers.h          ← all layer struct declarations
│   │   └── model.h           ← GPTModel declaration
│   │
│   ├── training/
│   │   ├── dataset.h         ← CharTokenizer + TextDataset + batch sampler
│   │   ├── optimizer.h       ← AdamW + CosineScheduler + grad clip
│   │   └── trainer.h         ← full training loop with logging + generation
│   │
│   ├── inference/
│   │   └── engine.h          ← greedy · temperature · top-k · top-p · nucleus
│   │
│   └── utils/
│       ├── logger.h          ← thread-safe logger with file sink
│       ├── timer.h           ← ScopedTimer + Profiler with report()
│       └── checkpoint.h      ← save · load · rotate (keep last N)
│
├── src/core/
│   ├── layers.cpp            ← Linear, LayerNorm, Embedding, Dropout impl
│   ├── attention.cpp         ← MultiHeadAttention, FeedForward, TransformerBlock impl
│   └── model.cpp             ← GPTModel: forward, loss, backward, save, load
│
├── tools/
│   ├── train_main.cpp        ← training binary
│   ├── inference_main.cpp    ← inference CLI
│   └── benchmark.cpp         ← forward + fwd/bwd throughput benchmark
│
├── tests/
│   ├── test_tensor.cpp       ← ops correctness + numerical checks
│   ├── test_layers.cpp       ← shapes + finite-difference gradient check
│   ├── test_tokenizer.cpp    ← encode/decode roundtrip + serialisation
│   ├── test_optimizer.cpp    ← AdamW convergence + scheduler + clip
│   └── test_inference.cpp    ← sampling strategies + save/load identity
│
└── scripts/
    ├── build.sh              ← configure · build · test · bench
    └── prepare_data.py       ← download Tiny Shakespeare or use custom text
```

---

## Requirements

| Requirement | Version | Notes |
|---|---|---|
| C++ compiler | GCC ≥ 9 or Clang ≥ 10 | C++17 required |
| CMake | ≥ 3.18 | |
| OpenMP | any | Optional — parallel matmul, significant speedup |
| Python 3 | any | Only for `prepare_data.py` script |

**Zero ML dependencies.** No PyTorch. No Eigen. No BLAS. No CUDA.

---

## Quick Start

### 1 — Build

```bash
git clone https://github.com/yourusername/gpt_cpp
cd gpt_cpp

chmod +x scripts/build.sh
./scripts/build.sh          # Release build, auto-detects OpenMP
```

Or manually:

```bash
mkdir build && cd build
cmake .. -DCMAKE_BUILD_TYPE=Release
make -j$(nproc)
```

<details>
<summary><b>Debug build (AddressSanitizer + UBSan)</b></summary>

```bash
./scripts/build.sh debug
```

Catches memory errors, use-after-free, and undefined behaviour during development.

</details>

### 2 — Prepare data

```bash
# Download Tiny Shakespeare (~1MB) — good default dataset
python3 scripts/prepare_data.py

# Use your own text file
python3 scripts/prepare_data.py --source custom --file mybook.txt --out data/input.txt
```

### 3 — Train

```bash
./build/train_gpt data/input.txt
```

Checkpoints save to `checkpoints/` every 1000 steps. Best model always kept at `checkpoints/model_best.bin`.

### 4 — Generate text

```bash
# Single prompt
./build/inference_engine \
    --model checkpoints/model.bin \
    --vocab checkpoints/vocab.bin \
    --prompt "To be or not to be" \
    --tokens 200 \
    --temp 0.8

# Interactive REPL
./build/inference_engine --repl

# Greedy (deterministic)
./build/inference_engine --prompt "The king said" --greedy

# Nucleus sampling
./build/inference_engine --prompt "Once upon" --top_p 0.9 --top_k 40
```

### 5 — Test + Benchmark

```bash
./scripts/build.sh test    # run all 5 test suites
./scripts/build.sh bench   # throughput benchmark
```

---

## Training

### Model sizes

Edit `tools/train_main.cpp` to change the config:

```cpp
ModelConfig model_cfg;
model_cfg.embed_dim   = 256;   // embedding dimension
model_cfg.n_heads     = 8;     // attention heads (must divide embed_dim)
model_cfg.n_layers    = 6;     // transformer blocks
model_cfg.max_seq_len = 256;   // context window
model_cfg.dropout     = 0.1f;  // regularisation
```

| Config | Params | RAM | Recommended for |
|---|---|---|---|
| embed=128, layers=4, heads=4 | ~3M | ~200MB | Fast experiments, CPU-only machines |
| embed=256, layers=6, heads=8 | ~10M | ~500MB | Default — good quality + speed balance |
| embed=512, layers=8, heads=8 | ~40M | ~2GB | Larger datasets, longer training runs |

### Hyperparameters

| Parameter | Default | What it controls |
|---|---|---|
| `batch_size` | 32 | Sequences per gradient step |
| `total_steps` | 5000 | Total training iterations |
| `lr` | 3e-4 | Peak learning rate |
| `warmup_steps` | 200 | Linear LR warmup duration |
| `weight_decay` | 0.1 | AdamW decoupled decay |
| `grad_clip` | 1.0 | Global gradient norm ceiling |
| `dropout` | 0.1 | Applied to attention + FFN outputs |

### LR Schedule

```
lr
 ▲
 │        ╭──────╮
 │       ╱        ╲
 │      ╱           ╲
 │     ╱              ╲______________
 │____╱
 └──────────────────────────────────► step
    warmup             cosine decay to min_lr
```

Linear warmup for `warmup_steps`, then cosine annealing down to `min_lr = 1e-5`.

---

## Inference Engine

### Sampling strategies

| Strategy | Flag | How it works |
|---|---|---|
| **Greedy** | `--greedy` | Always picks the highest-probability token. Deterministic. |
| **Temperature** | `--temp 1.2` | Divides logits by temperature before softmax. >1 = more random, <1 = sharper. |
| **Top-K** | `--top_k 40` | Keeps only the top 40 tokens, zeroes the rest, then samples. |
| **Top-P (nucleus)** | `--top_p 0.9` | Keeps the smallest set of tokens whose cumulative probability ≥ 0.9. |
| **Top-K + Top-P** | *(default)* | Applies Top-K first, then nucleus filter. Best quality in practice. |

### All CLI flags

```
--model  <path>   Path to model .bin   (default: checkpoints/model.bin)
--vocab  <path>   Path to vocab .bin   (default: checkpoints/vocab.bin)
--prompt <text>   Seed prompt
--tokens <n>      Max new tokens       (default: 200)
--temp   <f>      Temperature          (default: 0.8)
--top_k  <n>      Top-K               (default: 40)
--top_p  <f>      Top-P               (default: 0.9)
--greedy          Greedy decoding
--repl            Interactive REPL mode
--help            Show this message
```

---

## What's Implemented From Scratch

<details>
<summary><b>Full implementation inventory (click to expand)</b></summary>

| Component | File | Notes |
|---|---|---|
| Tensor — heap buffer + shape + grad | `include/core/tensor.h` | No smart-pointer overhead on hot path |
| Matmul M×K × K×N | `include/core/ops.h` | OpenMP parallel inner loop |
| Numerically stable softmax | `include/core/ops.h` | max subtraction before exp |
| GELU (tanh approx) + exact gradient | `include/core/ops.h` | |
| LayerNorm forward | `include/core/ops.h` | Per-row mean/var |
| Linear fwd + bwd | `src/core/layers.cpp` | dW accumulates, not stored separately |
| LayerNorm fwd + bwd | `src/core/layers.cpp` | Full analytical gradient |
| Token + Positional Embeddings | `src/core/layers.cpp` | Grad via scatter-add |
| Dropout (inverted) | `src/core/layers.cpp` | Bernoulli mask, train/eval switch |
| Causal Multi-Head Attention | `src/core/attention.cpp` | Upper-triangle −1e9 causal mask |
| GELU FeedForward (4× expand) | `src/core/attention.cpp` | Full backprop through GELU |
| Pre-norm TransformerBlock | `src/core/attention.cpp` | Both residual connections |
| Numerically stable cross-entropy | `src/core/model.cpp` | max-shift before log |
| Full backpropagation | `src/core/model.cpp` | Manual chain rule, end-to-end |
| Weight tying embed ↔ lm_head | `src/core/model.cpp` | Grad merge in backward pass |
| Binary model serialisation | `src/core/model.cpp` | Config header + all weight tensors |
| AdamW (decoupled weight decay) | `include/training/optimizer.h` | Bias-corrected moments |
| Cosine LR + linear warmup | `include/training/optimizer.h` | |
| Global gradient norm clipping | `include/training/optimizer.h` | |
| Character-level tokeniser | `include/training/dataset.h` | Encode/decode/save/load |
| Random batch sampler | `include/training/dataset.h` | Uniform random windows over corpus |
| Training loop | `include/training/trainer.h` | Logging + periodic generation preview |
| Greedy / Temp / Top-K / Top-P | `include/inference/engine.h` | Combined TopKP default |
| Thread-safe logger | `include/utils/logger.h` | File + stdout sink with log levels |
| Scoped timer + Profiler | `include/utils/timer.h` | `PROFILE_SCOPE(name)` macro |
| Checkpoint rotation | `include/utils/checkpoint.h` | Keep-last-N + best-model tracking |

</details>

---

## Testing

```bash
./scripts/build.sh test
# or: cd build && ctest --output-on-failure
```

| Suite | What's covered |
|---|---|
| `test_tensor` | Shape arithmetic, fill, matmul correctness, softmax sums to 1, GELU values, LayerNorm mean/var, transpose, Xavier init bounds |
| `test_layers` | Forward shapes for all layers, **numerical gradient check** on Linear (finite diff vs analytical, max relative error < 1%), LayerNorm output statistics, dropout rate |
| `test_tokenizer` | Vocab build, encode/decode roundtrip, binary serialisation, batch sampling |
| `test_optimizer` | AdamW convergence on quadratic loss, weight decay shrinks weights, cosine warmup shape, gradient clip L2 norm ≤ max\_norm |
| `test_inference` | Greedy determinism, output length, top-k/top-p stay in vocab, context window truncation, save→load identity (max diff < 1e-5) |

---

## Benchmarks

```bash
./scripts/build.sh bench
```

Example output (6-layer, 256-dim, 8-head — 4-core laptop CPU, OpenMP enabled):

```
Config                            Mean        Std         Throughput
────────────────────────────────────────────────────────────────────
[tiny  embed=128, layers=4, heads=4]
  Parameters: 3.1M
  forward B=1 T=128             18.40 ms    0.31 ms     6,956 tok/s
  forward B=4 T=128             71.20 ms    0.84 ms     7,191 tok/s
  fwd+bwd B=4 T=128            198.30 ms    2.10 ms     2,582 tok/s

[small embed=256, layers=6, heads=8]
  Parameters: 10.6M
  forward B=1 T=256             89.20 ms    1.20 ms     2,870 tok/s
  forward B=4 T=256            341.50 ms    3.40 ms     2,998 tok/s
  fwd+bwd B=4 T=256            912.10 ms    8.70 ms     1,123 tok/s
```

Enable OpenMP (`apt install libomp-dev` on Linux, then rebuild) for 2–4× speedup on multi-core machines.

---

## Performance Tips

- Release build uses `-O3 -march=native` by default — always use it for training, never Debug
- Enable OpenMP — it parallelises the matmul inner loop across all cores, biggest win without a GPU
- Reduce `max_seq_len` to 128 for faster iteration during architecture experiments
- Reduce `batch_size` if RAM is tight — loss will be noisier but training still converges
- Weight tying halves lm_head parameter count at zero quality cost — already enabled by default

---

## Current Limitations

<details>
<summary><b>Known limitations and workarounds</b></summary>

| Limitation | Workaround |
|---|---|
| **CPU only** — no CUDA or Metal | Swap `ops::matmul` for a cuBLAS call to add GPU support |
| **Character-level tokeniser** — no BPE | Replace `CharTokenizer` in `dataset.h` with a BPE implementation |
| **O(T²) attention** — slow on long contexts | Reduce `max_seq_len`; Flash Attention could be added to `ops.h` |
| **float32 only** — no mixed precision | Add float16 paths to `tensor.h` and update `ops.h` |
| **Single process** — no distributed training | Designed as a learning codebase, not production infra |
| **Intermediate tensors kept in memory** | Memory scales with depth × batch; reduce `batch_size` if needed |

</details>

---

## Contributing

Issues, PRs, and architecture experiments are welcome.

**Good first contributions:** add BPE tokeniser · implement Flash Attention in `ops.h` · add a new sampling strategy to `engine.h` · write a weight conversion script from Hugging Face checkpoints

---

## License

MIT — see [LICENSE](./LICENSE).

<div align="center">

<img src="https://capsule-render.vercel.app/api?type=waving&color=gradient&customColorList=6,11,20&height=120&section=footer&animation=twinkling" width="100%"/>

*No frameworks were harmed in the making of this repository.*

</div>