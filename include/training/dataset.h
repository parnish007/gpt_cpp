#pragma once
#include <string>
#include <vector>
#include <unordered_map>
#include <fstream>
#include <stdexcept>
#include <algorithm>
#include <random>
#include <iostream>

namespace gpt {

// ─────────────────────────────────────────────────────────────────────────────
//  CharTokenizer — character-level vocabulary
// ─────────────────────────────────────────────────────────────────────────────
class CharTokenizer {
public:
    std::vector<char>              id2ch;
    std::unordered_map<char, int>  ch2id;
    size_t                         vocab_size{0};

    void build(const std::string& text) {
        std::vector<char> chars(text.begin(), text.end());
        std::sort(chars.begin(), chars.end());
        chars.erase(std::unique(chars.begin(), chars.end()), chars.end());
        id2ch    = chars;
        vocab_size = chars.size();
        ch2id.clear();
        for (size_t i = 0; i < chars.size(); ++i) ch2id[chars[i]] = (int)i;
    }

    std::vector<int> encode(const std::string& text) const {
        std::vector<int> out;
        out.reserve(text.size());
        for (char c : text) {
            auto it = ch2id.find(c);
            if (it != ch2id.end()) out.push_back(it->second);
        }
        return out;
    }

    std::string decode(const std::vector<int>& ids) const {
        std::string out;
        out.reserve(ids.size());
        for (int id : ids) {
            if (id >= 0 && id < (int)id2ch.size())
                out += id2ch[id];
        }
        return out;
    }

    // ── Serialization ─────────────────────────────────────────────────────────
    void save(const std::string& path) const {
        std::ofstream f(path, std::ios::binary);
        if (!f) throw std::runtime_error("Cannot write vocab: " + path);
        size_t n = id2ch.size();
        f.write(reinterpret_cast<const char*>(&n), sizeof(n));
        f.write(id2ch.data(), n);
    }

    void load(const std::string& path) {
        std::ifstream f(path, std::ios::binary);
        if (!f) throw std::runtime_error("Cannot read vocab: " + path);
        size_t n;
        f.read(reinterpret_cast<char*>(&n), sizeof(n));
        id2ch.resize(n);
        f.read(id2ch.data(), n);
        vocab_size = n;
        ch2id.clear();
        for (size_t i = 0; i < n; ++i) ch2id[id2ch[i]] = (int)i;
    }
};

// ─────────────────────────────────────────────────────────────────────────────
//  TextDataset — serves random (input, target) batches
// ─────────────────────────────────────────────────────────────────────────────
class TextDataset {
public:
    CharTokenizer       tokenizer;
    std::vector<int>    tokens;
    size_t              seq_len;
    std::mt19937        rng;

    TextDataset(const std::string& text_path, size_t seq_len_)
        : seq_len(seq_len_), rng(std::random_device{}())
    {
        std::ifstream f(text_path);
        if (!f) throw std::runtime_error("Cannot open data file: " + text_path);
        std::string text((std::istreambuf_iterator<char>(f)),
                          std::istreambuf_iterator<char>());
        tokenizer.build(text);
        tokens = tokenizer.encode(text);
        std::cout << "Dataset: " << text.size() << " chars | "
                  << "vocab: " << tokenizer.vocab_size << " | "
                  << "seq_len: " << seq_len << "\n";
    }

    // Fill pre-allocated x (B*T) and y (B*T) integer arrays
    void get_batch(std::vector<int>& x, std::vector<int>& y,
                   size_t B, size_t T) {
        x.resize(B * T); y.resize(B * T);
        size_t max_start = tokens.size() - T - 1;
        std::uniform_int_distribution<size_t> dist(0, max_start);
        for (size_t b = 0; b < B; ++b) {
            size_t start = dist(rng);
            for (size_t t = 0; t < T; ++t) {
                x[b * T + t] = tokens[start + t];
                y[b * T + t] = tokens[start + t + 1];
            }
        }
    }
};

} // namespace gpt
