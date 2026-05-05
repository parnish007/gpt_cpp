#include "training/dataset.h"
#include <cassert>
#include <iostream>
#include <cstdio>

using namespace gpt;

void test_build_vocab() {
    CharTokenizer tok;
    tok.build("hello world");
    // unique sorted chars: ' ', 'd', 'e', 'h', 'l', 'o', 'r', 'w'
    assert(tok.vocab_size == 8);
    std::cout << "[PASS] build_vocab (size=" << tok.vocab_size << ")\n";
}

void test_encode_decode_roundtrip() {
    CharTokenizer tok;
    std::string text = "Hello, World! 123";
    tok.build(text);
    auto ids = tok.encode(text);
    assert(ids.size() == text.size());
    std::string recovered = tok.decode(ids);
    assert(recovered == text);
    std::cout << "[PASS] encode_decode_roundtrip\n";
}

void test_vocab_serialization() {
    CharTokenizer tok;
    tok.build("abcdefghijklmnopqrstuvwxyz");
    const char* path = "/tmp/test_vocab.bin";
    tok.save(path);

    CharTokenizer tok2;
    tok2.load(path);
    assert(tok2.vocab_size == tok.vocab_size);
    for (size_t i = 0; i < tok.vocab_size; ++i)
        assert(tok2.id2ch[i] == tok.id2ch[i]);
    std::remove(path);
    std::cout << "[PASS] vocab_serialization\n";
}

void test_dataset_batch() {
    // Write a temp text file
    const char* path = "/tmp/test_data.txt";
    {
        std::ofstream f(path);
        for (int i = 0; i < 1000; ++i) f << "abcdefghijklmnopqrstuvwxyz ";
    }

    TextDataset ds(path, 32);
    assert(ds.tokenizer.vocab_size == 27);  // 26 + space

    std::vector<int> x, y;
    ds.get_batch(x, y, 4, 32);
    assert(x.size() == 4 * 32);
    assert(y.size() == 4 * 32);

    // Targets should be one step ahead of inputs
    // (not guaranteed for random batches across different start positions,
    //  but within one sequence they should match)
    std::remove(path);
    std::cout << "[PASS] dataset_batch\n";
}

int main() {
    std::cout << "=== Tokenizer & Dataset Tests ===\n";
    test_build_vocab();
    test_encode_decode_roundtrip();
    test_vocab_serialization();
    test_dataset_batch();
    std::cout << "\nAll tokenizer tests passed.\n";
    return 0;
}
