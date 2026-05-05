#!/usr/bin/env python3
"""
prepare_data.py — prepare training data for gpt_cpp

Options:
  --source tiny_shakespeare   Download Tiny Shakespeare (~1MB)  [default]
  --source custom <file>      Use your own .txt file
  --out    data/input.txt     Output path

Usage:
  python scripts/prepare_data.py
  python scripts/prepare_data.py --source custom myfile.txt --out data/input.txt
"""
import os
import sys
import argparse
import urllib.request

TINY_SHAKESPEARE_URL = (
    "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/"
    "tinyshakespeare/input.txt"
)

def download_shakespeare(out_path: str):
    print(f"Downloading Tiny Shakespeare → {out_path}")
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    urllib.request.urlretrieve(TINY_SHAKESPEARE_URL, out_path)
    size = os.path.getsize(out_path)
    print(f"Done. {size:,} bytes saved.")

def copy_custom(src: str, out_path: str):
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    with open(src, "r", encoding="utf-8") as f:
        text = f.read()
    with open(out_path, "w", encoding="utf-8") as f:
        f.write(text)
    unique_chars = sorted(set(text))
    print(f"Copied {len(text):,} chars | vocab size: {len(unique_chars)}")
    print(f"Saved to {out_path}")

def print_stats(path: str):
    with open(path, "r", encoding="utf-8") as f:
        text = f.read()
    unique = sorted(set(text))
    print(f"\n─── Data Stats ───────────────────")
    print(f"  Characters : {len(text):,}")
    print(f"  Vocab size : {len(unique)}")
    print(f"  First 200  : {repr(text[:200])}")
    print(f"──────────────────────────────────")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", default="tiny_shakespeare",
                        choices=["tiny_shakespeare", "custom"])
    parser.add_argument("--file",   default="",
                        help="Path to custom text file (when --source custom)")
    parser.add_argument("--out",    default="data/input.txt")
    args = parser.parse_args()

    if args.source == "tiny_shakespeare":
        download_shakespeare(args.out)
    elif args.source == "custom":
        if not args.file:
            sys.exit("Error: --file required when --source custom")
        copy_custom(args.file, args.out)

    print_stats(args.out)

if __name__ == "__main__":
    main()
