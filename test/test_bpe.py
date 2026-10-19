import tokenizer
from pathlib import Path

TRAIN_PATH = [f"data/doc_{i:03d}.txt" for i in range(1, 101)]


if __name__ == "__main__":
    bpe = tokenizer.ByteBPE()
    texts = [Path(path).read_text(encoding="utf-8") for path in TRAIN_PATH]
    bpe.train(texts, 8192)
    bpe.store_tokens()
