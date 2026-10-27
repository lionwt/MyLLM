"""Evaluate generation quality, repetition, and training-data memorization."""

import argparse
import json
import random
from collections import Counter
from pathlib import Path

from generate import generate
from train.dataset import sample_windows
from train.device import describe_device, resolve_device
from train.io import load_checkpoint, load_tokenizer
from train.loss import estimate_loss
from train.model import MiniGPT


def _ngrams(ids, n):
    """Return every contiguous token n-gram in ``ids``."""
    return [tuple(ids[i : i + n]) for i in range(len(ids) - n + 1)]


def repetition_stats(ids):
    """Measure token diversity and detect strongly repetitive generations."""
    ids = list(ids)
    total = len(ids)
    longest_run = 0
    run = 0
    prev = None
    for token in ids:
        if token == prev:
            run += 1
        else:
            run = 1
            prev = token
        if run > longest_run:
            longest_run = run

    def distinct(n):
        """Compute the fraction of unique n-grams in this generation."""
        grams = _ngrams(ids, n)
        if not grams:
            return 0.0
        return len(set(grams)) / len(grams)

    four = _ngrams(ids, 4)
    if four:
        counts = Counter(four)
        repeated_frac = sum(c - 1 for c in counts.values() if c > 1) / len(four)
    else:
        repeated_frac = 0.0
    degenerate = repeated_frac > 0.3 or longest_run >= 8
    return {
        "tokens": total,
        "distinct_1": distinct(1),
        "distinct_2": distinct(2),
        "distinct_3": distinct(3),
        "repeated_4gram_frac": repeated_frac,
        "longest_run": longest_run,
        "is_degenerate": degenerate,
    }


def longest_copy(text, corpus):
    """Find the longest contiguous substring of ``text`` present in ``corpus``."""
    if not text or not corpus:
        return 0, ""
    n = len(text)
    left = 0
    best_len = 0
    best = ""
    for right in range(1, n + 1):
        while left < right and text[left:right] not in corpus:
            left += 1
        length = right - left
        if left < right and length > best_len:
            best_len = length
            best = text[left:right]
    return best_len, best


class MemorizationIndex:
    """Index training and validation text for generation-overlap measurements."""

    def __init__(self, train_texts, val_texts, tok):
        """Build character corpora and token 8-gram lookup sets."""
        self.train_text = "".join(train_texts)
        self.val_text = "".join(val_texts)
        self.train_8grams = self._grams(train_texts, tok)
        self.val_8grams = self._grams(val_texts, tok)

    @staticmethod
    def _grams(texts, tok):
        """Collect all token 8-grams from a sequence of documents."""
        grams = set()
        for text in texts:
            ids = tok.encode(text)
            grams.update(_ngrams(ids, 8))
        return grams

    def score(self, text, ids, split="train"):
        """Score generated text against either the train or validation corpus."""
        corpus = self.train_text if split == "train" else self.val_text
        grams = self.train_8grams if split == "train" else self.val_8grams
        window = _ngrams(list(ids), 8)
        frac = sum(1 for gram in window if gram in grams) / len(window) if window else 0.0
        span_len, span = longest_copy(text, corpus)
        return {
            "token_8gram_overlap": frac,
            "longest_char_span": span_len,
            "longest_char_span_text": span,
        }

    def compare(self, text, ids):
        """Return overlap measurements against both dataset splits."""
        return {
            "train": self.score(text, ids, "train"),
            "val": self.score(text, ids, "val"),
        }


def load_split_texts(data_dir, output_dir):
    """Load texts from a saved split, falling back to all available documents."""
    split_path = Path(output_dir) / "split.json"
    data = Path(data_dir)
    if split_path.exists():
        split = json.loads(split_path.read_text(encoding="utf-8"))
        train_texts = [(data / name).read_text(encoding="utf-8") for name in split["train"]]
        val_texts = [(data / name).read_text(encoding="utf-8") for name in split["val"]]
        return train_texts, val_texts, split
    paths = sorted(data.glob("*.txt"))
    texts = [p.read_text(encoding="utf-8") for p in paths]
    return texts, texts[:1], {"train": [p.name for p in paths], "val": []}


def load_trained(output, checkpoint_name, device, token_path):
    """Restore a tokenizer and trained model checkpoint for inference."""
    checkpoint = load_checkpoint(f"{output}/{checkpoint_name}.pt")
    tok = load_tokenizer(token_path, checkpoint["config"]["vocab_size"])
    model = MiniGPT(**checkpoint["config"])
    model.load_state_dict(checkpoint["model"])
    model.to(device)
    model.eval()
    return model, tok, checkpoint


def parse_temperatures(text):
    """Parse a comma-separated list of sampling temperatures."""
    return [float(part.strip()) for part in str(text).split(",") if part.strip()]


def run_suite(args):
    """Run validation and generation checks, then write an evaluation report."""
    device = resolve_device(getattr(args, "device", "auto"))
    checkpoint_name = getattr(args, "checkpoint", "best")
    model, tok, checkpoint = load_trained(args.output, checkpoint_name, device, args.token)
    train_texts, val_texts, split = load_split_texts(args.data, args.output)
    val_ids = [tok.encode(text) for text in val_texts]
    cfg = checkpoint["config"]
    eval_batches = getattr(args, "eval_batches", 20)
    windows = []
    if val_ids:
        try:
            windows = sample_windows(
                val_ids,
                max(eval_batches * args.batch, args.batch),
                cfg["context"],
                random.Random(args.seed),
            )
        except ValueError:
            windows = []
    val_loss = None
    if windows:
        val_loss = estimate_loss(model, windows, cfg["vocab_size"], device, args.batch)

    prompts = list(getattr(args, "prompts", None) or [])
    if not prompts:
        prompts = [
            getattr(args, "prompt", None) or "Once upon a time",
            "Emily has a good",
            "When she received a job offer from a marketing company in Chicago, she decided to accept it",
        ]
        prompts = list(dict.fromkeys(prompts))
    temperatures = parse_temperatures(getattr(args, "temperatures", "0,0.5,0.8,1.0"))
    new_tokens = getattr(args, "new_tokens", 200)
    memo = MemorizationIndex(train_texts, val_texts, tok)
    runs = []
    print(f"device: {describe_device(device)} checkpoint: {checkpoint_name}.pt", flush=True)
    if val_loss is not None:
        print(f"val_loss={val_loss:.3f}", flush=True)

    for prompt in prompts:
        for temperature in temperatures:
            for controls in (False, True):
                kwargs = dict(
                    temperature=temperature,
                    top_k=getattr(args, "top_k", 20),
                    top_p=getattr(args, "top_p", None),
                    repetition_penalty=1.2 if controls else 1.0,
                    no_repeat_ngram=3 if controls else 0,
                    seed=args.seed,
                )
                text, ids = generate(model, tok, prompt, new_tokens, **kwargs)
                continuation = text[len(prompt) :] if text.startswith(prompt) else text
                stats = repetition_stats(ids[len(tok.encode(prompt)) :])
                memory = memo.compare(continuation, ids[len(tok.encode(prompt)) :])
                record = {
                    "prompt": prompt,
                    "temperature": temperature,
                    "repetition_controls": controls,
                    "text": text,
                    "repetition": stats,
                    "memorization": memory,
                }
                runs.append(record)
                flag = " degenerate" if stats["is_degenerate"] else ""
                print(
                    f"temp={temperature:.1f} controls={int(controls)} "
                    f"d1={stats['distinct_1']:.2f} d3={stats['distinct_3']:.2f} "
                    f"rep4={stats['repeated_4gram_frac']:.2f} "
                    f"train8={memory['train']['token_8gram_overlap']:.2f} "
                    f"copy={memory['train']['longest_char_span']}{flag}",
                    flush=True,
                )
                print(text[:240].replace("\n", "\\n"), flush=True)
                print("---", flush=True)

    report = {
        "checkpoint": f"{checkpoint_name}.pt",
        "device": describe_device(device),
        "config": cfg,
        "val_loss": val_loss,
        "split": {"train": split.get("train", []), "val": split.get("val", [])},
        "runs": runs,
    }
    out = Path(args.output) / "eval_report.json"
    out.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print("Wrote", out)
    return report


def main():
    """Parse evaluation options and write an evaluation report."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default="data")
    parser.add_argument("--output", default="out")
    parser.add_argument("--token", default="out/bpe.json")
    parser.add_argument("--batch", type=int, default=64)
    parser.add_argument("--eval-batches", type=int, default=20)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--prompt", default="Once upon a time")
    parser.add_argument("--prompts", action="append", default=None)
    parser.add_argument("--new-tokens", type=int, default=200)
    parser.add_argument("--device", default="cuda", choices=["auto", "cpu", "cuda"])
    parser.add_argument("--checkpoint", default="best", choices=["best", "last"])
    parser.add_argument("--temperatures", default="0,0.6,0.8,1.0")
    parser.add_argument("--top-k", type=int, default=20)
    parser.add_argument("--top-p", type=float, default=None)
    args = parser.parse_args()
    run_suite(args)


if __name__ == "__main__":
    main()
