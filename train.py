"""Command-line entry point for training, evaluating, and sampling MiniGPT."""

import argparse

from evaluate import load_trained, run_suite
from generate import generate
from train import train
from train.device import resolve_device


def str_to_bool(value):
    """Parse common command-line boolean spellings into a bool."""
    if isinstance(value, bool):
        return value
    lowered = str(value).strip().lower()
    if lowered in ("true", "1", "yes"):
        return True
    if lowered in ("false", "0", "no"):
        return False
    raise argparse.ArgumentTypeError("expected true or false")


def main():
    """Parse command-line options and dispatch to the selected operating mode."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default="data")
    parser.add_argument("--output", default="out")
    parser.add_argument("--token", default="out/bpe.json")
    parser.add_argument("--context", type=int, default=512)
    parser.add_argument("--dim", type=int, default=256)
    parser.add_argument("--heads", type=int, default=6)
    parser.add_argument("--layers", type=int, default=6)
    parser.add_argument("--dropout", type=float, default=0.2)
    parser.add_argument("--batch", type=int, default=64)
    parser.add_argument("--steps", type=int, default=5000)
    parser.add_argument("--eval-every", type=int, default=50)
    parser.add_argument("--eval-batches", type=int, default=20)
    parser.add_argument("--patience", type=int, default=10)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--prompt", default="Once upon a time")
    parser.add_argument("--prompts", action="append", default=None)
    parser.add_argument("--new-tokens", type=int, default=200)
    parser.add_argument("--device", default="cuda", choices=["auto", "cpu", "cuda"])
    parser.add_argument("--resume", type=str_to_bool, default=False, metavar="true/false")
    parser.add_argument("--checkpoint", default="best", choices=["best", "last"])
    parser.add_argument("--temperature", type=float, default=0.8)
    parser.add_argument("--temperatures", default="0,0.6,0.8,1.0")
    parser.add_argument("--top-k", type=int, default=20)
    parser.add_argument("--top-p", type=float, default=None)
    parser.add_argument("--repetition-penalty", type=float, default=1.1)
    parser.add_argument("--penalty-window", type=int, default=64)
    parser.add_argument("--no-repeat-ngram", type=int, default=3)
    parser.add_argument("--mode", choices=["train", "generate", "eval"], default="train")
    args = parser.parse_args()
    if args.mode == "train":
        train(args)
        return
    if args.mode == "eval":
        run_suite(args)
        return
    device = resolve_device(args.device)
    model, tok, _ = load_trained(args.output, args.checkpoint, device, args.token)
    text, _ = generate(
        model,
        tok,
        args.prompt,
        args.new_tokens,
        temperature=args.temperature,
        top_k=args.top_k,
        top_p=args.top_p,
        repetition_penalty=args.repetition_penalty,
        penalty_window=args.penalty_window,
        no_repeat_ngram=args.no_repeat_ngram,
        seed=args.seed,
    )
    print(text)


if __name__ == "__main__":
    main()
