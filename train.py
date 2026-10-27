"""Command-line entry point for training MiniGPT."""

import argparse

from train import train


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
    """Parse training options and run the trainer."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default="data")
    parser.add_argument("--output", default="out")
    parser.add_argument("--token", default="out/bpe.json")
    parser.add_argument("--context", type=int, default=512)
    parser.add_argument("--dim", type=int, default=240)
    parser.add_argument("--heads", type=int, default=6)
    parser.add_argument("--layers", type=int, default=6)
    parser.add_argument("--dropout", type=float, default=0.2)
    parser.add_argument("--batch", type=int, default=64)
    parser.add_argument("--steps", type=int, default=5000)
    parser.add_argument("--eval-every", type=int, default=50)
    parser.add_argument("--eval-batches", type=int, default=50)
    parser.add_argument("--patience", type=int, default=10)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="cuda", choices=["auto", "cpu", "cuda"])
    parser.add_argument("--resume", type=str_to_bool, default=False, metavar="true/false")
    args = parser.parse_args()
    train(args)


if __name__ == "__main__":
    main()
