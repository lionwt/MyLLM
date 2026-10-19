import argparse

from device import resolve_device
from evaluate import load_trained, run_suite
from generate import generate
from train import train


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default="data")
    parser.add_argument("--output", default="out")
    parser.add_argument("--vocab", type=int, default=1024)
    parser.add_argument("--context", type=int, default=128)
    parser.add_argument("--dim", type=int, default=256)
    parser.add_argument("--heads", type=int, default=4)
    parser.add_argument("--layers", type=int, default=4)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--batch", type=int, default=16)
    parser.add_argument("--steps", type=int, default=2000)
    parser.add_argument("--eval-every", type=int, default=100)
    parser.add_argument("--eval-batches", type=int, default=20)
    parser.add_argument("--patience", type=int, default=0)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--prompt", default="Once upon a time")
    parser.add_argument("--prompts", action="append", default=None)
    parser.add_argument("--new-tokens", type=int, default=150)
    parser.add_argument("--device", default="cuda", choices=["auto", "cpu", "cuda"])
    parser.add_argument("--checkpoint", default="best", choices=["best", "last"])
    parser.add_argument("--temperature", type=float, default=0.8)
    parser.add_argument("--temperatures", default="0,0.5,0.8,1.0")
    parser.add_argument("--top-k", type=int, default=40)
    parser.add_argument("--top-p", type=float, default=None)
    parser.add_argument("--repetition-penalty", type=float, default=1.0)
    parser.add_argument("--penalty-window", type=int, default=64)
    parser.add_argument("--no-repeat-ngram", type=int, default=0)
    parser.add_argument("--mode", choices=["train", "generate", "eval"], default="train")
    args = parser.parse_args()
    if args.mode == "train":
        train(args)
        return
    if args.mode == "eval":
        run_suite(args)
        return
    device = resolve_device(args.device)
    model, tok, _ = load_trained(args.output, args.checkpoint, device)
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
