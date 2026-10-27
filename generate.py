"""Autoregressive text generation and token-sampling controls."""

import argparse

import torch
import torch.nn.functional as F


def _apply_top_p(logits, top_p):
    """Mask tokens outside the smallest probability mass exceeding ``top_p``."""
    sorted_logits, sorted_idx = torch.sort(logits, descending=True)
    probs = F.softmax(sorted_logits, dim=-1)
    cumsum = torch.cumsum(probs, dim=-1)
    remove = cumsum - probs > top_p
    filtered = sorted_logits.clone()
    filtered[remove] = float("-inf")
    out = torch.full_like(logits, float("-inf"))
    out.scatter_(0, sorted_idx, filtered)
    return out


def _banned_tokens(ids, n):
    """Find tokens that would repeat an n-gram ending at the current position."""
    prefix = tuple(ids[-(n - 1) :])
    banned = set()
    for i in range(len(ids) - n + 1):
        if tuple(ids[i : i + n - 1]) == prefix:
            banned.add(ids[i + n - 1])
    return banned


@torch.no_grad()
def generate(
    model,
    tok,
    prompt,
    new_tokens=200,
    temperature=0.6,
    top_k=20,
    top_p=None,
    repetition_penalty=1.1,
    penalty_window=64,
    no_repeat_ngram=3,
    seed=None,
):
    """Generate a continuation from ``prompt`` using configurable sampling.

    Returns the decoded full sequence and its token IDs, including the prompt.
    A temperature of zero enables deterministic greedy decoding.
    """
    if temperature < 0:
        raise ValueError("temperature must be non-negative")
    if seed is not None:
        torch.manual_seed(seed)
    device = next(model.parameters()).device
    ids = tok.encode(prompt)
    if not ids:
        ids = [32]
    model.eval()
    for _ in range(new_tokens):
        x = torch.tensor([ids[-model.context :]], dtype=torch.long, device=device)
        logits = model(x)[0, -1].clone()
        if repetition_penalty != 1.0 and penalty_window > 0:
            for tid in set(ids[-penalty_window:]):
                if logits[tid] > 0:
                    logits[tid] /= repetition_penalty
                else:
                    logits[tid] *= repetition_penalty
        if no_repeat_ngram and no_repeat_ngram > 1 and len(ids) >= no_repeat_ngram - 1:
            banned = _banned_tokens(ids, no_repeat_ngram)
            if banned and len(banned) < logits.numel():
                logits[list(banned)] = float("-inf")
        if temperature == 0:
            nxt = int(logits.argmax().item())
        else:
            logits = logits / temperature
            if top_k:
                k = min(int(top_k), logits.numel())
                threshold = torch.topk(logits, k).values[-1]
                logits = logits.masked_fill(logits < threshold, float("-inf"))
            if top_p is not None and 0 < top_p < 1:
                logits = _apply_top_p(logits, top_p)
            probs = F.softmax(logits, dim=-1)
            if not torch.isfinite(probs).any() or float(probs.sum()) == 0:
                nxt = int(model(x)[0, -1].argmax().item())
            else:
                nxt = int(torch.multinomial(probs, 1).item())
        ids.append(nxt)
    return tok.decode(ids), ids


def main():
    """Parse generation options, load a checkpoint, and print sampled text."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", default="out")
    parser.add_argument("--token", default="out/bpe.json")
    parser.add_argument("--prompt", default="Once upon a time")
    parser.add_argument("--new-tokens", type=int, default=200)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="cuda", choices=["auto", "cpu", "cuda"])
    parser.add_argument("--checkpoint", default="best", choices=["best", "last"])
    parser.add_argument("--temperature", type=float, default=0.8)
    parser.add_argument("--top-k", type=int, default=20)
    parser.add_argument("--top-p", type=float, default=None)
    parser.add_argument("--repetition-penalty", type=float, default=1.1)
    parser.add_argument("--penalty-window", type=int, default=64)
    parser.add_argument("--no-repeat-ngram", type=int, default=3)
    args = parser.parse_args()

    from evaluate import load_trained
    from train.device import resolve_device

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
