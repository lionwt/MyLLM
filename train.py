import csv
import json
import os
import random
import re
from pathlib import Path

import torch
from torch import nn
import torch.nn.functional as F

from device import describe_device, resolve_device, save_checkpoint
from model import MiniGPT
from tokenizer import ByteBPE


def _sentences(texts):
    sentences = []
    for text in texts:
        for sentence in re.split(r"(?<=[.!?])\s+", text):
            sentence = sentence.strip()
            if sentence:
                sentences.append(sentence)
    return sentences


def report_data(train_texts, val_texts, train_paths, val_paths):
    all_texts = train_texts + val_texts
    total_bytes = sum(len(text.encode("utf-8")) for text in all_texts)
    all_sents = _sentences(all_texts)
    unique = set(all_sents)
    duplicate_share = 1 - (len(unique) / len(all_sents)) if all_sents else 0.0
    train_sents = set(_sentences(train_texts))
    val_sents = _sentences(val_texts)
    overlap = (
        sum(1 for sentence in val_sents if sentence in train_sents) / len(val_sents)
        if val_sents
        else 0.0
    )
    print(
        f"data: {len(all_texts)} documents, {total_bytes} bytes, "
        f"duplicate_sentences={duplicate_share:.1%}, val_in_train={overlap:.1%}",
        flush=True,
    )
    return {
        "documents": len(all_texts),
        "bytes": total_bytes,
        "train_documents": [p.name for p in train_paths],
        "val_documents": [p.name for p in val_paths],
        "duplicate_sentence_share": duplicate_share,
        "val_sentence_overlap": overlap,
    }


def assert_roundtrip(tok, texts, paths):
    total_bytes = 0
    total_tokens = 0
    for text, path in zip(texts, paths):
        ids = tok.encode(text)
        if tok.decode(ids) != text:
            raise ValueError(f"Tokenizer round-trip failed for {path}")
        total_bytes += len(text.encode("utf-8"))
        total_tokens += max(len(ids), 1)
    print(f"tokenizer round-trip ok, bytes/token={total_bytes / total_tokens:.3f}", flush=True)


def sample_windows(documents, n_windows, context, rng):
    eligible = [seq for seq in documents if len(seq) > context]
    if not eligible:
        raise ValueError("Need documents longer than context length")
    windows = []
    for _ in range(n_windows):
        seq = eligible[rng.randrange(len(eligible))]
        start = rng.randrange(len(seq) - context)
        windows.append(seq[start : start + context + 1])
    return windows


def estimate_loss(model, windows, vocab_size, device, batch_size):
    model.eval()
    total = 0.0
    count = 0
    with torch.no_grad():
        for i in range(0, len(windows), batch_size):
            chunk = windows[i : i + batch_size]
            xs = torch.tensor([w[:-1] for w in chunk], dtype=torch.long, device=device)
            ys = torch.tensor([w[1:] for w in chunk], dtype=torch.long, device=device)
            logits = model(xs)
            loss = F.cross_entropy(logits.reshape(-1, vocab_size), ys.reshape(-1))
            total += loss.item() * len(chunk)
            count += len(chunk)
    return total / count


def _args_dict(args):
    out = {}
    for key, value in vars(args).items():
        if isinstance(value, (int, float, str, bool, type(None), list)):
            out[key] = value
    return out


def train(args):
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = resolve_device(getattr(args, "device", "cuda"))
    paths = sorted(Path(args.data).glob("*.txt"))
    if len(paths) < 2:
        raise ValueError("Need at least two .txt documents for train/validation split")
    shuffled = paths[:]
    random.shuffle(shuffled)
    n_val = max(1, round(0.1 * len(shuffled)))
    train_paths, val_paths = shuffled[n_val:], shuffled[:n_val]
    train_texts = [p.read_text(encoding="utf-8") for p in train_paths]
    val_texts = [p.read_text(encoding="utf-8") for p in val_paths]
    stats = report_data(train_texts, val_texts, train_paths, val_paths)

    tok = ByteBPE().train(train_texts, args.vocab)
    os.makedirs(args.output, exist_ok=True)
    tok.save(f"{args.output}/tokenizer.json")
    Path(f"{args.output}/split.json").write_text(
        json.dumps(
            {
                "train": [p.name for p in train_paths],
                "val": [p.name for p in val_paths],
                "data": stats,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    assert_roundtrip(tok, train_texts + val_texts, train_paths + val_paths)
    train_ids = [tok.encode(s) for s in train_texts]
    val_ids = [tok.encode(s) for s in val_texts]

    def batch(documents):
        eligible = [s for s in documents if len(s) > args.context]
        if not eligible:
            raise ValueError("Need documents longer than context length")
        xs = []
        ys = []
        for _ in range(args.batch):
            seq = random.choice(eligible)
            start = random.randint(0, len(seq) - args.context - 1)
            window = seq[start : start + args.context + 1]
            xs.append(window[:-1])
            ys.append(window[1:])
        return (
            torch.tensor(xs, dtype=torch.long, device=device),
            torch.tensor(ys, dtype=torch.long, device=device),
        )

    eval_batches = getattr(args, "eval_batches", 20)
    eval_rng = random.Random(args.seed)
    n_eval_windows = eval_batches * args.batch
    train_eval_windows = sample_windows(train_ids, n_eval_windows, args.context, eval_rng)
    val_eval_windows = sample_windows(val_ids, n_eval_windows, args.context, eval_rng)

    cfg = dict(
        vocab_size=len(tok.vocab),
        context=args.context,
        dim=args.dim,
        heads=args.heads,
        layers=args.layers,
        dropout=args.dropout,
    )
    model = MiniGPT(**cfg).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.1)
    print(
        "device:",
        describe_device(device),
        "parameters:",
        sum(p.numel() for p in model.parameters()),
        flush=True,
    )

    log_path = Path(args.output) / "train_log.csv"
    with log_path.open("w", encoding="utf-8", newline="") as handle:
        csv.writer(handle).writerow(["step", "train_loss", "val_loss", "lr"])

    best_val = float("inf")
    stale = 0
    patience = getattr(args, "patience", 0)
    meta = {"args": _args_dict(args)}

    for step in range(1, args.steps + 1):
        model.train()
        x, y = batch(train_ids)
        logits = model(x)
        loss = F.cross_entropy(logits.reshape(-1, cfg["vocab_size"]), y.reshape(-1))
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        if step % args.eval_every == 0 or step == 1 or step == args.steps:
            train_eval = estimate_loss(model, train_eval_windows, cfg["vocab_size"], device, args.batch)
            val_eval = estimate_loss(model, val_eval_windows, cfg["vocab_size"], device, args.batch)
            lr = optimizer.param_groups[0]["lr"]
            print(
                f"step {step:5d} train_loss={train_eval:.3f} val_loss={val_eval:.3f}",
                flush=True,
            )
            with log_path.open("a", encoding="utf-8", newline="") as handle:
                csv.writer(handle).writerow([step, f"{train_eval:.6f}", f"{val_eval:.6f}", lr])
            if val_eval < best_val:
                best_val = val_eval
                stale = 0
                save_checkpoint(
                    f"{args.output}/best.pt",
                    model,
                    cfg,
                    step=step,
                    val_loss=val_eval,
                    train_loss=train_eval,
                    **meta,
                )
            else:
                stale += 1
            if patience and stale >= patience:
                print(f"early stop at step {step} (patience={patience})", flush=True)
                break

    save_checkpoint(
        f"{args.output}/last.pt",
        model,
        cfg,
        step=step,
        val_loss=best_val,
        train_loss=loss.item(),
        **meta,
    )
    print("Saved model and tokenizer in", args.output)
