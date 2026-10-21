import csv
import json
import os
import random
import re
from pathlib import Path

import torch
from torch import nn
import torch.nn.functional as F

from device import describe_device, load_checkpoint, resolve_device, save_checkpoint
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


def _jaccard(a, b):
    if not a and not b:
        return 1.0
    if not a or not b:
        return 0.0
    return len(a & b) / len(a | b)


def _dedupe_exact(paths, texts):
    seen = set()
    out_paths = []
    out_texts = []
    for path, text in zip(paths, texts):
        if text in seen:
            continue
        seen.add(text)
        out_paths.append(path)
        out_texts.append(text)
    return out_paths, out_texts


def _cluster_indices(texts, threshold=0.35):
    n = len(texts)
    parent = list(range(n))

    def find(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    sets = [set(_sentences([text])) for text in texts]
    for i in range(n):
        for j in range(i + 1, n):
            if _jaccard(sets[i], sets[j]) >= threshold:
                parent[find(j)] = find(i)
    clusters = {}
    for i in range(n):
        clusters.setdefault(find(i), []).append(i)
    return list(clusters.values())


def _strip_train_sentences(val_texts, train_texts):
    train_sents = set(_sentences(train_texts))
    cleaned = []
    for text in val_texts:
        kept = [sentence for sentence in _sentences([text]) if sentence not in train_sents]
        cleaned.append(" ".join(kept))
    return cleaned


def split_documents(paths, texts, val_frac=0.1, threshold=0.35):
    paths, texts = _dedupe_exact(paths, texts)
    if len(paths) < 2:
        raise ValueError("Need at least two distinct documents for train/validation split")
    clusters = _cluster_indices(texts, threshold=threshold)
    random.shuffle(clusters)
    n_val = max(1, round(val_frac * len(clusters)))
    if n_val >= len(clusters):
        n_val = len(clusters) - 1
    val_idx = [i for cluster in clusters[:n_val] for i in cluster]
    train_idx = [i for cluster in clusters[n_val:] for i in cluster]
    train_paths = [paths[i] for i in train_idx]
    train_texts = [texts[i] for i in train_idx]
    val_paths = [paths[i] for i in val_idx]
    val_texts = _strip_train_sentences([texts[i] for i in val_idx], train_texts)
    kept = [(path, text) for path, text in zip(val_paths, val_texts) if text.strip()]
    if not kept:
        raise ValueError("Validation set is empty after removing duplicated content")
    val_paths, val_texts = zip(*kept)
    return list(train_paths), list(val_paths), train_texts, list(val_texts)


def _weighted_choice(eligible, context, rng):
    weights = [len(seq) - context for seq in eligible]
    return rng.choices(eligible, weights=weights, k=1)[0]


def sample_windows(documents, n_windows, context, rng):
    eligible = [seq for seq in documents if len(seq) > context]
    if not eligible:
        raise ValueError("Need documents longer than context length")
    windows = []
    for _ in range(n_windows):
        seq = _weighted_choice(eligible, context, rng)
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


def _load_split_paths(args):
    data = Path(args.data)
    split_path = Path(args.output) / "split.json"
    if not split_path.exists():
        raise FileNotFoundError(f"Cannot resume: missing {split_path}")
    split = json.loads(split_path.read_text(encoding="utf-8"))
    train_paths = [data / name for name in split["train"]]
    val_paths = [data / name for name in split["val"]]
    missing = [str(path) for path in train_paths + val_paths if not path.exists()]
    if missing:
        raise FileNotFoundError("Cannot resume: missing documents: " + ", ".join(missing))
    return train_paths, val_paths, split.get("data") or {}


def _resume_checkpoint_path(args):
    output = Path(args.output)
    for path in (output / "last.pt", output / "best.pt"):
        if path.exists():
            return path
    raise FileNotFoundError(f"Cannot resume: no last.pt or best.pt in {output}")


def train(args):
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = resolve_device(getattr(args, "device", "cuda"))
    resume = bool(getattr(args, "resume", False))
    os.makedirs(args.output, exist_ok=True)
    paths = sorted(Path(args.data).glob("*.txt"))
    if len(paths) < 2:
        raise ValueError("Need at least two .txt documents for train/validation split")

    checkpoint = None
    start_step = 0
    if resume:
        ckpt_path = _resume_checkpoint_path(args)
        tokenizer_path = Path(args.output) / "tokenizer.json"
        if not tokenizer_path.exists():
            raise FileNotFoundError(f"Cannot resume: missing {tokenizer_path}")
        checkpoint = load_checkpoint(ckpt_path)
        cfg = dict(checkpoint["config"])
        train_paths, val_paths, stats = _load_split_paths(args)
        train_texts = [p.read_text(encoding="utf-8") for p in train_paths]
        val_texts = _strip_train_sentences(
            [p.read_text(encoding="utf-8") for p in val_paths],
            train_texts,
        )
        if not stats:
            stats = report_data(train_texts, val_texts, train_paths, val_paths)
        else:
            print(
                f"data: {stats.get('documents', len(train_texts) + len(val_texts))} documents, "
                f"{stats.get('bytes', 0)} bytes, "
                f"duplicate_sentences={stats.get('duplicate_sentence_share', 0):.1%}, "
                f"val_in_train={stats.get('val_sentence_overlap', 0):.1%}",
                flush=True,
            )
        tok = ByteBPE.load(tokenizer_path)
        start_step = int(checkpoint.get("step") or 0)
        print(f"resume: loaded {ckpt_path} from step {start_step}", flush=True)
    else:
        all_texts = [p.read_text(encoding="utf-8") for p in paths]
        train_paths, val_paths, train_texts, val_texts = split_documents(paths, all_texts)
        stats = report_data(train_texts, val_texts, train_paths, val_paths)
        tok = ByteBPE().train(train_texts, args.vocab)
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
        cfg = dict(
            vocab_size=len(tok.vocab),
            context=args.context,
            dim=args.dim,
            heads=args.heads,
            layers=args.layers,
            dropout=args.dropout,
        )

    assert_roundtrip(tok, train_texts + val_texts, train_paths + val_paths)
    train_ids = [tok.encode(s) for s in train_texts]
    val_ids = [tok.encode(s) for s in val_texts]

    def batch(documents):
        eligible = [s for s in documents if len(s) > cfg["context"]]
        if not eligible:
            raise ValueError("Need documents longer than context length")
        xs = []
        ys = []
        for _ in range(args.batch):
            seq = _weighted_choice(eligible, cfg["context"], random)
            start = random.randrange(len(seq) - cfg["context"])
            window = seq[start : start + cfg["context"] + 1]
            xs.append(window[:-1])
            ys.append(window[1:])
        return (
            torch.tensor(xs, dtype=torch.long, device=device),
            torch.tensor(ys, dtype=torch.long, device=device),
        )

    eval_batches = getattr(args, "eval_batches", 20)
    eval_rng = random.Random(args.seed)
    n_eval_windows = eval_batches * args.batch
    train_eval_windows = sample_windows(train_ids, n_eval_windows, cfg["context"], eval_rng)
    val_eval_windows = sample_windows(val_ids, n_eval_windows, cfg["context"], eval_rng)

    model = MiniGPT(**cfg).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.1)
    best_val = float("inf")
    if resume and checkpoint is not None:
        model.load_state_dict(checkpoint["model"])
        if checkpoint.get("optimizer"):
            optimizer.load_state_dict(checkpoint["optimizer"])
            for state in optimizer.state.values():
                for key, value in list(state.items()):
                    if torch.is_tensor(value):
                        state[key] = value.to(device)
        for group in optimizer.param_groups:
            group["lr"] = args.lr
        if checkpoint.get("best_val_loss") is not None:
            best_val = float(checkpoint["best_val_loss"])
        else:
            best_val = float(checkpoint.get("val_loss") or float("inf"))
    print(
        "device:",
        describe_device(device),
        "parameters:",
        sum(p.numel() for p in model.parameters()),
        flush=True,
    )

    log_path = Path(args.output) / "train_log.csv"
    if not (resume and log_path.exists()):
        with log_path.open("w", encoding="utf-8", newline="") as handle:
            csv.writer(handle).writerow(["step", "train_loss", "val_loss", "lr"])

    stale = 0
    patience = getattr(args, "patience", 0)
    meta = {"args": _args_dict(args)}
    end_step = start_step + args.steps
    loss = None
    last_train_eval = None
    last_val_eval = None

    for step in range(start_step + 1, end_step + 1):
        model.train()
        x, y = batch(train_ids)
        logits = model(x)
        loss = F.cross_entropy(logits.reshape(-1, cfg["vocab_size"]), y.reshape(-1))
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        if step % args.eval_every == 0 or step == start_step + 1 or step == end_step:
            train_eval = estimate_loss(model, train_eval_windows, cfg["vocab_size"], device, args.batch)
            val_eval = estimate_loss(model, val_eval_windows, cfg["vocab_size"], device, args.batch)
            lr = optimizer.param_groups[0]["lr"]
            print(
                f"step {step:5d} train_loss={train_eval:.3f} val_loss={val_eval:.3f}",
                flush=True,
            )
            with log_path.open("a", encoding="utf-8", newline="") as handle:
                csv.writer(handle).writerow([step, f"{train_eval:.6f}", f"{val_eval:.6f}", lr])
            last_train_eval = train_eval
            last_val_eval = val_eval
            if val_eval < best_val:
                best_val = val_eval
                stale = 0
                save_checkpoint(
                    f"{args.output}/best.pt",
                    model,
                    cfg,
                    optimizer=optimizer,
                    step=step,
                    val_loss=val_eval,
                    best_val_loss=best_val,
                    current_val_loss=val_eval,
                    train_loss=train_eval,
                    **meta,
                )
            else:
                stale += 1
            if patience and stale >= patience:
                print(f"early stop at step {step} (patience={patience})", flush=True)
                break

    if loss is None:
        raise RuntimeError("No training steps ran; increase --steps")
    save_checkpoint(
        f"{args.output}/last.pt",
        model,
        cfg,
        optimizer=optimizer,
        step=step,
        val_loss=last_val_eval,
        best_val_loss=best_val,
        current_val_loss=last_val_eval,
        train_loss=last_train_eval if last_train_eval is not None else loss.item(),
        **meta,
    )
    print("Saved model and tokenizer in", args.output)
