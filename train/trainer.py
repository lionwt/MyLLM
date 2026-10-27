"""End-to-end training orchestration for MiniGPT."""

import csv
import os
import random
import time
from pathlib import Path

import torch
from torch import nn

from train.dataset import (
    LazyEncodedDocuments,
    ShuffledWindowEpochs,
    load_split,
    sample_windows,
    save_split,
    split_documents,
)
from train.device import describe_device, resolve_device
from train.io import load_checkpoint, load_tokenizer, resume_checkpoint_path, save_checkpoint
from train.loss import batch_loss, estimate_loss
from train.model import MiniGPT


def _args_dict(args):
    """Convert serializable command-line arguments into checkpoint metadata."""
    out = {}
    for key, value in vars(args).items():
        if isinstance(value, (int, float, str, bool, type(None), list)):
            out[key] = value
    return out


def _load_tokenizer(path, vocab_size=None):
    """Load a tokenizer while reporting progress to the console."""
    print(f"Loading tokenizer {path}...", flush=True)
    tok = load_tokenizer(path, vocab_size)
    print(f"Loading tokenizer finished: {len(tok.vocab)} tokens...", flush=True)
    return tok


def _resume_run(args):
    """Restore run configuration, split, tokenizer, and checkpoint state."""
    ckpt_path = resume_checkpoint_path(args.output)
    print(f"Loading checkpoint {ckpt_path}...", flush=True)
    checkpoint = load_checkpoint(ckpt_path)
    cfg = dict(checkpoint["config"])
    print("Loading checkpoint finished...", flush=True)
    train_paths, val_paths = load_split(Path(args.output) / "split.json", args.data)
    tok = _load_tokenizer(args.token, cfg["vocab_size"])
    start_step = int(checkpoint.get("step") or 0)
    print(f"resume: loaded {ckpt_path} from step {start_step}", flush=True)
    return cfg, tok, train_paths, val_paths, checkpoint, start_step


def _new_run(args):
    """Create the data split and model configuration for a fresh run."""
    paths = sorted(Path(args.data).glob("*.txt"))
    if len(paths) < 2:
        raise ValueError("Need at least two .txt documents for train/validation split")
    train_paths, val_paths = split_documents(paths, random.Random(args.seed))
    tok = _load_tokenizer(args.token)
    save_split(Path(args.output) / "split.json", train_paths, val_paths)
    cfg = dict(
        vocab_size=len(tok.vocab),
        context=args.context,
        dim=args.dim,
        heads=args.heads,
        layers=args.layers,
        dropout=args.dropout,
    )
    return cfg, tok, train_paths, val_paths, None, 0


def _lazy_split(tok, train_paths, val_paths):
    """Wrap splits so files are tokenized on first use and then cached."""
    print(
        f"Deferring encode of {len(train_paths) + len(val_paths)} texts "
        f"({len(train_paths)} train, {len(val_paths)} val)...",
        flush=True,
    )
    train_ids = LazyEncodedDocuments(train_paths, tok, name="train")
    val_ids = LazyEncodedDocuments(val_paths, tok, name="val")
    return train_ids, val_ids


def _build_model(cfg, args, device, checkpoint):
    """Build model and optimizer, optionally restoring checkpoint state."""
    print("Building model...", flush=True)
    model = MiniGPT(**cfg).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.1)
    best_val = float("inf")
    if checkpoint is not None:
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
    return model, optimizer, best_val


def train(args):
    """Train MiniGPT, evaluate periodically, and persist best/last checkpoints."""
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = resolve_device(getattr(args, "device", "cuda"))
    resume = bool(getattr(args, "resume", False))
    os.makedirs(args.output, exist_ok=True)

    if resume:
        cfg, tok, train_paths, val_paths, checkpoint, start_step = _resume_run(args)
    else:
        cfg, tok, train_paths, val_paths, checkpoint, start_step = _new_run(args)
    train_ids, val_ids = _lazy_split(tok, train_paths, val_paths)

    train_loader = ShuffledWindowEpochs(
        train_ids, cfg["context"], args.batch, random.Random(args.seed)
    )
    if start_step:
        train_loader.skip_batches(start_step)
    print(
        f"train documents: {len(train_ids)}, val documents: {len(val_ids)}",
        flush=True,
    )

    eval_batches = getattr(args, "eval_batches", 20)
    eval_rng = random.Random(args.seed)
    n_eval_windows = eval_batches * args.batch
    train_eval_windows = None
    val_eval_windows = None

    model, optimizer, best_val = _build_model(cfg, args, device, checkpoint)

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

    def persist(path, step, val_loss, train_loss):
        """Save the current model, optimizer, metrics, and run metadata."""
        save_checkpoint(
            path,
            model,
            cfg,
            optimizer=optimizer,
            step=step,
            val_loss=val_loss,
            best_val_loss=best_val,
            current_val_loss=val_loss,
            train_loss=train_loss,
            **meta,
        )

    print(f"Training steps {start_step + 1}-{end_step}...", flush=True)
    last_eval_time = time.perf_counter()
    for step in range(start_step + 1, end_step + 1):
        model.train()
        x, y = train_loader.next_batch(device)
        loss = batch_loss(model, x, y, cfg["vocab_size"])
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        if step % args.eval_every == 0 or step == start_step + 1 or step == end_step:
            if train_eval_windows is None:
                train_eval_windows = sample_windows(
                    train_ids, n_eval_windows, cfg["context"], eval_rng
                )
                val_eval_windows = sample_windows(
                    val_ids, n_eval_windows, cfg["context"], eval_rng
                )
            train_eval = estimate_loss(model, train_eval_windows, cfg["vocab_size"], device, args.batch)
            val_eval = estimate_loss(model, val_eval_windows, cfg["vocab_size"], device, args.batch)
            now = time.perf_counter()
            elapsed = now - last_eval_time
            last_eval_time = now
            lr = optimizer.param_groups[0]["lr"]
            print(
                f"step {step:5d} train_loss={train_eval:.3f} val_loss={val_eval:.3f} elapsed={elapsed:.1f}s",
                flush=True,
            )
            with log_path.open("a", encoding="utf-8", newline="") as handle:
                csv.writer(handle).writerow([step, f"{train_eval:.6f}", f"{val_eval:.6f}", lr])
            last_train_eval = train_eval
            last_val_eval = val_eval
            if val_eval < best_val:
                best_val = val_eval
                stale = 0
                persist(f"{args.output}/best.pt", step, val_eval, train_eval)
            else:
                stale += 1
            if patience and stale >= patience:
                print(f"early stop at step {step} (patience={patience})", flush=True)
                break

    if loss is None:
        raise RuntimeError("No training steps ran; increase --steps")
    print("Training finished...", flush=True)
    print("Saving last checkpoint...", flush=True)
    persist(
        f"{args.output}/last.pt",
        step,
        last_val_eval,
        last_train_eval if last_train_eval is not None else loss.item(),
    )
    print("Saved model in", args.output)
