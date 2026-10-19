# Mini-GPT

A small GPT trained from scratch on the documents in `data/`.

The output layer shares the token embedding (`F.linear(..., token.weight)`), so the weights stay tied on CPU and CUDA. Training writes `best.pt` when validation loss improves and `last.pt` at the end.

Install a CUDA build of PyTorch for an NVIDIA GPU:

```bash
pip install torch --index-url https://download.pytorch.org/whl/cu128
```

## Train

```bash
python mini_gpt_from_scratch.py --mode train --device cuda --vocab 2048 --dim 192 --heads 6 --layers 4 --dropout 0.1 --lr 1e-3 --batch 32 --steps 5000 --eval-every 100 --patience 10 --output out
```

`--device` defaults to `cuda`. Do not pass `--device cpu` on the NVIDIA machine. If PyTorch was installed from plain PyPI it is often CPU-only; reinstall with the CUDA index above. `--device cpu` is only for machines with no NVIDIA GPU.

Training also writes:

- `tokenizer.json` — byte-level BPE with word-level pre-tokenization
- `split.json` — train/validation file names and data stats
- `train_log.csv` — step, train loss, val loss, learning rate

Validation averages `--eval-batches` fixed batches so the printed `val_loss` is comparable across steps. `--patience N` stops after N evaluations without improvement (`0` disables it).

## Generate

```bash
python mini_gpt_from_scratch.py --mode generate --prompt "Once upon a time" --new-tokens 150 --output out
```

Useful flags: `--checkpoint best|last`, `--temperature` (`0` is greedy), `--top-k`, `--top-p`, `--repetition-penalty`, `--no-repeat-ngram`.

## Evaluate

```bash
python mini_gpt_from_scratch.py --mode eval --output out --new-tokens 150
```

This reloads the saved split, recomputes validation loss, and generates each prompt at temperatures `0,0.5,0.8,1.0` with and without repetition controls (penalty 1.2 and no-repeat 3-gram). Results go to `eval_report.json`.

How to read a run in the report:

- `repetition.distinct_1` / `distinct_3` — higher means more variety
- `repetition.repeated_4gram_frac` and `is_degenerate` — looping output
- `memorization.train.token_8gram_overlap` — share of generated 8-grams that appear in the training set
- `memorization.train.longest_char_span` — longest verbatim copy from training text

The current `data/` files repeat many sentences, so memorization scores will be high even when generation is fluent.

## Tests

```bash
python -m unittest discover test
```
