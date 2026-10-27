# Mini-GPT

A small GPT trained from scratch on the documents in `data/`.

The output layer shares the token embedding (`F.linear(..., token.weight)`), so the weights stay tied on CPU and CUDA. Training writes `best.pt` when validation loss improves and `last.pt` at the end.

Install a CUDA build of PyTorch for an NVIDIA GPU:

```bash
pip install torch --index-url https://download.pytorch.org/whl/cu128
```

## Tokenizer

The byte-level BPE tokenizer is built separately with the `tokenizers` package and saved as `bpe.json`:

```bash
python -m tokenizers.prepare --data data --output out --vocab-size 2048 --max-token-bytes 64
```

Training, generation, and evaluation load it with `--token` (default `out/bpe.json`). The model's vocabulary size comes from this file, and a checkpoint can only be used with the tokenizer it was trained on.

## Train

```bash
python train.py --device cuda --token out/bpe.json --dim 192 --heads 6 --layers 4 --dropout 0.1 --lr 1e-3 --batch 32 --steps 5000 --eval-every 100 --patience 10 --output out
```

`--device` defaults to `cuda` and errors if CUDA is missing. `--device auto` uses CUDA when it is available and otherwise falls back to CPU. `--device cpu` is only for machines with no NVIDIA GPU.

Continue from a saved run with `--resume true`. That reloads `split.json`, the `--token` tokenizer, and `last.pt` (falling back to `best.pt`), then trains `--steps` more updates:

```bash
python train.py --resume true --token out/bpe.json --steps 2000 --lr 3e-4 --output out
```

`--resume false` (the default) starts over and overwrites the split in `--output`.

Training also writes:

- `split.json` — train/validation file names. New runs shuffle the files with `--seed` and hold out 10% for validation.
- `train_log.csv` — step, train loss, val loss, learning rate

Validation averages `--eval-batches` fixed batches so the printed `val_loss` is comparable across steps. `--patience N` stops after N evaluations without improvement (`0` disables it).

## Generate

```bash
python generate.py --prompt "Once upon a time" --token out/bpe.json --output out
```

Defaults are `--temperature 0.6`, `--top-k 20`, `--repetition-penalty 1.1`, `--no-repeat-ngram 3`, and `--new-tokens 200`. Other useful flags: `--checkpoint best|last`, `--top-p`, `--temperature 0` for greedy decoding.

## Evaluate

```bash
python evaluate.py --token out/bpe.json --output out --new-tokens 150
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