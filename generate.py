import torch
import torch.nn.functional as F


def _apply_top_p(logits, top_p):
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
    new_tokens=150,
    temperature=0.8,
    top_k=40,
    top_p=None,
    repetition_penalty=1.0,
    penalty_window=64,
    no_repeat_ngram=0,
    seed=None,
):
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
