"""CPU teaching implementations. They do not train or serve a real LLM.
Requires Python 3.10+ and NumPy. All log probabilities use natural logarithms.
"""
import math
from collections import defaultdict
import numpy as np


def log_softmax(logits, axis=-1):
    x = np.asarray(logits, dtype=np.float64)
    if not np.all(np.isfinite(x)):
        raise ValueError("logits must be finite")
    z = x - np.max(x, axis=axis, keepdims=True)
    return z - np.log(np.exp(z).sum(axis=axis, keepdims=True))


def masked_softmax(logits, mask, axis=-1):
    """True means allowed. Empty attention rows are rejected explicitly."""
    x = np.asarray(logits, dtype=np.float64)
    m = np.broadcast_to(np.asarray(mask, dtype=bool), x.shape)
    if not np.all(np.isfinite(x)):
        raise ValueError("input logits must be finite")
    if np.any(~m.any(axis=axis)):
        raise ValueError("at least one allowed position per row is required")
    z = np.where(m, x, -np.inf)
    z = z - np.max(z, axis=axis, keepdims=True)
    exp = np.exp(z)
    return exp / exp.sum(axis=axis, keepdims=True)


def sequence_logprob(logits, targets, response_mask):
    """Inputs already aligned: logits[b,t] predicts targets[b,t].
    No implicit causal shift here. Caller must align next-token targets once.
    Masked labels may be -100; empty responses raise instead of silently dividing.
    """
    x = np.asarray(logits)
    target = np.asarray(targets)
    mask = np.asarray(response_mask, dtype=bool)
    if x.ndim != 3 or target.shape != x.shape[:2] or mask.shape != target.shape:
        raise ValueError("expected logits B,T,V and targets/mask B,T")
    if not np.issubdtype(target.dtype, np.integer):
        raise ValueError("targets must be integer token IDs")
    if np.any((target[mask] < 0) | (target[mask] >= x.shape[-1])):
        raise ValueError("valid target outside vocabulary")
    lengths = mask.sum(axis=1)
    if np.any(lengths == 0):
        raise ValueError("empty response")
    safe_targets = np.where(mask, target, 0)
    token_logp = np.take_along_axis(log_softmax(x), safe_targets[..., None], axis=-1)[..., 0]
    sums = np.where(mask, token_logp, 0.0).sum(axis=1)
    return sums, sums / lengths, lengths


def causal_attention(q, k, v):
    """Same-length self-attention, shape (..., sequence, head_dim)."""
    q, k, v = map(lambda z: np.asarray(z, dtype=np.float64), (q, k, v))
    if q.shape != k.shape or q.shape[:-1] != v.shape[:-1]:
        raise ValueError("incompatible self-attention shapes")
    scores = q @ np.swapaxes(k, -1, -2) / math.sqrt(q.shape[-1])
    mask = np.tril(np.ones(scores.shape[-2:], dtype=bool))
    weights = masked_softmax(scores, mask)
    return weights @ v, weights


def lora_forward(x, w, a, b, alpha):
    """w: out,in; a: rank,in; b: out,rank; x: ...,in."""
    rank = a.shape[0]
    if rank < 1 or a.shape[1] != w.shape[1] or b.shape != (w.shape[0], rank):
        raise ValueError("invalid LoRA shapes")
    return x @ w.T + (alpha / rank) * ((x @ a.T) @ b.T)


def dpo_loss(chosen, rejected, ref_chosen, ref_rejected, beta=0.1):
    if beta <= 0:
        raise ValueError("beta must be positive")
    z = beta * ((np.asarray(chosen) - rejected) - (np.asarray(ref_chosen) - ref_rejected))
    return np.logaddexp(0.0, -z)


def simpo_loss(chosen_sum, rejected_sum, chosen_len, rejected_len, beta=2.0, gamma=0.5):
    if beta <= 0 or gamma < 0 or np.any(np.asarray(chosen_len) <= 0) or np.any(np.asarray(rejected_len) <= 0):
        raise ValueError("invalid scales or response lengths")
    z = beta * (np.asarray(chosen_sum) / chosen_len - np.asarray(rejected_sum) / rejected_len) - gamma
    return np.logaddexp(0.0, -z)


def grpo_advantages(rewards, eps=1e-8, scale=True):
    """Each row is one prompt's group. Population std (ddof=0), explicit choice."""
    r = np.asarray(rewards, dtype=np.float64)
    if r.ndim != 2 or r.shape[1] < 2 or not np.all(np.isfinite(r)):
        raise ValueError("expected finite rewards, shape prompts x group>=2")
    centered = r - r.mean(axis=1, keepdims=True)
    return centered / (r.std(axis=1, keepdims=True) + eps) if scale else centered


def kto_loss(log_ratio, desirable, z0=0.0, beta=0.1, lambda_d=1.0, lambda_u=1.0):
    """KTO v3 per-example values. z0 is supplied externally, not estimated here.
    A tensor/autograd implementation must detach the baseline as specified.
    """
    if min(beta, lambda_d, lambda_u) <= 0 or np.any(np.asarray(z0) < 0):
        raise ValueError("positive scales and nonnegative baseline required")
    r = np.asarray(log_ratio, dtype=float)
    good = np.asarray(desirable, dtype=bool)
    value = beta * np.where(good, r-z0, z0-r)
    sigmoid = np.exp(-np.logaddexp(0.0, -value))
    return np.where(good, lambda_d, lambda_u) * (1-sigmoid)


def ppo_surrogate(ratio, advantage, eps=0.2):
    ratio, advantage = np.asarray(ratio), np.asarray(advantage)
    if not 0 < eps < 1 or np.any(ratio < 0):
        raise ValueError("invalid clip or ratios")
    return np.minimum(ratio * advantage, np.clip(ratio, 1 - eps, 1 + eps) * advantage)


def aggregate_token_losses(losses, mask, mode="grpo", max_length=None):
    losses, mask = np.asarray(losses), np.asarray(mask, dtype=bool)
    if losses.ndim != 2 or losses.shape != mask.shape:
        raise ValueError("expected matching group x token arrays")
    lengths = mask.sum(axis=1)
    if np.any(lengths == 0):
        raise ValueError("empty response")
    sums = np.where(mask, losses, 0).sum(axis=1)
    if mode == "grpo":
        return np.mean(sums / lengths)
    if mode == "dapo":
        return sums.sum() / lengths.sum()
    if mode == "dr_grpo":
        if max_length is None or max_length < max(lengths):
            raise ValueError("max_length must cover all responses")
        return sums.sum() / (len(lengths) * max_length)
    raise ValueError("unknown mode")


def pass_at_k(n, correct, k):
    if not all(isinstance(v, (int, np.integer)) for v in (n, correct, k)) or not 0 <= correct <= n or not 1 <= k <= n:
        raise ValueError("require 0<=correct<=n and 1<=k<=n")
    if n - correct < k:
        return 1.0
    return 1.0 - math.prod((n - correct - j) / (n - j) for j in range(k))


def classification_metrics(tp, fp, fn):
    if min(tp, fp, fn) < 0:
        raise ValueError("counts must be nonnegative")
    p = tp / (tp + fp) if tp + fp else 0.0
    r = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * p * r / (p + r) if p + r else 0.0
    return {"precision": p, "recall": r, "f1": f1}


def paired_bootstrap(a, b, repetitions=5000, seed=42):
    """B minus A. One array element must be one independent sampling unit."""
    a, b = np.asarray(a, dtype=float), np.asarray(b, dtype=float)
    if a.ndim != 1 or a.shape != b.shape or not len(a) or repetitions < 1:
        raise ValueError("equal nonempty one-dimensional arrays required")
    if not np.all(np.isfinite(a)) or not np.all(np.isfinite(b)):
        raise ValueError("finite scores required")
    diff = b - a
    rng = np.random.default_rng(seed)
    estimates = np.array([diff[rng.integers(0, len(diff), size=len(diff))].mean()
                          for _ in range(repetitions)])
    low, high = np.quantile(estimates, [0.025, 0.975])
    return {"delta": float(diff.mean()), "ci95": [float(low), float(high)], "seed": seed}


def reciprocal_rank_fusion(rankings, offset=60):
    if offset < 0:
        raise ValueError("offset must be nonnegative")
    scores = defaultdict(float)
    for ranking in rankings:
        seen = set()
        for rank, doc in enumerate(ranking, 1):
            if doc not in seen:
                scores[doc] += 1.0 / (offset + rank)
                seen.add(doc)
    return sorted(scores.items(), key=lambda item: (-item[1], str(item[0])))


def kv_cache_bytes(layers, batch, tokens, kv_heads, head_dim, bytes_per_element=2):
    values = [layers, batch, tokens, kv_heads, head_dim, bytes_per_element]
    if any(not isinstance(v, (int, np.integer)) or v < 1 for v in values):
        raise ValueError("positive integers required")
    return 2 * math.prod(values)


class MockRefundService:
    """In-memory toy service: first response is lost after successful write."""
    def __init__(self):
        self.orders = {"order-demo": "paid"}
        self.requests = {}
        self.writes = 0

    def refund(self, order_id, idempotency_key):
        if idempotency_key in self.requests:
            previous_order = self.requests[idempotency_key]
            if previous_order != order_id:
                raise ValueError("idempotency key reused for another order")
            return self.orders[order_id]
        if self.orders.get(order_id) != "paid":
            raise ValueError("order missing or not refundable")
        self.orders[order_id] = "refunded"
        self.requests[idempotency_key] = order_id
        self.writes += 1
        raise TimeoutError("response lost AFTER commit")

    def status(self, order_id):
        return self.orders.get(order_id)


def recover_refund(service, order_id, idempotency_key):
    try:
        return service.refund(order_id, idempotency_key)
    except TimeoutError:
        # Real systems may be eventually consistent: use a bounded state query
        # and keep 'unknown' explicit. This mock has immediate consistency.
        current = service.status(order_id)
        if current == "refunded":
            return current
        return "unknown"
