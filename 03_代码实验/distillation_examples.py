"""Small CPU calculations for KD03/KD04/KD11; not a language-model trainer."""
import json
import numpy as np


def log_softmax(logits, temperature=1.0):
    if not np.isfinite(temperature) or temperature <= 0:
        raise ValueError("temperature must be finite and positive")
    z = np.asarray(logits, dtype=float) / temperature
    if z.ndim == 0 or not z.shape[-1] or not np.all(np.isfinite(z)):
        raise ValueError("finite logits with a nonempty vocabulary are required")
    z = z - np.max(z, axis=-1, keepdims=True)
    return z - np.log(np.sum(np.exp(z), axis=-1, keepdims=True))


def kl_from_logs(log_p, log_q):
    """KL(p || q), sum over vocabulary, preserve any leading dimensions."""
    log_p, log_q = np.asarray(log_p), np.asarray(log_q)
    if log_p.shape != log_q.shape:
        raise ValueError("probabilities must have matching shapes")
    return np.sum(np.exp(log_p) * (log_p - log_q), axis=-1)


def soft_kd(student_logits, teacher_logits, temperature=1.0):
    """Fixed teacher and fixed context: T² forward KL and gradient wrt z."""
    log_q = log_softmax(student_logits, temperature)
    log_p = log_softmax(teacher_logits, temperature)
    if log_q.ndim != 1 or log_p.shape != log_q.shape:
        raise ValueError("this teaching function expects two matching vectors")
    loss = temperature**2 * kl_from_logs(log_p, log_q)
    gradient = temperature * (np.exp(log_q) - np.exp(log_p))
    return float(loss), gradient


def masked_nll(aligned_logits, target_ids, loss_mask):
    """Rows already predict their matching targets; no implicit causal shift.

    A caller using causal LM outputs must perform the one-token shift first.
    Masked rows can have target -100. They stay out of direct loss, while in
    an actual LM those tokens can still condition subsequent predictions.
    """
    logits = np.asarray(aligned_logits, dtype=float)
    ids, mask = np.asarray(target_ids), np.asarray(loss_mask)
    if logits.ndim != 2 or ids.shape != (len(logits),) or mask.shape != ids.shape:
        raise ValueError("expected [positions, vocabulary] and position vectors")
    if not np.issubdtype(ids.dtype, np.integer) or not np.all(np.isin(mask, [0, 1])):
        raise ValueError("targets must be integers; mask must contain 0 or 1")
    active = mask.astype(bool)
    if not np.any(active):
        raise ValueError("no supervised target positions")
    if np.any(ids[active] < 0) or np.any(ids[active] >= logits.shape[1]):
        raise ValueError("supervised target outside vocabulary")
    logs = log_softmax(logits[active])
    return float(-logs[np.arange(active.sum()), ids[active]].mean())


def finite_difference(function, x, step=1e-5):
    x = np.asarray(x, dtype=float)
    grad = np.zeros_like(x)
    for i in range(x.size):
        plus, minus = x.copy(), x.copy()
        plus[i] += step
        minus[i] -= step
        grad[i] = (function(plus) - function(minus)) / (2 * step)
    return grad


def main():
    teacher, student = np.log([0.9, 0.08, 0.02]), np.log([0.6, 0.3, 0.1])
    rows = []
    for temperature in [1, 2, 4]:
        loss, grad = soft_kd(student, teacher, temperature)
        numerical = finite_difference(lambda z: soft_kd(z, teacher, temperature)[0], student)
        rows.append({"温度": temperature,
                     "教师分布": np.exp(log_softmax(teacher, temperature)).round(6).tolist(),
                     "乘T平方后的KL": round(loss, 6),
                     "梯度有限差分最大误差": float(np.max(np.abs(grad - numerical)))})
    log_p, log_q = log_softmax(teacher), log_softmax(student)
    logits = np.array([[3., 0, -1], [0, 3., -1], [1., 0, -1], [0, -1, 3.]])
    targets, mask = [0, 1, -100, 2], [1, 1, 0, 1]
    before = masked_nll(logits, targets, mask)
    changed = logits.copy()
    changed[2] = [-50, 80, -30]
    after = masked_nll(changed, targets, mask)
    report = {"说明": "固定上下文的 CPU 教学计算；不是 MiniLLM/GKD/Agent 训练结果",
              "温度实验": rows, "正向KL": float(kl_from_logs(log_p, log_q)),
              "反向KL": float(kl_from_logs(log_q, log_p)),
              "工具观察目标位置变化前的损失": before,
              "只改变被mask位置的logits后的损失": after,
              "假设成本摊平任务数": 3000 / (0.1 - (0.01 + 0.1 * 0.1))}
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
