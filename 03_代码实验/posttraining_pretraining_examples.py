"""CPU teaching calculations, not a model trainer or paper reproduction.

The OPD calculation fixes the history and teacher. It verifies only the
current-token reverse-KL gradient, not derivatives through future histories.
"""
import json
import numpy as np


def log_softmax(logits):
    z = np.asarray(logits, dtype=np.float64)
    shifted = z - np.max(z)
    return shifted - np.log(np.exp(shifted).sum())


def reverse_kl_and_grad(student_logits, teacher_logits):
    logq, logp = log_softmax(student_logits), log_softmax(teacher_logits)
    q = np.exp(logq)
    ratio = logq - logp
    kl = float(q @ ratio)
    return kl, q * (ratio - kl)


def enumerated_opd_grad(student_logits, teacher_logits):
    """Average detached-advantage score gradients over sampled-token outcomes."""
    logq, logp = log_softmax(student_logits), log_softmax(teacher_logits)
    q = np.exp(logq)
    advantage = logp - logq
    scores = np.eye(len(q)) - q[None, :]
    # q represents the sampling measure; do not differentiate this surrogate
    # again as if q, advantage and score were an ordinary scalar loss graph.
    return np.sum(q[:, None] * (-advantage[:, None] * scores), axis=0)


def gspo_ratio(token_log_ratios):
    return float(np.exp(np.mean(token_log_ratios)))


def sapo_gate(ratio, tau):
    if tau <= 0:
        raise ValueError('tau must be positive')
    x = tau * (np.asarray(ratio, dtype=np.float64) - 1)
    sigmoid = np.exp(-np.logaddexp(0, -x))
    return 4 * sigmoid / tau


def sapo_gate_derivative(ratio, tau):
    sigmoid = tau * sapo_gate(ratio, tau) / 4
    return 4 * sigmoid * (1 - sigmoid)


def gdpo_advantages(rewards, epsilon=1e-12):
    """Teaching version: [prompt, group sample, reward dimension].

    Uses population std, epsilon and equal reward weights explicitly.
    Frameworks may choose other numerical conventions; record those choices.
    """
    r = np.asarray(rewards, dtype=np.float64)
    if r.ndim != 3 or r.shape[1] < 2:
        raise ValueError('expected [prompts, at least 2 samples, rewards]')
    a = (r - r.mean(axis=1, keepdims=True)) / (r.std(axis=1, keepdims=True) + epsilon)
    combined = a.sum(axis=2)
    return (combined - combined.mean()) / (combined.std() + epsilon)


def causal_training_example(tokens):
    if len(tokens) < 2:
        raise ValueError('need at least two tokens')
    inputs, targets = list(tokens[:-1]), list(tokens[1:])
    # At input position i, predict original token i+1. Self-attention to i
    # is allowed; access to the target's input position i+1 is blocked.
    allowed = np.tril(np.ones((len(inputs), len(inputs)), dtype=bool))
    return inputs, targets, allowed


def token_weighted_mean(batch_means, valid_counts):
    counts = np.asarray(valid_counts, dtype=np.float64)
    means = np.asarray(batch_means, dtype=np.float64)
    if counts.shape != means.shape or np.any(counts < 0) or counts.sum() == 0:
        raise ValueError('invalid token counts')
    return float(np.sum(means * counts) / counts.sum())


def training_budget(data_parallel, microbatch, accumulation, length, steps, parameters):
    tokens_per_update = data_parallel * microbatch * accumulation * length
    total_tokens = tokens_per_update * steps
    return dict(tokens_per_update=tokens_per_update, total_tokens=total_tokens,
                approximate_dense_flops=6 * parameters * total_tokens)


def examples():
    z = np.array([0.3, -0.2, 0.8])
    teacher = np.array([0.7, 0.4, -0.1])
    kl, grad = reverse_kl_and_grad(z, teacher)
    pg = enumerated_opd_grad(z, teacher)
    inputs, targets, mask = causal_training_example(['我', '喜欢', '机器', '学习'])
    rewards = [[[0, 0], [50, 0], [100, 1]], [[25, 1], [75, 0], [50, 1]]]
    return {
        'opd_advantage_log_0_5_over_0_2': float(np.log(0.5 / 0.2)),
        'reverse_kl_fixed_history': kl,
        'direct_gradient': grad.tolist(),
        'enumerated_policy_gradient': pg.tolist(),
        'gradient_max_absolute_difference': float(np.max(np.abs(grad - pg))),
        'gspo_geometric_ratio_2_and_half': gspo_ratio(np.log([2, 0.5])),
        'arithmetic_ratio_for_comparison': float(np.mean([2, 0.5])),
        'sapo_derivatives_at_one': {str(t): float(sapo_gate_derivative(1, t)) for t in [0.5, 1, 2]},
        'gdpo_example_advantages': gdpo_advantages(rewards).tolist(),
        'ntp_inputs': inputs, 'ntp_targets': targets, 'allowed_attention': mask.astype(int).tolist(),
        'weighted_mean_loss': token_weighted_mean([1, 3], [2, 6]),
        'batch_budget': training_budget(8, 2, 4, 1024, 1000, 10**9),
        'one_billion_parameters_twenty_billion_tokens_flops': 6 * 10**9 * (20 * 10**9),
        'scope': 'CPU numerical teaching only; no model or teacher execution.'
    }


if __name__ == '__main__':
    print(json.dumps(examples(), ensure_ascii=False, indent=2))
