"""从零拼出 Transformer 的 CPU 教学实现。运行本文件可做一次小规模学习实验。

不使用 nn.Transformer / nn.MultiheadAttention 隐藏核心步骤。
统一张量顺序 [batch, tokens, channels]；allowed=True 表示可以关注。
各 # region 区间与第 21 章逐节对应。性能优化不是本实现的目的。
"""
import math
import json
import torch
from torch import nn
from torch.nn import functional as F


# region shapes
def split_heads(x, heads):
    batch, tokens, channels = x.shape
    if channels % heads:
        raise ValueError("channels 必须能被 heads 整除")
    # reshape 只拆最后一维；transpose 才把头移到前面。
    return x.reshape(batch, tokens, heads, channels // heads).transpose(1, 2)


def merge_heads(x):
    batch, heads, tokens, width = x.shape
    return x.transpose(1, 2).contiguous().view(batch, tokens, heads * width)
# endregion shapes


# region attention
def attention(q, k, v, allowed=None, dropout_p=0.0, training=False):
    # q: [B,H,Tq,Dh]；k/v: [B,H,Tk,Dh]（v 的末维也可不同）
    scores = q @ k.transpose(-2, -1) / math.sqrt(q.size(-1))
    if allowed is not None:
        if allowed.dtype != torch.bool:
            raise TypeError("本教程只接受 bool mask；True = 可以看")
        allowed = torch.broadcast_to(allowed, scores.shape)
        if not allowed.any(dim=-1).all():
            raise ValueError("存在完全看不到任何 key 的行；请检查 padding/mask")
        scores = scores.masked_fill(~allowed, float("-inf"))
    weights = torch.softmax(scores, dim=-1)  # 沿 key 轴分配权重
    used = F.dropout(weights, p=dropout_p, training=training)
    return used @ v, weights  # 返回 dropout 前权重，便于教学验算行和
# endregion attention


# region masks
def causal_allowed(query_len, key_len, past_len=0, device=None):
    # 缓存后 query 的绝对位置从 past_len 开始。
    query_pos = past_len + torch.arange(query_len, device=device)
    key_pos = torch.arange(key_len, device=device)
    return key_pos[None, :] <= query_pos[:, None]


def padding_allowed(valid_keys):
    # valid_keys: [B,Tk]；扩展后能广播到 [B,H,Tq,Tk]。
    return valid_keys[:, None, None, :]
# endregion masks


# region mha
class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, heads, dropout=0.0):
        super().__init__()
        if d_model % heads:
            raise ValueError("d_model 必须能被 heads 整除")
        self.heads, self.dropout = heads, dropout
        self.q_proj = nn.Linear(d_model, d_model)
        self.k_proj = nn.Linear(d_model, d_model)
        self.v_proj = nn.Linear(d_model, d_model)
        self.out_proj = nn.Linear(d_model, d_model)

    def forward(self, x, context=None, allowed=None, causal=False,
                cache=None, use_cache=False):
        if context is not None and (cache is not None or use_cache):
            raise ValueError("这个教学缓存仅支持 self-attention")
        source = x if context is None else context
        q = split_heads(self.q_proj(x), self.heads)
        k = split_heads(self.k_proj(source), self.heads)
        v = split_heads(self.v_proj(source), self.heads)
        past = 0
        if cache is not None:
            past = cache[0].size(-2)
            k = torch.cat([cache[0], k], dim=-2)
            v = torch.cat([cache[1], v], dim=-2)
        if causal:
            causal_mask = causal_allowed(q.size(-2), k.size(-2), past, x.device)
            allowed = causal_mask if allowed is None else allowed & causal_mask
        y, weights = attention(q, k, v, allowed, self.dropout, self.training)
        y = self.out_proj(merge_heads(y))
        return y, weights, (k, v) if use_cache else None
# endregion mha


# region norm
class LayerNorm(nn.Module):
    def __init__(self, width, eps=1e-5):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(width))
        self.bias = nn.Parameter(torch.zeros(width))
        self.eps = eps

    def forward(self, x):
        mean = x.mean(dim=-1, keepdim=True)
        variance = (x - mean).square().mean(dim=-1, keepdim=True)
        return (x - mean) * torch.rsqrt(variance + self.eps) * self.weight + self.bias


class RMSNorm(nn.Module):
    def __init__(self, width, eps=1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(width))
        self.eps = eps

    def forward(self, x):
        return x * torch.rsqrt(x.square().mean(-1, keepdim=True) + self.eps) * self.weight
# endregion norm


# region ffn
class FeedForward(nn.Module):
    def __init__(self, width, hidden):
        super().__init__()
        self.up = nn.Linear(width, hidden)
        self.down = nn.Linear(hidden, width)

    def forward(self, x):
        return self.down(F.gelu(self.up(x)))


class SwiGLU(nn.Module):
    def __init__(self, width, hidden):
        super().__init__()
        self.gate = nn.Linear(width, hidden, bias=False)
        self.up = nn.Linear(width, hidden, bias=False)
        self.down = nn.Linear(hidden, width, bias=False)

    def forward(self, x):
        return self.down(F.silu(self.gate(x)) * self.up(x))
# endregion ffn


# region positions
def sinusoidal_positions(length, width, device=None):
    if width % 2:
        raise ValueError("这个简化正弦实现要求 width 为偶数")
    pos = torch.arange(length, device=device).float()[:, None]
    freq = torch.exp(torch.arange(0, width, 2, device=device).float()
                     * (-math.log(10000.0) / width))
    pe = torch.zeros(length, width, device=device)
    pe[:, 0::2], pe[:, 1::2] = torch.sin(pos * freq), torch.cos(pos * freq)
    return pe  # [T,D]，可广播加到 [B,T,D]
# endregion positions


# region block
class TransformerBlock(nn.Module):
    """Pre-LN：先归一化再进子层；不同于原始论文的 Post-LN 布局。"""
    def __init__(self, width, heads, hidden, dropout=0.0):
        super().__init__()
        self.norm1, self.norm2 = LayerNorm(width), LayerNorm(width)
        self.attn = MultiHeadAttention(width, heads, dropout)
        self.ffn = FeedForward(width, hidden)
        self.drop = nn.Dropout(dropout)

    def forward(self, x, allowed=None, causal=False, cache=None, use_cache=False):
        update, _, new_cache = self.attn(self.norm1(x), allowed=allowed,
                                         causal=causal, cache=cache, use_cache=use_cache)
        x = x + self.drop(update)
        x = x + self.drop(self.ffn(self.norm2(x)))
        return x, new_cache
# endregion block


# region decoder
class DecoderBlock(nn.Module):
    """翻译式 decoder：因果 self-attention → cross-attention → FFN。"""
    def __init__(self, width, heads, hidden):
        super().__init__()
        self.norms = nn.ModuleList([LayerNorm(width) for _ in range(3)])
        self.self_attn = MultiHeadAttention(width, heads)
        self.cross_attn = MultiHeadAttention(width, heads)
        self.ffn = FeedForward(width, hidden)

    def forward(self, x, memory, source_valid=None, target_valid=None):
        target_mask = None if target_valid is None else padding_allowed(target_valid)
        update, _, _ = self.self_attn(self.norms[0](x), allowed=target_mask, causal=True)
        x = x + update
        source_mask = None if source_valid is None else padding_allowed(source_valid)
        update, _, _ = self.cross_attn(self.norms[1](x), context=memory, allowed=source_mask)
        x = x + update
        return x + self.ffn(self.norms[2](x))
# endregion decoder


# region lm
class TinyLanguageModel(nn.Module):
    def __init__(self, vocab_size=8, width=24, heads=3, layers=2, max_len=64):
        super().__init__()
        self.token_emb = nn.Embedding(vocab_size, width)
        self.pos_emb = nn.Embedding(max_len, width)
        self.blocks = nn.ModuleList([TransformerBlock(width, heads, 4 * width)
                                     for _ in range(layers)])
        self.norm = LayerNorm(width)
        self.lm_head = nn.Linear(width, vocab_size, bias=False)
        self.max_len = max_len

    def forward(self, ids, caches=None, use_cache=False):
        # 教学 LM 只接收无 padding 的等长序列；batch 内缓存长度相同。
        if caches is not None and len(caches) != len(self.blocks):
            raise ValueError("每一层都需要自己的 KV cache")
        past = 0 if caches is None else caches[0][0].size(-2)
        if past + ids.size(1) > self.max_len:
            raise ValueError("超过位置表长度，请用更短的序列")
        positions = torch.arange(past, past + ids.size(1), device=ids.device)
        x = self.token_emb(ids) + self.pos_emb(positions)
        new_caches = []
        for i, block in enumerate(self.blocks):
            x, cache = block(x, causal=True, cache=None if caches is None else caches[i],
                             use_cache=use_cache)
            new_caches.append(cache)
        return self.lm_head(self.norm(x)), new_caches if use_cache else None
# endregion lm


# region seq2seq
class TinySeq2Seq(nn.Module):
    def __init__(self, vocab_size=12, width=16, heads=2, layers=2, max_len=32):
        super().__init__()
        self.source_emb = nn.Embedding(vocab_size, width)
        self.target_emb = nn.Embedding(vocab_size, width)
        self.register_buffer("positions", sinusoidal_positions(max_len, width))
        self.encoders = nn.ModuleList([TransformerBlock(width, heads, 4*width) for _ in range(layers)])
        self.decoders = nn.ModuleList([DecoderBlock(width, heads, 4*width) for _ in range(layers)])
        self.encoder_norm, self.decoder_norm = LayerNorm(width), LayerNorm(width)
        self.output = nn.Linear(width, vocab_size)

    def forward(self, source_ids, target_input, source_valid=None, target_valid=None):
        # 目标已在调用处右移；target_input[0] 通常是 BOS。
        memory = self.source_emb(source_ids) + self.positions[:source_ids.size(1)]
        mask = None if source_valid is None else padding_allowed(source_valid)
        for block in self.encoders:
            memory, _ = block(memory, allowed=mask)
        memory = self.encoder_norm(memory)
        x = self.target_emb(target_input) + self.positions[:target_input.size(1)]
        for block in self.decoders:
            x = block(x, memory, source_valid, target_valid)
        return self.output(self.decoder_norm(x))
# endregion seq2seq


# region loss
def next_token_loss(model, token_ids):
    # [0,1,2,3] -> 输入 [0,1,2]，标签 [1,2,3]；只移位一次。
    if token_ids.size(1) < 2:
        raise ValueError("至少两个 token 才能构造下一 token 标签")
    inputs, labels = token_ids[:, :-1], token_ids[:, 1:]
    logits, _ = model(inputs)
    return F.cross_entropy(logits.reshape(-1, logits.size(-1)), labels.reshape(-1))
# endregion loss


# region generation
@torch.no_grad()
def generate(model, prompt, steps, use_cache=True):
    was_training = model.training
    model.eval()
    try:
        ids, caches = prompt.clone(), None
        for _ in range(steps):
            current = ids[:, -1:] if use_cache and caches is not None else ids
            logits, caches = model(current, caches=caches, use_cache=use_cache)
            next_id = logits[:, -1].argmax(dim=-1, keepdim=True)  # 贪心选择
            ids = torch.cat([ids, next_id], dim=1)
        return ids
    finally:
        model.train(was_training)
# endregion generation


# region rope
def apply_rope(x, positions, base=10000.0):
    # 相邻两个通道为一对；有的模型使用前半/后半配对，不能混用权重约定。
    width = x.size(-1)
    if width % 2:
        raise ValueError("RoPE 头维度必须为偶数")
    freq = base ** (-torch.arange(0, width, 2, device=x.device).float() / width)
    angle = positions.to(x.device).float()[:, None] * freq
    cos, sin = angle.cos().to(x.dtype), angle.sin().to(x.dtype)
    even, odd = x[..., 0::2], x[..., 1::2]
    return torch.stack([even*cos - odd*sin, even*sin + odd*cos], dim=-1).flatten(-2)
# endregion rope


# region gqa
def expand_kv(k, v, query_heads):
    kv_heads = k.size(1)
    if v.size(1) != kv_heads or query_heads % kv_heads:
        raise ValueError("Q 头数必须是相同 K/V 头数的整数倍")
    group = query_heads // kv_heads
    # 为看清语义先显式复制；生产内核可避免物理展开。
    return k.repeat_interleave(group, dim=1), v.repeat_interleave(group, dim=1)
# endregion gqa


# region training
def learning_demo(steps=120):
    torch.manual_seed(17)
    torch.set_num_threads(1)
    model = TinyLanguageModel()
    # 玩具规律：0 -> 1 -> ... -> 7 -> 0；不是自然语言语料。
    offsets = torch.arange(8)[:, None]
    batch = (offsets + torch.arange(13)[None, :]) % 8
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.01, weight_decay=0.0)
    history = []
    for step in range(steps):
        optimizer.zero_grad(set_to_none=True)
        loss = next_token_loss(model, batch)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        if step in (0, steps - 1):
            history.append(float(loss.detach()))
    prediction = generate(model, torch.tensor([[0, 1, 2]]), 6).tolist()[0]
    return {"torch": torch.__version__, "device": "cpu", "steps": steps,
            "first_loss": history[0], "last_loss": history[-1],
            "generated": prediction, "expected": [0,1,2,3,4,5,6,7,0],
            "scope": "玩具循环序列的过拟合自检；不代表语言理解与泛化能力"}
# endregion training


if __name__ == "__main__":
    print(json.dumps(learning_demo(), ensure_ascii=False, indent=2))
