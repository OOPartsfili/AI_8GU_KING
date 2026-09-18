# Transformer 架构手写：从一行向量到完整模型

**学习目标：**新人能在纸上说明形状，逐步写出核心部件，运行小语言模型，并解释常见错误。TH01—TH16 是主线；TH17—TH20 是缓存、现代变体与验证；TH21—TH22 用于面试练习和学习迁移。

建议分 5 次练：① TH01—TH05；② TH06—TH10；③ TH11—TH14；④ TH15—TH17；⑤ TH18—TH22。每次先自己预测输出，再运行；不要求第一天记住全部代码。

配套 [完整实现](../03_代码实验/transformer_handwrite.py)、[数值与行为测试](../03_代码实验/test_transformer_handwrite.py)、[安装、运行和分层练习](../03_代码实验/Transformer手写运行与练习.md)。阅读器顶部有“Transformer 手写”和“算法可视化”入口。

**统一约定：**Python + PyTorch；[B,T,D]；True=允许关注；float32 CPU；主线使用 Pre-LN、GELU 与可训练绝对位置。原始 Post-LN、正弦位置、RoPE、RMSNorm、SwiGLU、GQA 在相应位置区分，不混称同一个模型。示例小模型不是现成预训练模型。

<a id="TH01"></a>
## TH01｜这章从哪里写起，最后要能写出什么？（P0）

你面前是一张白纸，面试官说“写一个 Transformer”。先别背两百行代码：先确认要单头注意力、多头注意力、一层 Block，还是完整编码器—解码器。

本章用同一套小积木逐步拼装：**形状 → 注意力 → mask → 多头 → 归一化与 FFN → Block → 模型 → loss → 生成与缓存**。主线是便于练习的 Pre-LN 小语言模型；同时给出翻译式 Encoder–Decoder。最后再看 RoPE、GQA 等变体。

默认例子是 2 句话、每句 3 个 token、每个 token 8 个数。先读 TH01—TH06，能够解释每行的输入输出，再向后走。

**直答：**先约定输入输出与 mask 语义，再逐块实现并验证；代码写完还要证明形状、数值、梯度与因果性正确。

### 跟着写

```python
import torch
from torch import nn
from torch.nn import functional as F

torch.manual_seed(9)
x = torch.randn(2, 3, 8)
print(x.shape)  # torch.Size([2, 3, 8])
```

### 进一步：公式、边界与追问

**练习：**用一句话解释三个数字，不能只说“B、T、D”。

**答案：**2 个独立样本；每个样本有 3 个位置；每个位置用 8 维向量表示。批次维不会让两句话互相读到内容。

**追问：能只复制每题代码运行吗？**短例子可在完整模块旁导入相关函数；较长类依赖前面的函数。配套 `transformer_handwrite.py` 已按本章顺序合并，`Transformer手写运行与练习.md` 给出从安装到运行的方法。网页中的动画执行小型数值计算，不执行 Python，也不下载模型。

参考 Happy-LLM 的基础到实现顺序、LLMs-from-scratch 的分模块练习，以及面试指南的连续追问组织；本章例子与实现重新编写。

依据：S01、S132、S134、S135。

<a id="TH02"></a>
## TH02｜B、T、D、H、Dh：怎样先把形状写对？（P0）

把一个 token 的 8 个数分给 2 个头，每个头拿 4 个数。词的位置没有改变，改变的是“按哪个头组织这些数”。不能把 reshape 当成任意换轴。

请先画四个标签：句子、头、词的位置、每头通道。可视化的“拆头与合头”会显示每个数字搬到了哪里。

**直答：**本章从 [B,T,D] 出发，先拆成 [B,T,H,Dh]，再换轴到 [B,H,T,Dh]；D=H×Dh 是这套实现的约束。

### 跟着写

```python
def split_heads(x, heads):
    batch, tokens, channels = x.shape
    if channels % heads:
        raise ValueError("channels 必须能被 heads 整除")
    # reshape 只拆最后一维；transpose 才把头移到前面。
    return x.reshape(batch, tokens, heads, channels // heads).transpose(1, 2)


def merge_heads(x):
    batch, heads, tokens, width = x.shape
    return x.transpose(1, 2).contiguous().view(batch, tokens, heads * width)
```

### 进一步：公式、边界与追问

**练习：**对 `torch.arange(24).reshape(1,3,8)` 拆成 2 个头，预测第 2 个头、第 2 个词会拿到哪些数。

**答案：**原词向量是 8 到 15，第二头拿 12、13、14、15；合头后必须逐元素回到原张量。

**追问：为什么 transpose 后常接 contiguous？**换轴可以只改访问步长，底层数字尚未重排；`view` 要求相容的内存布局。这里先连续化再合并。`reshape` 可能在需要时复制，不能假定所有换形状都免费。

**边界：**有些架构的总注意力通道数不等于残差宽度。本章先采用最常见的等宽手写版本；不要把 D=H×Dh 推成所有模型的硬规则。

依据：S01、S129。

<a id="TH03"></a>
## TH03｜Embedding 和 Q/K/V 投影到底怎样写？（P0）

“猫”的编号是 2，不代表猫比编号 1 的“狗”大。Embedding 是按编号取一行可训练向量；线性层再把这行向量变成用于匹配或传递信息的数。

像同一个人分别写“我想找什么”（Q）、“我提供什么线索”（K）、“可以取走什么内容”（V）；这些只是帮助理解作用，真正的向量由训练学得。

**直答：**Embedding 是查表；Q/K/V 是三套独立投影。本章 nn.Linear 的权重存成 [输出维,输入维]，实际运算是 x @ weight.T + bias。

### 跟着写

```python
embedding = nn.Embedding(10, 8)
ids = torch.tensor([[2, 5, 2]])
x = embedding(ids)               # [1,3,8]
q_proj = nn.Linear(8, 8)
q = q_proj(x)                    # [1,3,8]
manual = x @ q_proj.weight.T + q_proj.bias
torch.testing.assert_close(q, manual)
torch.testing.assert_close(x[:, 0], x[:, 2])
```

### 逐行拆解

1. `ids` 是整数编号，不能拿编号大小当语义距离。
2. `embedding(ids)` 一次取出每个位置的一行，所以多出通道维 8。
3. `nn.Linear(8,8)` 创建可训练权重和偏置；调用它才执行乘法。
4. 最后两条断言分别检查线性运算和同编号查表是否一致。

### 进一步：公式、边界与追问

**练习：**词表 10、宽度 8，Embedding 有几个参数？有 bias 的 8→8 线性层呢？

**答案：**分别是 80 和 64+8=72。Embedding 通常不需要在运行时真的构造巨大 one-hot。

**追问：同一个 token 每次向量都相同吗？**这里的原始查表结果相同；加入位置并经过注意力后，其上下文表示通常不同。不要把“查表向量”与“深层隐藏状态”混为一谈。

依据：S01。

<a id="TH04"></a>
## TH04｜单头注意力：五行核心计算怎样理解？（P0）

只有两个可选位置，V 分别是 [2,0] 与 [0,4]。如果注意力权重是 0.67 和 0.33，输出约为 [1.34,1.32]：从两份内容按比例取信息。

先写 Q 与 K 的匹配分数，再把一整行变成分配比例，最后加权 V。softmax 放错轴，代码也可能运行，却变成另一个算法。

**直答：**核心是 QK 转置打分、除以头维度平方根、遮住禁看位置、沿 key 轴 softmax、再乘 V。

### 跟着写

```python
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
```

### 逐行拆解

1. `q @ k.transpose(-2,-1)`：把最后两维转置，才得到每个 query 与每个 key 的两两打分。
2. 除以头维度平方根，让维度增加时打分尺度更可控。
3. 用负无穷替换禁看位置，使其 softmax 权重变为 0。
4. `dim=-1` 沿 key 轴分配比例；`weights @ v` 才把比例变成取出的内容。

### 进一步：公式、边界与追问

**练习：**令 Q=[1,0]，K=[[1,0],[0,1]]。预测输出大概更靠近哪份 V，再运行 `test_attention_weighted_sum`。

**答案：**第一份。缩放后分数约 [0.707,0]，权重约 [0.670,0.330]，输出约 [1.340,1.321]。

**追问：返回的 weights 加起来一定是 1 吗？**这里返回 dropout 前权重；每个可用行的和约为 1。训练中实际乘 V 的权重经过 dropout，单次采样后的行和未必等于 1。

**边界：**教学实现显式生成打分矩阵，空间随序列长度平方增长。生产内核可以采用更节省中间存储的实现。

依据：S01、S128。

<a id="TH05"></a>
## TH05｜因果 mask 与 padding mask 怎样组合？（P0）

让模型根据“我 爱”预测下一个词时，不能让它在输入中先看到答案。因果 mask 遮住未来；padding mask 遮住补齐用的空位置，两者解决不同问题。

**直答：**先统一 True 的含义，再按“既非 padding，又不在未来”组合允许矩阵。本章约定 True = 可以看。

### 跟着写

```python
def causal_allowed(query_len, key_len, past_len=0, device=None):
    # 缓存后 query 的绝对位置从 past_len 开始。
    query_pos = past_len + torch.arange(query_len, device=device)
    key_pos = torch.arange(key_len, device=device)
    return key_pos[None, :] <= query_pos[:, None]


def padding_allowed(valid_keys):
    # valid_keys: [B,Tk]；扩展后能广播到 [B,H,Tq,Tk]。
    return valid_keys[:, None, None, :]
```

### 进一步：公式、边界与追问

**练习：**画出长度 3 的因果矩阵。第一行能看谁？缓存已有 3 个词、当前新来 2 个词时又如何？

**答案：**普通矩阵三行是 100、110、111。缓存后的两行是 11110、11111；query 绝对位置从 3 开始，不能直接取一个左上角的 2×5 下三角。

**追问：给 nn.MultiheadAttention 可以原样传这个 bool mask 吗？**不可以，其屏蔽型 bool mask 用 True 表示禁止。我们的 allowed 与 PyTorch SDPA 的 bool 约定相同。换接口要重新核对；不能根据变量名猜。

**边界：**本实现遇到整行不可见直接报错，避免 softmax 全为负无穷后出现 NaN。先练习右 padding、有效开头的样本。padding query 的输出未被自动清零，训练时仍需忽略对应标签；key mask 与 loss mask 不能互相替代。

依据：S128、S129。

<a id="TH06"></a>
## TH06｜多头注意力：怎样把前面几块拼在一起？（P0）

两位读者各自用一套线索看同一句话，然后合并各自找出的内容。代码不是“把单头结果复制两份”，而是把投影通道分组，独立计算注意力，再拼回去。

先忽略 cache 三行，把 `forward` 按“投影 → 拆头 → 注意力 → 合头 → 输出投影”默写一遍。TH17 再回到缓存。

**直答：**每个头都在 token 之间做注意力；合头只拼通道，最后的 out_proj 负责混合来自不同头的信息。

### 跟着写

```python
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
```

### 逐行拆解

1. `__init__` 创建四个有参数的层；它们会被优化器更新。
2. `source=x` 是自注意力；给定 context 时，K/V 从另一个序列来。
3. 三次投影后调用拆头函数；头在第二维，词的位置在第三维。
4. 第一遍学习先假设 cache=None；因果遮罩按当前 query/key 长度创建。
5. `attention` 负责打分、分配和取内容；`merge_heads` 合通道；`out_proj` 混合各头结果。
6. 返回三件东西：输出、用于检查的权重、可选的新缓存。后两件不是新的可训练参数。

### 进一步：公式、边界与追问

**练习：**输入 [2,3,8]、2 头，分别写出 Q、scores、合头后输出的形状。

**答案：**[2,2,3,4]、[2,2,3,3]、[2,3,8]。若是 cross-attention 且来源有 5 个位置，scores 变成 [2,2,3,5]。

**追问：头越多越好吗？**固定宽度时每头会更窄，表达与开销的变化需要任务验证。“一个头天然负责语法”也不是由代码保证的。

**核对：**测试把 Q/K/V 与输出投影权重复制到 PyTorch 参考层，再比较 self/cross 的输出及每头权重，避免只检查张量形状。

依据：S01、S129。

<a id="TH07"></a>
## TH07｜LayerNorm 和 RMSNorm：为什么只沿最后一维算？（P0）

每个词的一行数可能整体偏大。LayerNorm 在这一行里先减平均数，再除以标准差；它没有拿另一句话或另一个词来凑平均数。RMSNorm 则不做减均值。

**直答：**这两个模块按 token 的通道维归一化，形状保持 [B,T,D]；它们的统计量与可训练缩放参数需要分清。

### 跟着写

```python
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
```

### 进一步：公式、边界与追问

**练习：**把第 3 个词的输入加 99，前两个词的归一化输出是否应该改变？

**答案：**不应该，这是按位置独立操作。LayerNorm 的方差分母为 D，手写时别误用默认的样本方差 D−1。

**追问：RMSNorm 就是没有 bias 的 LayerNorm 吗？**还少了减均值这一步。这里为 float32/float64 教学写法；大规模混合精度常需单独考虑统计量的计算精度。

依据：S56、S131。

<a id="TH08"></a>
## TH08｜FFN 与 SwiGLU：为什么“放大再缩小”不是白忙？（P0）

每个词已经从其他词收集了信息，接着需要加工自己的那一行表示。8→32→8 的中间层先建立更多特征，再组合回来；中间有非线性，不能把两层简单合并成一层。

**直答：**FFN 逐 token 加工通道，不直接混合词的位置。SwiGLU 用一条经过 SiLU 的支路调制另一条支路，再投影回来。

### 跟着写

```python
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
```

### 进一步：公式、边界与追问

**练习：**没有激活的 `down(up(x))` 能否合并？有 GELU 时呢？

**答案：**前者可合并成一个仿射变换，后者通常不能。SwiGLU 还有逐元素乘法，也不是简单线性叠加。

**追问：同样 hidden=32 就是公平比较吗？**不一定。普通 FFN 有两组矩阵，SwiGLU 有三组，比较时要控制参数量或计算预算。这里保留可读的分支结构。

依据：S01、S57。

<a id="TH09"></a>
## TH09｜位置编码：怎样让模型区分“我爱你”和“你爱我”？（P0）

把词向量当作资料卡，只看卡片内容并不能直接得到绝对次序。位置编码给每张卡补一个位置标记；本节先实现不需要训练的位置表。

**直答：**正弦位置表按位置产生 [T,D]，广播加到 token 向量上；可训练位置表则用 nn.Embedding(max_len,D)。两者要选清楚。

### 跟着写

```python
def sinusoidal_positions(length, width, device=None):
    if width % 2:
        raise ValueError("这个简化正弦实现要求 width 为偶数")
    pos = torch.arange(length, device=device).float()[:, None]
    freq = torch.exp(torch.arange(0, width, 2, device=device).float()
                     * (-math.log(10000.0) / width))
    pe = torch.zeros(length, width, device=device)
    pe[:, 0::2], pe[:, 1::2] = torch.sin(pos * freq), torch.cos(pos * freq)
    return pe  # [T,D]，可广播加到 [B,T,D]
```

### 进一步：公式、边界与追问

**练习：**位置 0 的偶数通道与奇数通道分别是什么？

**答案：**sin(0)=0、cos(0)=1。不同通道使用不同变化频率。

**追问：本章最终模型用哪种？**TinyLanguageModel 用可训练绝对位置；TinySeq2Seq 用正弦位置。RoPE 在 TH18 单独实现，它旋转 Q/K，并非把这个正弦表再加一次。每种示例都明确约定，不能随意混装。

依据：S01。

<a id="TH10"></a>
## TH10｜Transformer Block：残差和归一化放在哪里？（P0）

把一层想成“收集信息”和“加工信息”两道工序。残差保留进入工序之前的表示，把新得到的修正加上去，所以子层输出必须回到原宽度。

**直答：**本章采用 Pre-LN：x 加上 Attention(Norm(x))，再加上 FFN(Norm(x))。残差连接不是把两个向量拼接。

### 跟着写

```python
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
```

### 逐行拆解

1. norm1 与 norm2 是两套参数，不是同一个归一化层反复用。
2. 注意力读入归一化的 x，但残差加回的是归一化之前的 x。
3. FFN 前再次归一化，再加回第二条残差。
4. Dropout 在训练和评估模式下行为不同；学习基本数据流时先设 dropout=0。

### 进一步：公式、边界与追问

**练习：**把所有 attention 和 FFN 参数置零，这一层的输出应是什么？

**答案：**残差保留原 x。若写成只返回子层输出，就失去了这条路径。

**追问：这就是原始论文的精确架构吗？**原论文是 Post-LN 布局，且 FFN 等细节不同。本章用 Pre-LN、GELU 的教学变体，并在堆叠末端补最终归一化；面试要求哪一种，就按哪一种画图并实现。

依据：S01、S134。

<a id="TH11"></a>
## TH11｜Encoder、Decoder-only 和 Encoder–Decoder 怎样分清？（P0）

理解一段完整输入、接着前文写下一个词、把中文翻成英文，这三种任务的“能看谁”不同。先画可见关系，比先背模型名字有效。

**直答：**Encoder 通常双向看来源序列；Decoder-only 因果地看自身前文；翻译式 Decoder 还要通过 cross-attention 读取 Encoder 输出。

### 跟着写

```python
encoder = TransformerBlock(8, 2, 32)
memory, _ = encoder(torch.randn(2, 5, 8), causal=False)
language_block = TransformerBlock(8, 2, 32)
hidden, _ = language_block(torch.randn(2, 3, 8), causal=True)
print(memory.shape, hidden.shape)  # [2,5,8] 与 [2,3,8]
```

### 进一步：公式、边界与追问

**练习：**英文目标有 3 个词，中文来源有 5 个词，cross-attention 打分矩阵是 3×3 还是 3×5？

**答案：**3×5；Q 来自目标，K/V 来自来源。

**追问：Decoder-only 少了 cross-attention 就等于原 Decoder 吗？**还需明确位置、归一化、FFN 等具体实现。架构分类描述信息流，不代表每个细节一样。

依据：S01。

<a id="TH12"></a>
## TH12｜手写翻译式 Decoder：为什么需要第二种注意力？（P0）

写英文第 3 个词时，先看已经写出的英文，再去完整中文来源找信息。第一步不能偷看未来英文；第二步可以读取允许的所有中文位置。

**直答：**翻译式 Decoder 一层有因果自注意力、交叉注意力、FFN 三个子层，各自有残差。

### 跟着写

```python
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
```

### 进一步：公式、边界与追问

**练习：**交换 cross-attention 的 x 与 memory，会发生什么？

**答案：**输出位置数跟随 Q，会变成按来源位置输出，偏离当前要预测目标词的任务。

**追问：source padding 与 target padding 能共用一张 mask 吗？**通常不能，它们长度与可见关系不同。目标还要与因果约束合并。源码的缓存功能只给自注意力教学用；这里没有实现 cross KV 缓存。

依据：S01。

<a id="TH13"></a>
## TH13｜完整小语言模型：怎样串起 Embedding、层堆叠和词表预测？（P0）

假设词表只有 8 个符号。每个位置先拿到 24 维内部表示，经过 2 层处理，最后输出 8 个分数，表示下一符号的不同候选。内部维度与词表大小是两件事。

**直答：**先把 token 和位置变成向量，经过独立参数的多层 Block，再映射到词表 logits；这里返回分数，不提前 softmax。

### 跟着写

```python
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
```

### 逐行拆解

1. token_emb 和 pos_emb 是两张不同的查表参数；一张表示符号，一张表示位置。
2. ModuleList 把每个独立创建的 Block 登记成模型子模块。
3. positions 在整个 batch 间广播，但不会混合不同样本。
4. 每次循环把上一层输出交给下一层，宽度保持不变。
5. lm_head 把内部宽度投影成词表大小；logits 的每个数还不是概率。
6. 初次阅读设 caches=None；学习 TH17 时再追踪各层独立缓存与位置偏移。

### 进一步：公式、边界与追问

**练习：**输入 [2,5] 的 token 编号，模型输出是什么形状？两层如何保证不是同一层重复引用？

**答案：**logits 是 [2,5,8]；用列表推导每次创建新的 Block。写 `[block]*2` 会重复引用同一个对象。

**追问：参数与激活有什么区别？**Embedding 表、投影矩阵属于可学习参数；这次前向得到的 x 和 logits 是激活。ModuleList 使每层参数被模块正确登记；普通 Python 列表不会自动登记其中的子模块。

**运行边界：**这个 LM 只支持无 padding、批内等长的输入；最大位置长度为 64；没有声称具备真实语言能力。

依据：S01、S134。

<a id="TH14"></a>
## TH14｜完整 Encoder–Decoder：怎样把两种模块连接起来？（P0）

中文整句先编码成 memory；英文输入以 BOS 起头，逐位置预测目标标签。把这个整体跑通，才能确认你会的是完整 Transformer，而不仅是一段 Attention。

**直答：**Encoder 处理来源，Decoder 使用右移后的目标输入并读取来源 memory，最后输出每个目标位置的词表分数。

### 跟着写

```python
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
```

### 进一步：公式、边界与追问

**练习：**source_ids=[2,5]，target_input=[2,3]，词表 12，输出形状是什么？

**答案：**[2,3,12]。目标长度控制输出位置数，来源长度只影响 cross-attention 的 key 数。

**追问：padding 目标怎样计算 loss？**把 padding 标签设为忽略编号，并在交叉熵中用相应 ignore_index；输入的 target_valid 只管可见关系。还要避免整批所有标签都被忽略。本章测试验证 forward、反向、目标因果性和来源 padding 隔离；没有把它宣称为训练好的翻译模型。

依据：S01、S130。

<a id="TH15"></a>
## TH15｜从 logits 到 loss：下一 token 标签怎样右移？（P0）

给模型 0、1、2，希望它分别预测 1、2、3。不是让同一个位置抄回自己的输入，也不是先 softmax 后再把概率交给接收 logits 的损失函数。

**直答：**输入取原序列去掉最后一个，标签取去掉第一个；模型给每个位置输出词表分数，再按对应标签计算交叉熵。

### 跟着写

```python
def next_token_loss(model, token_ids):
    # [0,1,2,3] -> 输入 [0,1,2]，标签 [1,2,3]；只移位一次。
    if token_ids.size(1) < 2:
        raise ValueError("至少两个 token 才能构造下一 token 标签")
    inputs, labels = token_ids[:, :-1], token_ids[:, 1:]
    logits, _ = model(inputs)
    return F.cross_entropy(logits.reshape(-1, logits.size(-1)), labels.reshape(-1))
```

### 进一步：公式、边界与追问

**练习：**原序列 [4,5,6,7]，写出三对输入位置与监督答案。

**答案：**看到 4 预测 5；看到 4、5 预测 6；看到 4、5、6 预测 7。整段可并行算前向，但因果 mask 限制每行的可见范围。

**追问：为什么 reshape 成 [B×T,V]？**交叉熵的这个调用把每个位置当一条分类样本；标签同步展平。用 M02 的交叉熵/KL 可视化先理解单个分数，回来再理解批量平均。

依据：S130。

<a id="TH16"></a>
## TH16｜真的更新一次参数：训练循环哪几行不能少？（P0）

让模型学习 0→1→…→7→0 的玩具规律。它不需要下载语料或显卡；这能检查“前向、反向、更新”有没有接通，不能用来证明理解自然语言。

**直答：**每步清梯度、前向算损失、反向求梯度、按需裁剪，再让优化器更新参数；要看损失和实际生成，而不只是程序退出成功。

### 跟着写

```python
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
```

### 逐行拆解

1. batch 的每行是一条可预测的循环序列，8 行从不同符号起步。
2. optimizer 持有 model.parameters，知道哪些数需要更新。
3. zero_grad 清上一轮梯度；loss 是本轮平均扣分。
4. backward 根据当前 loss 计算各参数梯度；它本身不会更新权重。
5. step 真正改变参数；下一次前向才会使用改变后的权重。
6. 最后用生成结果检查学到的规律，和单看 loss 做交叉验证。

### 进一步：公式、边界与追问

**练习：**注释 optimizer.step 后再运行，与正常结果比较；另外观察没有清梯度时，梯度为什么会累加。

**答案：**没有 step 就没有参数更新；不清梯度会把旧梯度留到下一步。梯度累积可以是有意设计，但必须同时设计 loss 缩放和更新频率。

**追问：loss 降到很低代表模型好了？**这里只验证一条极小训练规律。泛化还要用未参与训练的数据与独立任务测试。本章实际运行结果放在配套实验文件旁。

依据：S130、S134。

<a id="TH17"></a>
## TH17｜生成与 KV Cache：为什么只喂新 token 还能得到同样答案？（P0）

先读完提示词 0、1、2，模型选出 3。下一步只需处理新来的 3，并读取前面已算好的 K/V；如果仍然喂整段却又拼上缓存，前文会被重复加入。

**直答：**先 prefill 整段提示，再逐 token decode；每层保留自身历史 K/V，同时对齐绝对位置与 mask。

### 跟着写

```python
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
```

### 进一步：公式、边界与追问

**练习：**回看 TH06 的 cache 拼接和 TH13 的 positions：已有 3 个 token，新 token 的位置编号是多少？

**答案：**从 0 编号时是 3；它应能看到全部 4 个 key。测试还用“前缀 3 个、随后一块 2 个、最后 1 个”与完整前向逐项比 logits，专门检查非方形 mask。

**追问：缓存以后注意力不用算了吗？**新 Q 仍要与全部已缓存 K 比较；省去的是旧 token 的重复投影与多层前向。缓存增加内存，不等于消除序列长度成本。

**边界：**生成函数用 no_grad 与 eval，并在结束后恢复原训练模式。本教学缓存不支持跨样本不同长度、滑窗淘汰或服务端共享，也没有为了演示静默截断历史。

依据：S01、S128。

<a id="TH18"></a>
## TH18｜RoPE 手写：旋转到底发生在哪个张量上？（P0）

先只看两个通道形成的小箭头。位置越后，按该频率旋转越多；箭头长度不变，两支箭头做点积时会带入相对位置差。

**直答：**本节按相邻通道配对旋转 Q/K；一般不旋转 V。配对方式、频率、位置偏移必须和模型约定一致。

### 跟着写

```python
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
```

### 进一步：公式、边界与追问

**练习：**把 [1,0] 旋转约 90 度，预期得到什么？缓存增加后，positions 能否从 0 重新开始？

**答案：**约 [0,1]；不能每步重置位置。二维点积的相对角度来自两者旋转角之差。

**追问：把这段加到 TinyLanguageModel 就是 LLaMA 吗？**不是。这里给的是独立部件；替换时还要删除绝对位置加法，在 Q/K 的正确阶段旋转，并处理缓存中 K 已被旋转的事实。归一化、FFN、分组头和权重约定也需逐项对应。

依据：S02。

<a id="TH19"></a>
## TH19｜GQA/MQA 手写：K/V 头怎样分给 Q 头？（P0）

有 4 个提问者、2 份资料。前两人共用第 1 份 K/V，后两人共用第 2 份；每人的 Q 仍然独立，所以注意力权重不必相同。

**直答：**GQA 让多个 Q 头共享一组 K/V 头；MQA 是只有一组 K/V 的特殊情形，常用于降低 KV 的存储和读取量。

### 跟着写

```python
def expand_kv(k, v, query_heads):
    kv_heads = k.size(1)
    if v.size(1) != kv_heads or query_heads % kv_heads:
        raise ValueError("Q 头数必须是相同 K/V 头数的整数倍")
    group = query_heads // kv_heads
    # 为看清语义先显式复制；生产内核可避免物理展开。
    return k.repeat_interleave(group, dim=1), v.repeat_interleave(group, dim=1)
```

### 进一步：公式、边界与追问

**练习：**Q 有 8 头，K/V 有 2 头，每份 K/V 对应几个 Q 头？换成 3 个 K/V 头呢？

**答案：**每份对应 4 头；本规则下 8 不能被 3 整除，应报错。

**追问：repeat_interleave 不是又把内存复制回去了吗？**这是便于观察分组的参考写法；高效内核可按共享关系读取。缓存应存未展开的 K/V。理论缓存元素数为 2×层数×批量×KV头数×长度×每头维度；字节数还要乘数据类型字节数。

依据：S03。

<a id="TH20"></a>
## TH20｜怎样证明自己手写正确，而不是碰巧能运行？（P0）

一个把头与位置顺序弄反的程序，可能仍输出 [B,T,D]。一个偷看未来的模型，训练 loss 甚至会更漂亮。正确性需要专门设计会揭穿这些错误的例子。

**直答：**至少验证形状与顺序、参考数值、梯度、遮罩因果性、缓存等价，以及小样本学习；每类检查针对不同错误。

### 跟着写

```python
# 在 transformer_handwrite.py 同目录运行
from transformer_handwrite import TinyLanguageModel
model = TinyLanguageModel().eval()
a = torch.tensor([[0, 1, 2, 3]])
b = torch.tensor([[0, 1, 2, 7]])
torch.testing.assert_close(model(a)[0][:, :3], model(b)[0][:, :3])
# 只改未来，过去位置的输出应不变。
```

### 进一步：公式、边界与追问

**练习：**临时去掉 causal=True，再运行这一检查；它是否能发现未来泄漏？

**答案：**随机权重下通常会不一致。正式测试固定随机种子，并同时做 PyTorch 对照、有限差分梯度、padding 隔离和缓存分块等价，避免依赖单一例子。

**追问：必须所有浮点位完全相同吗？**通常用合理容差比较；不同精度与内核有浮点误差。误差阈值应跟任务与精度对应，不能用很大的容差掩盖轴顺序错误。

依据：S128、S129。

<a id="TH21"></a>
## TH21｜面试 30 分钟：怎样安排手写与讲解？（P0）

时间有限时，不要先写十几个可选参数。先给一个可以解释和验证的核心版本，再按面试官追问扩展。以下是练习时间分配，不是任何公司的统一题型。

**直答：**先确认范围，用形状驱动实现，再用一个小例子自检；解释关键选择与边界和写出代码同样重要。

### 进一步：公式、边界与追问

| 时间 | 练习产物 | 自检 |
|---|---|---|
| 0—3 分钟 | 写 B/T/D/H 与输入输出 | 是否 batch_first；是否 causal |
| 3—10 分钟 | 单头 attention 与 mask | softmax 沿 key；未来权重为 0 |
| 10—18 分钟 | 多头拆合与输出投影 | 数字顺序可逆；不把 token 轴拼错 |
| 18—24 分钟 | Norm、FFN、残差 | 形状保持；参数被登记 |
| 24—30 分钟 | 样例测试与追问 | padding、all-masked、cache 的约定 |

**练习：**合上答案，仅凭这张表重写 TH06，再用参考测试检查。若题目要求完整 Transformer，则把 TH11—TH14 的 encoder/cross-attention 一起纳入更长一轮练习。

**答案判定：**不是和参考代码逐字相同，而是信息流、计算、mask、参数与结果满足同一约定。

**追问：可以用库层吗？**先确认面试要求。手写基础时不能用封装层替代正在考的模块；工程应用可以选成熟内核，并解释两者差异。

依据：S129、S135。

<a id="TH22"></a>
## TH22｜怎样把可视化、代码练习与 Agent 实战连起来？（P0）

会滑动一个动画，不等于会写算法；会背代码，也不等于知道为什么错。每次用同一小例子走完“先预测 → 操作可视化 → 手写 → 运行检查 → 解释差异”。

**直答：**用可视化建立可计算的直觉，用代码验证它，再把错误诊断方法迁移到真实模型或 Agent 的工具、训练与评测。

### 进一步：公式、边界与追问

**练习路线：**矩阵乘法看一个格子的来源 → TH03 写投影；注意力开关因果遮罩 → TH04—TH06 写 mask；概率滑块看 CE/KL → TH15 写 loss；缓存逐步增加 → TH17 比较完整与缓存 logits。

**答案标准：**每一步能说出输入输出、变化原因及一个失败例子。比如“没 mask 也能生成”不能证明没偷看答案；“工具返回 200”也不能证明 Agent 办成任务。

**追问：外部开源教程怎样搭配？**Happy-LLM 适合沿基础走向完整实现；LLMs-from-scratch 有模块代码和练习；Hello-Agents 让读者从基本循环走向工具、记忆与项目；面试指南可用于连环追问。这里只借鉴组织方式，算法事实仍按原论文与官方接口核对，没有把公开面经当作公司要求的证据。

依据：S132、S133、S134、S135。

## 本章来源

[S01](../90_维护与来源/来源索引.md#S01) · [S02](../90_维护与来源/来源索引.md#S02) · [S03](../90_维护与来源/来源索引.md#S03) · [S56](../90_维护与来源/来源索引.md#S56) · [S57](../90_维护与来源/来源索引.md#S57) · [S128](../90_维护与来源/来源索引.md#S128) · [S129](../90_维护与来源/来源索引.md#S129) · [S130](../90_维护与来源/来源索引.md#S130) · [S131](../90_维护与来源/来源索引.md#S131) · [S132](../90_维护与来源/来源索引.md#S132) · [S133](../90_维护与来源/来源索引.md#S133) · [S134](../90_维护与来源/来源索引.md#S134) · [S135](../90_维护与来源/来源索引.md#S135)
