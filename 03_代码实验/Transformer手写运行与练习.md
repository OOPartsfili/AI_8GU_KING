# Transformer 手写：运行与练习

先看第 21 章 TH01。你只需要能读 Python 函数、列表与类；不需要显卡、模型账号或下载训练语料。网页可直接阅读和操作可视化，只有运行 Python 实验时才需要下面的环境。

## 第一次运行

在当前学习目录打开 PowerShell。新建独立环境，不改其他项目的依赖。以下版本与本轮实际检查一致：Python 3.12、PyTorch 2.14.0 CPU。

```powershell
python -m venv .venv-transformer
.\.venv-transformer\Scripts\python.exe -m pip install torch==2.14.0 --index-url https://download.pytorch.org/whl/cpu
.\.venv-transformer\Scripts\python.exe -m pip install numpy
.\.venv-transformer\Scripts\python.exe -X utf8 03_代码实验/transformer_handwrite.py
.\.venv-transformer\Scripts\python.exe -X utf8 03_代码实验/test_transformer_handwrite.py
```

首次安装需要联网；后面的实验在 CPU 本地完成。如果系统找不到 python，先安装 Python 3.12 并重新打开终端。已有合适 PyTorch 环境也可直接运行最后两个文件；换版本后应重新跑测试。

本次助手实际验证环境保存在 `work/transformer-visual-revision/venv`，你在当前电脑上也可用它的 `Scripts/python.exe` 运行。运行过程中打印版本、损失和生成序列，不保存大模型权重。

## 先看怎样才算跑通

完整实现学习 0→1→…→7→0 的循环。预期输入 [0,1,2] 后续写出 [3,4,5,6,7,0]，训练损失显著下降。具体实测数值保存在同目录 `Transformer手写运行结果.json`；不同运行环境可能出现小幅数值差异。

这只是检查训练链路能工作的一道小题；没有自然语言数据、验证集或泛化成绩。TinySeq2Seq 做了前向、反向、因果性和 padding 检查，没有训练成翻译模型。

## 五次练习，每次有明确产物

| 次数 | 阅读 | 自己写出 | 验收 |
|---|---|---|---|
| 1 | TH01—TH05 | attention 与两种 mask | 小数值可手算；未来权重为 0 |
| 2 | TH06—TH10 | MultiHeadAttention、Norm、FFN、Block | 与参考层对照；拆合头顺序不变 |
| 3 | TH11—TH14 | TinyLanguageModel 与 TinySeq2Seq | 输出维度和双向/因果可见关系正确 |
| 4 | TH15—TH17 | loss、训练循环、生成与缓存 | 玩具规律学会；缓存 logits 对齐完整前向 |
| 5 | TH18—TH22 | RoPE、GQA 与限时手写 | 旋转不改长度；KV 分组正确；解释边界 |

每次复制完整实现为自己的练习文件，先删去本次要练的函数体，合上参考答案再写。保留一份正确实现，方便比较。不要为了让测试变绿而删掉测试；先读失败信息，再画形状或算一个格子。

## 建议主动制造的五个错误

1. softmax 改成 dim=-2：观察权重行和与参考输出。
2. 去掉 causal mask：修改未来 token，检查过去 logits 会不会变。
3. 去掉合头前 transpose：输出形状可能相同，数字顺序会错。
4. 缓存时把 positions 重置为 0：对比逐 token 与整段前向。
5. 注释 optimizer.step：看 loss 为什么不随训练改善。

上述错误只在自己的练习副本中尝试。参考实现与测试文件作为对照；每次只改一处，定位原因后恢复。

## 代码的明确边界

- attention 的 allowed=True 表示可见；完全屏蔽的行直接报错。padding 的可见性与标签忽略需分开处理。
- 主线 LM 使用无 padding、批内等长输入、可训练绝对位置与 Pre-LN；生成使用贪心选择。
- 缓存是各层的 K/V 张量元组；没有分页、跨请求共享、滑窗淘汰、量化或服务部署逻辑。
- RoPE、RMSNorm、SwiGLU、GQA 是可运行的独立教学部件，并未声称主线 LM 同时采用这些变体。
- Python 与浏览器演示使用同一数学定义和小例子；浏览器不运行 PyTorch、不做训练。不要把动画中的示意数值当作真实模型权重。

参考资料与版本见第 21 章各题的一手来源；开源教程教学组织的取舍见维护目录的“可视化与手写教学设计”。
