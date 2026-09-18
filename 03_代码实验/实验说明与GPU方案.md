# 实验说明与 GPU 补强方案

## 已提供的 CPU 练习

core_algorithms.py 是核心实现，test_core.py 做独立性质和数值校验，run_examples.py 生成“教学运行结果.json”。无需模型 API、数据上传或 GPU。

在该目录打开终端：

~~~powershell
python test_core.py
python run_examples.py
~~~

依赖 Python 3.10+ 与 NumPy；使用本机配套“验证与重建.ps1”也可运行。代码本身不训练大模型。

| 练习 | 核查的关键点 |
|---|---|
| softmax/CE | 大 logits、平移不变、空 mask、有限差分梯度 |
| attention | 未来 V 不影响早期输出，权重和为 1 |
| sequence logprob | 目标已对齐、忽略 −100、有效长度 |
| LoRA | 合并前后线性代数一致，零初始化还原基座 |
| DPO/SimPO | 初始 loss、梯度符号、长度归一与 β/γ |
| KTO | 参考点处效用、好坏梯度方向与饱和；参考点从外部提供 |
| GRPO/clip | 组全相同、正负优势、三类归一化 |
| pass@k | 小样本穷举对照 |
| paired bootstrap | 相同模型差值区间为零 |
| RRF/KV | 排名融合与单位换算 |
| Agent 恢复 | 提交后响应丢失，不重复写入 |

### 建议自己补做的题

为 attention 加 padding；为 sequence_logprob 增加显式 shift 包装；把分布式变长 token 权重错误写成一个失败用例；将退款模拟扩展为最终一致性并限制查询次数。完成后记录运行输出，不仅贴代码。

## GPU 实验一：DPO/KTO/SimPO 比较

**假设：** 同一 SFT 起点下，反馈形式与目标选择会改变安全双侧表现和训练成本。

**数据：** 训练、开发、冻结测试按会话/模板隔离；记录样本与有效 token；偏好标签带理由和置信度；单样本好坏标签单独定义。

**组别：** SFT 原点、同预算继续 SFT、DPO、SimPO；有合适二值反馈再加 KTO。数据量与信息量不等价时分别报告。

**配置：** 模型及 revision、模板、max length、精度、LoRA target/rank、batch/累积、学习率、epoch、β、γ、参考 logprob 方式、训练器版本。超参数只用开发集选择。

**指标：** precision/recall、关键类召回、正常请求误拒、生成帮助性、回答长度、偏好 win rate、通用能力、GPU 时和峰值显存。

**最低验证：** 训练前手查样本；先少量步检查梯度/标签；正式训练后独立测试；可用预算下多 seed，否则明确训练随机性未充分评估。

**否证条件：** 仅长度变长或阈值变化带来表面胜率；冻结集无稳健收益；关键安全类退化；额外成本超过可接受范围。

**产物：** 配置、版本、数据卡、日志、对照表、错误库、复现说明。此实验尚未在本次任务中训练。

## GPU 实验二：最小 RLVR

先选择可执行的算术/结构化任务，奖励测试应覆盖正确、错误、空答案、多个矛盾答案、截断与解析失败。基线至少含 SFT 与相同预算采样。

比较组大小、reward scaling 和一种明确的 loss_type；总生成 token 与训练 token 分开计。记录 reward mean、组全对/全错比例、熵、长度、ratio、clip、KL 与有效组数。

冻结测试使用未见模板与难度切片；报告 pass@1、pass@k、选择器表现和每成功题成本。避免把 oracle 选中正确答案当作系统实际能力。

这是实验设计，尚未运行真实模型训练。无需先追求大模型或大规模分布式，先证明验证器与采样更新链路正确。

## 无需训练的补强三：RAG/Agent 故障矩阵

每个任务登记目标、证据 ID、工具、政策、预期终态；为正常、空检索、错路由、工具超时、响应丢失、规则过期和提示注入分别创建版本。

比较固定工作流、基本 Agent、加入恢复、加入记忆；同总预算多次运行。记录成功率、违规、无效调用、恢复、引用和 p95 时延。

先人工核查 10 条完整轨迹，再扩大任务集，避免自动 judge 在错误协议上批量生产分数。

## 结果记录模板

~~~json
{
  "experiment_id": "由本人创建",
  "status": "planned",
  "model_revision": null,
  "code_revision": null,
  "data_snapshot": null,
  "split_unit": "conversation_or_template",
  "seed": null,
  "hardware": null,
  "training_config": {},
  "evaluation_protocol": {},
  "metrics": {},
  "cost": {},
  "failures": [],
  "conclusion_scope": ""
}
~~~

null 表示尚未填写的实验元数据，不是已有结果。正式报告需区分教学算例、复现结果和个人业务成果。
