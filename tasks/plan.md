# Current plan: learning task-aligned interaction kernels

> 本文件是当前唯一实施计划。历史 toy-to-Pythia 计划保存在 Git 历史与对应 protocol/result
> manifests 中，不再规定下一阶段方向。

## 成功标准

项目必须闭合下面的定理链：

\[
(\mathcal D,R,\theta_0)
\xrightarrow{\mathrm{gradient\ flow}}
\{B_{\ell h}(s),C_{\ell h}(s)\}
\xrightarrow{\mathrm{exact\ softmax}}
\mathcal K_{\ell,s}
\xrightarrow{\mathrm{depth}}
\Phi_{\theta_s}^{L}(X).
\]

最终结果必须说明：

1. 任务分布的什么结构驱动 \(QK/OV\) 参数更新；
2. 梯度流何时形成正确 source margin 与 value transport；
3. learned kernel 为什么在有限深度内实现目标 interaction operator；
4. 哪些条件不可缺少，并给出 bypass/cancellation/不可识别性的反例。

只增加层数、heads、数据集或模型大小不构成完成。

## Phase 0 — 冻结问题与 prior-art 边界

- [x] 把 Perspective 的训练缺口准确定位为当前 v5 的 §10，而不是 §3.4 或编号 Problem。
- [x] 区分 fixed-kernel depth dynamics 与 parameter-training dynamics。
- [x] 记录已解决的特殊情形：max-margin selection、Scan-and-Snap、co-occurrence GF、
  multi-head ICL allocation、LEGO/CoT training。
- [x] 将 retrieval、rank、collision、causal routing 和 module localization 降为从属对象。

**Gate 0：**任何 novelty statement 必须指出它同时超出哪个 training-only 特例和哪个
fixed-kernel dynamics 结果。

## Phase 1 — 最小完整 softmax 训练定理

冻结一个公开、可枚举且具有已知正确 interaction graph \(G^*(X)\) 的生成任务。模型保留：
learned representation、factorized \(Q/K/O/V\)、exact softmax 和训练 readout；先从单层、
单头、无 FFN 的最小标准子类开始。

### 1.1 精确训练方程

- 推导 population gradient flow 的 \(Q,K,O,V,E,w\) 方程。
- 同时跟踪 gauge-invariant \(B=Q^\top K\)、\(C=OV\) 与任务定义的 order parameters。
- 证明所选 order parameters 是否闭合；若不闭合，构造相同低阶状态、不同导数的反例。

### 1.2 Kernel alignment

- 定义正确 source margin \(\gamma_s\)。
- 定义正确 value transport error \(\mathcal E_{\rm transport}(s)\)。
- 证明或反驳

\[
R(\theta_s)\downarrow0,\qquad
\gamma_s\uparrow,\qquad
\mathcal E_{\rm transport}(s)\downarrow0.
\]

- 明确 norm、gain、initialization、separability、no-bypass 和 signed-cancellation 条件。

### 1.3 实验职责

- 枚举 population 或使用可验证的高精度近似，逐步记录理论中的同一变量。
- 至少 10 个独立训练 seeds；seed 是推断单位。
- 实验只判断假设是否合理、有限宽度误差多大、是否存在反例；不替代理论。

**Gate 1：**得到一个 kernel-learning 定理或一个能推翻拟议定理的完整 exact-softmax 反例。

## Phase 2 — Training-to-depth bridge

固定若干训练时间 \(s\)，把学出的 \(B_s,C_s\) 代回 layer/depth dynamics：

- 证明 softmax leakage 如何由训练产生的 margin 控制；
- 证明单层 message error 如何在有限深度内传播；
- 给出到任务目标算子/目标表示集合的误差界；
- 分清任务对齐 transport、局部 clustering 与全局 collapse。

训练时间 \(s\) 和深度时间 \(t\) 不得混写。

**Gate 2：**同一个定理同时出现由 gradient flow 得到的 kernel 条件，以及该条件推出的
depth-dynamics 结论。

## Phase 3 — 必要的架构扩展

按理论障碍一次加入一个组件：

1. multi-head：只研究 head 分工或 cancellation 是否改变 Gate 1/2；
2. multi-layer/residual：只研究 interaction operator 的组合与 identifiability；
3. RMSNorm/FFN：只研究它们是否破坏或恢复已有界；
4. finite width/rank：只在定理中出现明确容量项时研究。

每次扩展必须沿用同一任务、相同指标、配对初始化和独立 seeds。若只是新增现象，不晋级。

## Phase 4 — 公开任务与中模型验证

- 理论任务优先使用公开、版本冻结、具有已知 interaction graph 的 algorithmic/state-tracking
  generator；LEGO 可作为候选外部验证，但不重复其已证明的 CoT/长度外推结论。
- 20M–70M 从头训练必须有多个独立 seeds，检验 Gate 1/2 的量而非只看 accuracy。
- Pythia/OLMo checkpoint 只检验相同结构量是否在预训练轨迹出现；checkpoint 是 repeated
  measure，不是 seed。
- 自然文本通常没有唯一内部 interaction graph，只作外部描述性检验。

**Gate 3：**小模型定理预测在中模型多 seed 上方向一致；否则报告理论适用边界，不扩规模。

## 不再执行

- 围绕单个 \(C=32,d=8\) cell 无限调参；
- 把 low rank、非正交 embedding、fixed-QKV clustering 重新包装为创新；
- 从 attention heatmap、单条 Pythia trajectory 或局部 patch 命名普遍机制；
- 在 Gate 1/2 前继续扩展模型家族、数据集和诊断模块；
- 与主式无关的“有趣现象”进入论文主线。
