# 研究宪章：Training-Aware Transformer Dynamics

## 唯一研究问题

> 能否从任务分布和梯度流出发，推导完整 softmax Transformer 学出的
> \(QK/OV\) interaction kernel，并证明该 kernel 的逐层动力学实现任务要求的交互结构？

本问题固定不变。retrieval、clustering、rank、多头、superposition、collision 和 module
patch 都只能作为例子、条件或诊断，不能替代这个问题。

## 文献中的准确缺口

*A Mathematical Perspective on Transformers* 主要研究固定参数下，token 随网络深度演化的
interacting-particle dynamics。当前 arXiv v5 中，训练缺口位于 **§10**，不是 §3.4；原文把
Transformer training dynamics 称为尚未覆盖的重大挑战。它也不是该文的某个编号 Problem。

已有论文已经在单层 next-token prediction、co-occurrence、max-margin token selection、
multi-task ICL 和 LEGO/CoT 等特殊模型中证明了任务相关 attention 可以由梯度下降产生。因此
我们不声称“attention 能学会选择信息”本身是新发现。

| 原始文献 | 已经回答 | 原文留下的边界 |
|---|---|---|
| [A Mathematical Perspective on Transformers](https://arxiv.org/abs/2312.10794) | 固定 interaction law 的 token/measure depth dynamics 与 clustering | §10 明确未覆盖 parameter training |
| [Training-Induced Escape from Token Clustering](https://arxiv.org/abs/2605.07772) | 训练 parameter-linear FFN 可改变 prescribed-attention clustering | 明确把超越 noisy S-USA、FFN-only training 列为 future work |
| [Max-Margin Token Selection](https://proceedings.neurips.cc/paper_files/paper/2023/file/970f59b22f4c72aec75174aae63c7459-Paper-Conference.pdf) | 简化 attention 的 max-margin implicit bias | 明确留下 joint-GD local convergence、完整 self-attention、多个 tunable tokens、非可分与 transient dynamics |
| [Co-occurrence via Gradient Flow](https://arxiv.org/abs/2410.09605) | 单层、特殊正交数据上联合训练 Q/K/V 与线性 MLP | 明确留下 multi-head、multi-layer、相关语言序列、next-token prediction 与更实际优化器 |
| [Scan and Snap](https://arxiv.org/abs/2305.16380)、[Multi-head ICL dynamics](https://arxiv.org/abs/2402.19442)、[Provably Learn CoT](https://arxiv.org/abs/2511.07378) | 分别证明特殊任务中的 token selection、head allocation、attention concentration/长度外推 | 这些结果划定 prior art；我们不能把它们已证明的现象重新命名为贡献 |

仍未闭合的交叉地带是：

\[
\boxed{
(\mathcal D,R,\theta_0)
\xrightarrow{\text{training time }s}
\mathcal K_{\theta_s}
\xrightarrow{\text{depth }\ell}
\Phi_{\theta_s}^{\ell}(X)
}
\tag{1}
\]

即同时刻画训练如何选择 interaction kernel，以及这个 learned kernel 产生什么逐层动力学。

这不是把上述论文的 future-work 列表逐项补齐。真正的贡献必须给出一个统一对象和统一定理：
此前的 training papers 主要停在“特定任务下参数如何变化”，Perspective 主要从“给定 kernel
后表示如何变化”出发；我们要闭合两者之间尚未建立的映射。

## 数学对象

令 \((X,Y)\sim\mathcal D\)，population risk 与梯度流为

\[
R(\theta)=\mathbb E_{\mathcal D}\,\ell(f_\theta(X),Y),
\qquad
\dot\theta_s=-\nabla_\theta R(\theta_s).
\tag{2}
\]

第 \(\ell\) 层第 \(h\) 个 head 的两个 gauge-invariant composite 是

\[
B_{\ell h}(s)=Q_{\ell h}(s)^\top K_{\ell h}(s),
\qquad
C_{\ell h}(s)=O_{\ell h}(s)V_{\ell h}(s).
\tag{3}
\]

对 causal exact-softmax attention，

\[
a_{\ell h,ij}(s;X)
=
\frac{\exp\{(z_i^\ell)^\top B_{\ell h}(s)z_j^\ell/\sqrt{d_h}\}}
{\sum_{k\le i}\exp\{(z_i^\ell)^\top B_{\ell h}(s)z_k^\ell/\sqrt{d_h}\}},
\tag{4}
\]

矩阵值 interaction kernel 定义为

\[
\mathcal K_{\ell,s}(i,j;X)
=\sum_h a_{\ell h,ij}(s;X)C_{\ell h}(s),
\qquad
m_i^\ell=\sum_{j\le i}\mathcal K_{\ell,s}(i,j;X)z_j^\ell.
\tag{5}
\]

\(B\) 决定“从谁读取”，\(C\) 决定“读取的内容如何写入 residual stream”。只研究 attention
mass 或只研究 \(QK\) 都不是完整 interaction kernel。

## “任务对齐”的可检验含义

定理任务必须给出可审计的正确交互结构 \(G^*(X)\)，例如某个 query 应读取的 source，或
state-tracking 中当前步骤必须使用的规则。若任务没有可识别的 \(G^*(X)\)，仅凭最终标签通常
不能唯一识别内部 kernel。

对于唯一正确 source \(J^*(i,X)\) 的最小情形，至少需要同时控制：

\[
\gamma_s(X)
=u_{iJ^*}(s)-\max_{j\ne J^*}u_{ij}(s),
\tag{6}
\]

\[
\left\|\mathcal K_s(i,J^*;X)z_{J^*}-m_i^*(X)\right\|,
\qquad
\sum_{j\ne J^*}\left\|\mathcal K_s(i,j;X)z_j\right\|.
\tag{7}
\]

式 (6) 是“选对谁”，式 (7) 是“传对什么”。主定理不能只证明训练误差趋零；它必须说明
梯度流为何使这些结构量改善，并由此推出有限深度输出或表示误差界。

## 论文的定理包

1. **Kernel learning：**在明确的数据、初始化和尺度条件下，证明梯度流使
   \(B_{\ell h}(s),C_{\ell h}(s)\) 形成任务要求的 margin 与 value transport。
2. **Depth dynamics：**把 learned kernel 代回式 (5)，证明逐层动力学逼近任务要求的
   message-passing/interaction operator，而不是只证明一般 global clustering。
3. **Identifiability：**给出使“低风险 \(\Rightarrow\) 正确 interaction”成立的必要条件；
   对 bypass、signed cancellation 或不可识别任务构造精确反例。
4. **Finite-width boundary：**只有在前述定理需要时，研究 head dimension、rank 或多头如何
   限制 margin；low-rank attention 本身不是创新。

第一个定理应从可识别、可枚举的小任务和 population gradient flow 开始，不一开始声称覆盖
任意自然语言任务或完整大模型训练。

## 突出贡献的判据

一个结果只有同时满足以下条件，才进入论文主贡献：

1. 推导的是标准 exact-softmax 的 \(QK/OV\) interaction kernel，而不是把 kernel 当作自由参数；
2. 结论同时包含 training time 与 depth/layer dynamics，而不是只给收敛率或 attention heatmap；
3. 解释任务分布的哪个结构决定 learned kernel，并给出可失败的条件或反例；
4. 至少把一个已有特殊模型结果和一个 fixed-kernel dynamics 结果放进同一框架；
5. 实验检验定理中的量和边界，而不是通过增加模型、指标或数据制造新故事。

“多一层、多一头、换一个数据集”本身不是贡献。价值来自解释训练为什么选择某类 interaction
law，以及该 law 为什么产生任务需要的逐层计算。

## 已有工作的正确位置

| 已有内容 | 在主问题中的作用 | 不能声称 |
|---|---|---|
| fixed-kernel clustering baseline | 验证 Perspective 的右侧 depth dynamics | global clustering 等于任务对齐 |
| random-value retrieval toy | 提供已知 \(J^*(X)\) 的最小 kernel-learning 实例 | 代表一般语言任务 |
| Walsh、swap、slot blocking | 检查输出是否使用正确输入以及是否存在 cross-talk | 单独识别 QK/OV/FFN 因果模块 |
| dense/rank-matched、head controls | 检查 kernel 参数化的候选限制 | 把 rank/collision 升为主问题 |
| Pythia-70M checkpoints | 验证测量可在真实模型和训练轨迹上运行 | 从单条轨迹推出训练定律 |

当前最实质的数学进展是：在单层、value-linear、score value-blind、无 bypass、有效 gain 非负
的可识别子类中，低风险强迫正确 source 的 blocking effect；允许 signed head cancellation 后，
存在 \(R=0\) 但正确与错误 source 无法由 blocking effect 区分的 exact-softmax 反例。这说明
identifiability 条件不可省略，但还没有证明一般梯度流学出正确 kernel。

## 当前计划

### A. 闭合最小训练定理

冻结一个具有已知 \(G^*(X)\) 的公开生成任务；使用 learned representation、完整 \(Q/K/O/V\)
factorization 和 exact softmax。推导 \(B_s,C_s\) 的 population gradient flow，证明或反驳

\[
R(\theta_s)\downarrow0,
\qquad
\gamma_s\uparrow,
\qquad
\mathcal E_{\rm transport}(s)\downarrow0.
\tag{8}
\]

实验只用于检查证明假设、发现反例和验证有限宽度误差，不替代理论结论。

### B. 接上 Perspective 的逐层动力学

对 A 中学出的 kernel，分析冻结训练时间 \(s\) 后的 layer/depth dynamics；明确哪些 fixed-kernel
clustering/transport 结论适用，以及任务结构在有限深度内是否形成。训练时间 \(s\) 与深度
时间 \(t\) 始终分开。

### C. 受控扩展

只有 A/B 闭合后，才依次加入 multi-head、multi-layer、RMSNorm/FFN 和中模型。每次只增加
一个理论障碍，并保留相同的 \(G^*(X)\)、kernel 指标和独立训练 seeds。公开的 state-tracking
或 algorithmic 数据可作外部验证；自然文本因内部交互路径通常不唯一，只作描述性检验。

### D. 开源模型边界

Pythia/OLMo 等 checkpoint trajectory 用于检验已证明的结构量是否出现。checkpoint、template、
layer 和 head 都是同一训练轨迹的 repeated measures，不是独立样本；没有多训练 seed 时不作
总体训练规律推断。

## 明确停止的方向

- 不再把 retrieval、causal routing、rank、rare collision 或 compensator 写成顶层问题。
- 不把 fixed-\(QKV\) clustering、low-rank attention、非正交 embedding 当作创新。
- 不从 accuracy 或 attention heatmap 推断 kernel 已任务对齐。
- 不在 A/B 尚未闭合时无边界扩大数据、模型或机制诊断。
- 新实验必须回答式 (8) 或验证其必要假设；否则不进入主线。
