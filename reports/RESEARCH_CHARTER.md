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

## 固定任务、变量与取值范围

理论起点是公开可复现的 associative-recall 分布，而不是固定在某个实验 cell。令

\[
N\ge m\ge2,\quad d,L,H,d_h\in\mathbb N,\quad
s\in[0,\infty),\quad \ell\in\{0,\ldots,L-1\}.
\tag{2}
\]

\(N\) 是 concept 总数，\(m\) 是一次样本中的 memory 数，\(d\) 是 residual width，\(H\)
是 head 数，\(d_h\) 是每头 query/key/value width；\(s\) 只表示训练时间，\(\ell\) 只表示
网络深度。实验中的 \(N=32\) 只是一个压力测试取值，不属于问题定义。

从 \([N]\) 中无放回采样互异的 \(c_1,\ldots,c_m\)，再独立采样

\[
v_i\overset{\mathrm{iid}}\sim\operatorname{Unif}\{-1,+1\},
\qquad J\sim\operatorname{Unif}[m],
\qquad q=c_J,\qquad Y=v_J.
\tag{3}
\]

memory token 为 \((c_i,v_i)\)，query token 只给 \(q\)。同一位置在不同样本中可能是 target
也可能是 distractor，且 values 每次重采样，所以模型不能靠位置或 concept identity 记住答案。
这一定义与公开 MQAR 任务兼容；下一阶段的多步版本使用具有公开生成器和已知依赖图的 LEGO
state tracking。

令 learned concept dictionary \(E_s\in\mathbb R^{N\times d}\)。为保持 value 路径可识别，
最小定理模型把 memory/query type 与二值 value 写入预先指定的 carrier；它们与 learned
concept 部分分开。是否也学习 value carrier 是后续扩展，不在首个定理中偷偷加入。

## 模型与 interaction kernel

令 \((X,Y)\sim\mathcal D\)，population risk 与梯度流为

\[
R(\theta)=\frac12\mathbb E_{\mathcal D}(f_\theta(X)-Y)^2,
\qquad
\dot\theta_s=-\nabla_\theta R(\theta_s).
\tag{4}
\]

第 \(\ell\) 层第 \(h\) 个 head 的参数与 composite 为

\[
Q_{\ell h},K_{\ell h},V_{\ell h}\in\mathbb R^{d_h\times d},
\quad O_{\ell h}\in\mathbb R^{d\times d_h},
\quad w\in\mathbb R^d,
\tag{5}
\]

\[
B_{\ell h}(s)=Q_{\ell h}(s)^\top K_{\ell h}(s),
\qquad
C_{\ell h}(s)=O_{\ell h}(s)V_{\ell h}(s).
\tag{6}
\]

因此 \(B_{\ell h},C_{\ell h}\in\mathbb R^{d\times d}\)，且
\(\operatorname{rank}B_{\ell h},\operatorname{rank}C_{\ell h}\le d_h\)。这只是参数化事实，
不是本项目的创新命题。首个定理取 \(L=H=1\)、无 FFN；multi-head/multi-layer 只有在该
定理闭合后才加入。

记 \(\nu(z)\) 为明确给定的 pre-norm map，\(\beta>0\)。causal exact softmax 为

\[
u_{\ell h,ij}(s;X)
=\frac{\beta}{\sqrt{d_h}}\nu(z_i^\ell)^\top B_{\ell h}(s)\nu(z_j^\ell),
\tag{7}
\]

\[
a_{\ell h,ij}(s;X)
=\frac{e^{u_{\ell h,ij}(s;X)}}{\sum_{k\le i}e^{u_{\ell h,ik}(s;X)}}.
\tag{8}
\]

矩阵值 interaction kernel 定义为

\[
\mathcal K_{\ell,s}(i,j;X)
=\sum_h a_{\ell h,ij}(s;X)C_{\ell h}(s),
\qquad
m_i^\ell=\sum_{j\le i}\mathcal K_{\ell,s}(i,j;X)\nu(z_j^\ell).
\tag{9}
\]

\(B\) 决定“从谁读取”，\(C\) 决定“读取的内容如何写入 residual stream”。只研究 attention
mass 或只研究 \(QK\) 都不是完整 interaction kernel。若损失只通过 composite 依赖 factors，
则有精确恒等式

\[
\dot B=-G_BK^\top K-Q^\top QG_B,
\qquad
\dot C=-G_CV^\top V-OO^\top G_C,
\tag{10}
\]

其中 \(G_B=\partial R/\partial B,G_C=\partial R/\partial C\)。式 (10) 是起点，不是闭合
动力学：真正缺失的是把 \(G_B,G_C\) 写成任务统计量和少量 order parameters，并证明其轨迹。

## “任务对齐”的可检验含义

任务必须给出唯一、可审计的交互图 \(G^*(X)\)。在式 (3) 的一步任务中，正确边就是
\(q\to J\)；在 state tracking 中，正确边是当前状态到下一条规则。若 \(G^*\) 不可识别，
仅凭最终标签不能唯一识别内部 kernel。

对 query 可见的 \(M+1\) 个位置（一个 target、\(M\) 个非 target），定义 score margin

\[
\gamma_s(X)=u_{qJ}(s;X)-\max_{j\ne J}u_{qj}(s;X).
\tag{11}
\]

exact softmax 立即给出

\[
1-a_{qJ}\le M e^{-\gamma_s}.
\tag{12}
\]

它只说明“权重集中到谁”，不说明传输了什么。令 \(P_{\rm val}\) 是预先指定的 value-channel
投影，\(r_*\) 是任务要求的单步 value message，定义结构 transport error

\[
\mathcal E_{\mathcal K}(s)=\mathbb E\!\left[
 \|P_{\rm val}\mathcal K_s(q,J)\nu(z_J)-v_Jr_*\|^2
 +\sum_{i\ne J}\|P_{\rm val}\mathcal K_s(q,i)\nu(z_i)\|^2
\right].
\tag{13}
\]

另外，逐 skeleton \(\omega=(c_{1:m},J)\) 定义 Walsh 系数
\(\widehat f_S(\omega)=2^{-m}\sum_v f(\omega,v)\prod_{i\in S}v_i\)。于是

\[
2R_\omega=(\widehat f_{\{J\}}-1)^2+
\sum_{S\ne\{J\}}\widehat f_S^2.
\tag{14}
\]

式 (14) 精确检验输出是否只依赖正确 value，但仍不识别内部 kernel。direct-edge 量为

\[
S_{\rm key}=\mathbb E\!\left[
Y(f-f^{(-J)})-\frac1{m-1}\sum_{i\ne J}Y(f-f^{(-i)})
\right],
\tag{15}
\]

其中 \(f^{(-i)}\) 阻断 query 到 slot \(i\) 的 score edge，并重算全部后代。主定理必须同时
控制式 (11)、(13) 与最终深度误差；accuracy、attention heatmap 或式 (14) 单独都不够。

## 论文的四个定理目标

1. **训练选择。** 在明确的 \(\mathcal D,\theta_0,\beta\) 与 norm 条件下，从式 (10) 推出
   \(R(s)\to0\)、target margin 增长以及 \(\mathcal E_{\mathcal K}(s)\to0\)，并给有限时间界。
2. **factorization。** 证明 \(Q/K,O/V\) 的 Gram preconditioners 如何改变 composite flow；
   区分同函数类的优化效应与放宽 rank 后的容量效应。
3. **training-to-depth。** 若第 \(\ell\) 层 learned operator 的误差为 \(\eta_\ell(s)\)，目标
   层映射的 Lipschitz 常数为 \(\Lambda_\ell\)，证明
   \[
   \mathcal E_{\rm depth}(s,L)
   \le\sum_{\ell=0}^{L-1}\eta_\ell(s)
     \prod_{r=\ell+1}^{L-1}\Lambda_r.
   \tag{16}
   \]
   其中 \(\eta_\ell(s)\) 必须由第 1 个训练定理给出，不能假定 kernel 已正确。
4. **必要条件与反例。** 给出 low risk 何时能识别 \(G^*\)；对 bypass、signed cancellation、
   value-dependent scores 或不可识别任务给 exact-softmax 反例。

首篇论文的最低合格线是闭合 1+3，并把 4 的条件写清。multi-head/rank 只有在定理的误差项
中不可避免地出现时才进入主命题；low-rank attention 本身不是创新。

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

## 数据路线：为什么生成数据不是“编造证据”

定理中的 \(\mathcal D\) 本来就是数学对象。可靠性来自公开生成器、完整概率律、固定版本、
seed/hash 和已知 \(G^*\)，而不是来自“自然文本”四个字。自然文本通常没有唯一内部依赖图，
反而不能验证 kernel 是否正确。

| 层级 | 数据 | 用途 | 进入条件 |
|---|---|---|---|
| A | [MQAR / Zoology](https://proceedings.iclr.cc/paper_files/paper/2024/hash/448fc91f669c15d10364ee01d512cc10-Abstract-Conference.html) 兼容的一步生成器 | 闭合式 (10)--(16)；任意 \(N\ge m\ge2\) | 现在执行 |
| B | [Provably Learn CoT](https://proceedings.neurips.cc/paper_files/paper/2025/hash/b86a195e70f27017c514fa0e5f80595f-Abstract-Conference.html) 使用的 LEGO state tracking | 把一步 kernel 推到多步 interaction graph 与长度外推 | Gate 1 完成 |
| C | [CLRS](https://proceedings.mlr.press/v162/velickovic22a.html) 中依赖图明确的 search/graph 子任务 | 检查定理是否超出专用生成器 | Gate 2 完成 |
| D | [PolyPythias](https://arxiv.org/abs/2503.09543)；Pythia 只作校准 | 多训练 seed 的外部轨迹验证 | 预测和指标先冻结 |

中模型阶段只训练 \(20\)M--\(70\)M exact-softmax 模型，至少 10 个独立 seeds，并沿用相同
\(G^*,\gamma,\mathcal E_{\mathcal K},S_{\rm key}\)。不因为模型变大就更换问题。

## 当前证据与完成距离

| 交付物 | 状态 | 已回答什么 | 仍缺什么 |
|---|---|---|---|
| 问题、变量、测量、可复现管线 | 已完成 | training time 与 depth 分开；\(B,C,\mathcal K\) 可审计 | 无 |
| 可识别性边界 | 部分完成 | 受限单层模型中 \(S_{\rm key}\ge1-\sqrt{2R}\)；有 signed-gain 精确反例 | 多层/RMSNorm/一般 bypass |
| toy 多 seed 实验 | 已完成 discovery | 排除“稳定不可消除 residual”；dense 有效、rank-matched direct 无效，指向 rank/function-class 边界 | 不是训练定理；高精度 swap 尾部仍未全部达门槛 |
| Pythia-70M | 校准完成 | 8 checkpoints 仪器闭合；routing 非单调且模板异质 | 单轨迹；不支持总体训练律或 sparse-collision 故事 |
| kernel-learning GF 定理 | 未完成 | 只有式 (10) 的精确恒等式 | order-parameter closure、margin/transport 收敛 |
| training-to-depth 定理 | 未完成 | fixed-kernel baseline 只验证右半边 | 必须把训练产生的界代入式 (16) |
| 中模型多 seed / 外部数据 | 未开始 | — | MQAR/LEGO 多 seed 与 PolyPythias 复验 |

因此目前不是“实验已经回答主问题”。更准确地说：测量与边界实验约完成，四个定理目标中
只有 identifiability 的受限子问题部分闭合；主定理 1 和 3 尚未闭合。按论文关键交付物估算，
当前约完成 **25%--30%**；最大缺口是证明，不是再跑更多模型。

下一步严格只有两件事：先在 MQAR-compatible population 上闭合或推翻 kernel-learning
定理；随后在 LEGO 多步图上证明式 (16)。这两步之前不启动新的模型族或自然文本数据集。

## 明确停止的方向

- 不再把 retrieval、causal routing、rank、rare collision 或 compensator 写成顶层问题。
- 不把 fixed-\(QKV\) clustering、low-rank attention、非正交 embedding 当作创新。
- 不从 accuracy 或 attention heatmap 推断 kernel 已任务对齐。
- 不在 A/B 尚未闭合时无边界扩大数据、模型或机制诊断。
- 新实验必须检验式 (10)--(16) 的变量、假设或反例；否则不进入主线。
