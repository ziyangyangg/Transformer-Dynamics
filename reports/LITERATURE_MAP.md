# 文献、数学工具与开放问题地图（核对至 2026-08-24）

## 结论先行

本项目固定研究

\[
(\mathcal D,R,\theta_0)
\xrightarrow{\text{gradient flow }s}
\{B_{\ell h}(s),C_{\ell h}(s)\}
\xrightarrow{\text{softmax}}
\mathcal K_{\ell,s}
\xrightarrow{\text{depth }\ell}
\Phi_{\theta_s}^{L}(X).
\tag{L1}
\]

\(B=Q^\top K\) 选择交互对象，\(C=OV\) 规定传输内容。文献已经分别研究了式 (L1) 的左半边
若干特殊任务，以及右半边的 fixed-kernel 深度动力学；尚未出现把两边在完整、可识别
exact-softmax 任务中闭合的统一定理。

“没有覆盖”只表示下列一手来源及其参考链没有给出目标联合结论，不是对全部文献的逻辑穷尽。

## 1. Perspective 的准确缺口

[Geshkovski et al., A Mathematical Perspective on Transformers](https://arxiv.org/abs/2312.10794)
把 token 当作 interacting particles，主要研究固定 \(Q,K,V\) 时随层深变化的动力学。Bull. AMS
版 §3.4、当前 arXiv v5 §10 都明确把 parameter training 视为尚未覆盖的重大挑战；它不是一个
编号 Open Problem。

[Castin et al., A Unified Perspective on the Dynamics of Deep Transformers](https://arxiv.org/abs/2501.18322)
进一步得到多头、masked 等模型的 Transformer PDE、适定性与 mean-field 极限。其典型对象是

\[
\partial_t\mu_t+\nabla\!\cdot\{\mu_t\Gamma_{B,C}[\mu_t]\}=0,
\tag{L2}
\]

其中 \(t\) 是深度时间，\(B,C\) 是给定的。我们的目标则是二维系统

\[
\dot\theta_s=-\nabla_\theta R(\theta_s),
\qquad
\partial_t\mu_{s,t}
+\nabla\!\cdot\{\mu_{s,t}\Gamma_{\theta_s}[\mu_{s,t}]\}=0.
\tag{L3}
\]

式 (L3) 的难点不是把 \(B,C\) 手动设成正确值，而是证明任务分布为什么使 factorized
\(Q/K,O/V\) 学出该 interaction law，再证明该 law 的有限深度效果。

[Training-Induced Escape from Token Clustering](https://arxiv.org/abs/2605.07772) 已把训练接入
一个 noisy mean-field 模型，但只训练 parameter-linear FFN，prescribed attention 与 \(QK/OV/E\)
不学习。因此它缩小缺口，却没有关闭式 (L1)。

## 2. 已有 Transformer 训练定理：不可重复申明的 prior art

| 一手来源 | 已证明的训练结论 | 相对式 (L1) 的边界 |
|---|---|---|
| [Max-Margin Token Selection, NeurIPS 2023](https://proceedings.neurips.cc/paper_files/paper/2023/hash/970f59b22f4c72aec75174aae63c7459-Abstract-Conference.html) | 简化 softmax attention 的 GD/regularization path 选择 max-margin token | 不是完整 self-attention 的联合 \(E,QK,OV\) 与 depth dynamics |
| [Scan and Snap](https://arxiv.org/abs/2305.16380) | 一层 next-token 模型中 attention 从共现结构形成 token selection | 无 learned matrix-valued kernel 的一般定理 |
| [Co-occurrence via Gradient Flow, NeurIPS 2024](https://proceedings.neurips.cc/paper_files/paper/2024/hash/520416e27d3b0cef3cd70a083e2991c7-Abstract-Conference.html) | 随机初始化后联合训练三组 attention matrices 与线性 MLP；证明两阶段 GF | 单层、特殊共现分类与数据对称性 |
| [Unveiling Induction Heads](https://arxiv.org/abs/2409.10559) | 两层多头 softmax 在 Markov 数据上学出 induction circuit | 专门的 copier-selector-classifier 结构与位置假设 |
| [In-context convergence](https://openreview.net/forum?id=9GLvXGkUE2) | 一层 exact-softmax 在结构化线性 ICL 中的阶段收敛 | 任务、输入几何和预测头高度专门化 |
| [Multi-head softmax ICL](https://openreview.net/forum?id=3TM3fxwTps) | 高斯线性回归中学到 QK/OV 的多头结构 | 不给 learned token dictionary 或一般 interaction graph |
| [Provably Learn CoT, NeurIPS 2025](https://proceedings.neurips.cc/paper_files/paper/2025/hash/b86a195e70f27017c514fa0e5f80595f-Abstract-Conference.html) | LEGO state tracking 中，GD 学出 attention concentration，并证明长度外推/递归自训练 | 已经证明“特定 state tracking 可由训练学会”；我们的增量必须是统一 kernel-to-depth 机制 |
| [Infinite Limits of Multi-head Transformer Dynamics, NeurIPS 2024](https://proceedings.neurips.cc/paper_files/paper/2024/hash/3eff068e195daace49955348de9f8398-Abstract-Conference.html) | 用 DMFT 描述 feature-learning scaling 下的 infinite heads/key-width/depth | 给统计极限，不给有限任务的显式正确 interaction graph 定理 |

因此以下内容都不是新贡献：attention 可训练成 token selector；低秩 attention 存在容量瓶颈；
特殊任务上 QK/OV 能由 GD 对齐；固定 \(QKV\) 会 clustering。

## 3. 可直接借用的数学与物理理论

| 工具 | 一手来源 | 对本项目真正有用的部分 | 不能直接替代什么 |
|---|---|---|---|
| interacting particles / Vlasov transport | [Perspective](https://arxiv.org/abs/2312.10794)、[Transformer PDE](https://arxiv.org/abs/2501.18322) | fixed learned kernel 的适定性、mean-field 与深度误差传播 | 不解释训练为何选择 kernel |
| metastability / singular perturbation | [Dynamic Metastability](https://arxiv.org/abs/2410.06833) | 解释有限深度多 cluster 与最终 collapse 可同时存在 | 仍是 fixed interaction law，不是训练选择 |
| Wasserstein gradient flow / neural mean field | [Mei–Misiakiewicz–Montanari, COLT 2019](https://proceedings.mlr.press/v99/mei19a.html) | 把宽网络 SGD 写成参数分布的 PDE，并控制有限宽度误差 | attention 的乘法 softmax 与共享 factors 需重新闭合 |
| dynamical mean-field theory | [Infinite Limits of Multi-head Transformer Dynamics](https://proceedings.neurips.cc/paper_files/paper/2024/hash/3eff068e195daace49955348de9f8398-Abstract-Conference.html) | 大 head/key-width/depth 下的 feature-learning order parameters | 不能自动识别任务要求的 \(G^*(X)\) |
| matrix factorization 与 balancing | [Du–Hu–Lee 2018](https://arxiv.org/abs/1806.00900)、[Arora et al. 2019](https://arxiv.org/abs/1905.13655) | 研究 \(B=Q^\top K,C=OV\) 的 gauge、Gram preconditioner、隐式偏置 | softmax/data coupling 不是线性 matrix sensing |
| max-margin implicit bias | [Max-Margin Token Selection](https://proceedings.neurips.cc/paper_files/paper/2023/hash/970f59b22f4c72aec75174aae63c7459-Abstract-Conference.html) | score norm 发散时 margin 方向的候选证明技术 | value transport 与多层 residual 仍需单独控制 |
| associative-memory energy | [Hopfield Networks Is All You Need, ICLR 2021](https://openreview.net/forum?id=tL89RnzIiCd) | softmax retrieval 的固定点、能量和存储解释 | 固定 memories 的检索不是任务驱动的 kernel learning |
| low-rank expressivity | [Low-Rank Bottleneck in Multi-head Attention, ICML 2020](https://proceedings.mlr.press/v119/bhojanapalli20a.html) | 给 \(d_h\) 限制表达能力的既有边界 | rank 本身不是创新；需要证明训练在边界内选择什么 |

最可能成功的证明路线不是另造术语，而是组合三种成熟工具：

1. 用数据对称性和 factor balancing 找闭合 order parameters；
2. 用 max-margin/gradient-flow 技术证明 target score 与 value gain 的方向；
3. 用 Vlasov/稳定性方法把 learned-kernel 误差传播到有限深度。

[Hanin, Neural Networks: a Primer for Mathematicians](https://boris-hanin.github.io/nn-notes.pdf)
的 OP5（数据量与宽度联合极限）和 OP6（一般 DAG 的 DMFT）是更远期目标。当前有限
Transformer 定理最多提供它们需要的 order parameters；没有同时取相应极限前，不能声称回答
这两个 open problems。

## 4. 适合的数据：从定理对象到外部验证

| 数据 | 公开来源 | interaction graph 是否已知 | 正确用途 |
|---|---|---:|---|
| MQAR | [Zoology, ICLR 2024](https://proceedings.iclr.cc/paper_files/paper/2024/hash/448fc91f669c15d10364ee01d512cc10-Abstract-Conference.html)、[代码](https://github.com/HazyResearch/zoology) | 是 | 一步 kernel-learning 主定理；与仓库 random-value retrieval 对齐 |
| LEGO state tracking | [Provably Learn CoT](https://proceedings.neurips.cc/paper_files/paper/2025/hash/b86a195e70f27017c514fa0e5f80595f-Abstract-Conference.html) | 是 | 多步 training-to-depth 定理与长度外推 |
| CLRS / CLRS-Text | [CLRS, ICML 2022](https://proceedings.mlr.press/v162/velickovic22a.html)、[官方代码](https://github.com/google-deepmind/clrs) | 是，含 algorithm hints | Gate 2 后选 search/graph 子任务作外部验证 |
| Pythia | [Pythia](https://arxiv.org/abs/2304.01373)、[官方代码](https://github.com/EleutherAI/pythia) | 否 | 测量校准和单轨迹描述，不作总体推断 |
| PolyPythias | [PolyPythias](https://arxiv.org/abs/2503.09543) | 否，但有多 seed checkpoints | 冻结理论指标后的外部多 seed 轨迹验证 |

使用生成任务不是“自己编造结果”。定理必须指定 \(\mathcal D\)；公开生成器使概率律、正确边、
支持集和反例都可审计。自然文本没有唯一 \(G^*(X)\)，不能承担首个因果/结构定理，只能检验
已经冻结的预测是否外推。

## 5. 当前研究地图

| 分类 | 已知事实 | 项目中的位置 |
|---|---|---|
| 已解决 | fixed-kernel clustering/PDE；若干特殊任务的训练选择；低秩表达边界；受限子类的 \(S_{\rm key}\) bridge | 作为引理、边界或基线，不重新包装 |
| 已有近似理论 | Transformer DMFT、neural mean field、matrix-factor balancing、max-margin implicit bias | 提供 proof machinery，但尚未在目标模型中闭合 |
| 实验现象已知但理论缺失 | toy 中 dense 修复而 rank-matched direct 不修复；Pythia routing 非单调且模板异质 | 只作定理假设/反例线索 |
| 真正 open theorem target | 从 \(\mathcal D,\theta_0\) 推出 factorized \(B_s,C_s\) 的 margin/transport，再推出 learned operator 的有限深度误差界 | 唯一主问题 |

具体地，首个合格定理必须同时出现：

\[
R(s)\to0,\qquad
\gamma_s\to+\infty\ \text{或显式下界},\qquad
\mathcal E_{\mathcal K}(s)\to0,
\tag{L4}
\]

以及由这些训练结论推出的

\[
\mathcal E_{\rm depth}(s,L)
\le\sum_{\ell=0}^{L-1}\eta_\ell(s)
\prod_{r=\ell+1}^{L-1}\Lambda_r.
\tag{L5}
\]

若式 (L4) 在标准 exact-softmax 因 signed cancellation、bypass 或 non-identifiability 而失败，
一个完整、可实现的反例同样是重要结果。继续增加 heads、模型大小或自然数据，而没有关闭
式 (L4)--(L5)，不算推进 open problem。
