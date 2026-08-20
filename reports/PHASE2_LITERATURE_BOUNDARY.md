# Phase II 文献边界：哪些已经解决，哪些仍值得做

**检索冻结日：** 2026-08-20  
**范围：** 官方 conference proceedings、论文主页、作者代码仓库与官方模型卡；结论只针对
下列已检索来源及其引用链，不作“所有文献都不存在”的绝对断言。

## 1. 我们不能再声称什么

下面四个宽泛命题已经有直接理论或实验支持，不能当作本文的新 open problem：

1. **“gradient descent 能让 attention 选择任务相关 token。”**
   [Max-Margin Token Selection](https://proceedings.neurips.cc/paper_files/paper/2023/hash/970f59b22f4c72aec75174aae63c7459-Abstract-Conference.html)
   已在单 attention token-selection 模型中证明渐近 max-margin 选择；
   [Attention with Trained Embeddings](https://proceedings.neurips.cc/paper_files/paper/2025/hash/29be2340edb8b224728248d4bb1ea9d4-Abstract-Conference.html)
   还覆盖了受限的 embedding 更新。
2. **“factorization 会改变训练动力学。”**
   [Training Dynamics of In-Context Learning in Linear Attention](https://proceedings.mlr.press/v267/zhang25br.html)
   已证明 merged $KQ$ 与 separated $K,Q$ 有不同 fixed points 和 saddle-to-saddle
   dynamics；[官方代码](https://github.com/yedizhang/linattn-icl) 可作为 sanity baseline。
3. **“相关或压缩 embedding 会产生相互作用。”**
   [Learning Associative Memories with Gradient Descent](https://proceedings.mlr.press/v235/cabannes24a.html)
   已把 embedding correlation 写入 particle dynamics；
   [From Data Statistics to Feature Geometry](https://proceedings.iclr.cc/paper_files/paper/2026/hash/85cae4ea32697bbdc2a48e7ca0d15988-Abstract-Conference.html)
   进一步说明相关 feature 的 interference 也可能是建设性的。
4. **“下游组件会补偿上游损伤。”**
   [Explorations of Self-Repair in Language Models](https://proceedings.mlr.press/v235/rushing24a.html)
   已在 GPT-2、Pythia、Llama 中展示人工 ablation 后的补偿；
   [作者代码](https://github.com/starship006/backup_research) 可复现 LayerNorm scaling
   与 Anti-Erasure 现象。

因此，本文不能只画 attention heatmap、观察一个 loss plateau，或比较一次 direct 与
factorized training 后就宣称解决 open problem。

## 2. 仍未被上述工作共同覆盖的联合设定

我们的第一个问题严格限定为：

\[
\boxed{
\text{exact-softmax causal retrieval}
+\text{compressed and possibly learned }E
+\text{joint }Q/K,V/O,\text{readout}
+\text{multi-head residual/FFN}
}
\tag{L1}
\]

数据、风险、有限 causal estimand 已在
[`PHASE2_PROTOCOL.md`](PHASE2_PROTOCOL.md) 的式 (P1)--(P11) 冻结。要证明的不是
“target attention 较大”，而是排除等价 shortcut 后的函数级命题，例如

\[
R(\theta)\le\varepsilon
\quad\Longrightarrow\quad
S_{\rm key}(\theta)\ge 1-g(\varepsilon),
\tag{L2}
\]

其中 $S_{\rm key}$ 对每个 episode 的 target 和每个 distractor edge 分别做
$-\infty$ score intervention，再重算全部 descendants。现有训练动力学工作分别解决了
受限子集，但在本次检索范围内没有同时覆盖式 (L1) 的 learned compressed dictionary、
exact softmax、factorized QK/OV、multi-head residual/FFN 与逐 slot counterfactual routing。

最接近的工作包括：

- [Training Dynamics of Transformers to Recognize Word Co-occurrence](https://proceedings.neurips.cc/paper_files/paper/2024/hash/520416e27d3b0cef3cd70a083e2991c7-Abstract-Conference.html)：
  随机初始化、联合 attention/MLP、两阶段 GF；但 embedding 和任务结构更受限。
- [A Phase Transition between Positional and Semantic Learning](https://proceedings.neurips.cc/paper_files/paper/2024/hash/3fefebc2d4e3c1c6ee9b892bd293117d-Abstract-Conference.html)：
  给出 positional/semantic 解的统计物理相变；重点是极小值与样本复杂度，而不是本项目的
  完整有限训练轨迹。
- [From Shortcut to Induction Head](https://proceedings.neurips.cc/paper_files/paper/2025/hash/6499b639e8a4b5c9a780d9b88c09722f-Abstract-Conference.html)：
  证明单层 trigger-copy 中的数据多样性决定 shortcut 或 induction routing。
- [How Transformers Learn Causal Structures In-Context](https://proceedings.iclr.cc/paper_files/paper/2026/hash/934ebba637be11105cff78e86a99eb08-Abstract-Conference.html)：
  两层模型中证明初始梯度可恢复 Markov 父节点；这是 SCM structure recovery，不等于
  本项目的 autoregressive mask 或 counterfactual key-path necessity。

必须区分三种“因果”：

| 名称 | 数学含义 | 本项目用途 |
|---|---|---|
| autoregressive causal | $a_{ti}=0$ for $i>t$ | 架构约束 |
| counterfactual causal routing | 阻断/替换 source 后输出按注册方向变化 | 主 estimand |
| SCM causal discovery | 恢复数据生成图的 parent set | 不作本文主张 |

## 3. 与 Perspective 中 clustering 的准确关系

[A Mathematical Perspective on Transformers](https://arxiv.org/abs/2312.10794)
及其因果 mask 延伸
[Clustering in Causal Attention Masking](https://proceedings.neurips.cc/paper_files/paper/2024/hash/d18d208fa9c333483e5724ade7beff0f-Abstract-Conference.html)
主要研究固定 $Q,K,V$ 后，token 表示随层深/连续深度的相互作用：

\[
x_i^{\ell+1}=\Phi_{Q,K,V}(x_{1:T}^{\ell}).
\tag{L3}
\]

本项目研究训练时间 $s$ 如何改变 interaction kernel：

\[
\dot\theta_s=-\nabla R(\theta_s),
\qquad
B_h(s)=Q_h(s)^\top K_h(s),
\quad
C_h(s)=O_h(s)V_h(s).
\tag{L4}
\]

两者的桥是：训练产生的 $B_h(s),C_h(s)$ 是否进入 Perspective 已分析的 clustering/
consensus 区域。但 clustering 不是 selective routing。Phase-I 的官方球面动力学复现已经
给出反例：所有 token 最终全局共识，同时 attention entropy 达到最大、每个 key 权重
$1/T$；所以

\[
\text{global clustering}\not\Rightarrow
\text{task-selective or counterfactual routing}.
\tag{L5}
\]

Phase-II 会同时报告 global representation alignment、query-target selectivity、Walsh
leakage 与逐 slot $S_{\rm key}$，不再把其中任意一个替代另一个。

## 4. 第二个问题：自然 cross-talk 的训练时补偿

Self-repair 文献通常研究人工损伤 $A$ 后的反应

\[
f_{\theta}(x;\operatorname{do}(A=0))-f_{\theta}(x).
\tag{L6}
\]

我们的对象是训练自己产生的 compressed-dictionary interference。对完全 on-support 的
distractor swap $x\to x'$，逐层构造真实 finite response

\[
p_{M,e}(\Delta)=G_{M,e}(z_{M,e}+\Delta)-G_{M,e}(z_{M,e}),
\tag{L7}
\]

其中 $M\in\{QK,OV,FFN,\text{readout}\}$，$G_{M,e}$ 是该 episode、该 site 之后的
实际 nonlinear suffix。研究问题是：哪个模块在训练中使 cross-talk 的 output-relevant
energy 衰减，以及这种选择能否由 $E$ 的 geometry 和 factorization dynamics 预测。

因果 patching 本身也有已知陷阱：

- [Towards Best Practices of Activation Patching](https://proceedings.iclr.cc/paper_files/paper/2024/hash/06a52a54c8ee03cd86771136bc91eb1f-Abstract-Conference.html)
  说明 corruption 和 metric 会改变定位结果；
- [Interpretability Illusion for Subspace Activation Patching](https://proceedings.iclr.cc/paper_files/paper/2024/hash/70b8505ac79e3e131756f793cd80eb8d-Abstract-Conference.html)
  及
  [Addressing Divergent Representations from Causal Interventions](https://proceedings.iclr.cc/paper_files/paper/2026/hash/133e588e1429f9f1e25b215da145580e-Abstract-Conference.html)
  指出 off-manifold intervention 可激活 dormant pathways。

所以本文只把自然 support 内 swap 作为主干预，并保存 token alignment、finite suffix、
manifold/divergence 诊断。相邻 coherent donor patch 的理论等价仅作 instrumentation check，
不能制造“某模块消除了 perturbation”的假结论。

## 5. 六组控制分别回答什么

| 控制矩阵 | 排除的普通解释 | 能支持的最强结论 |
|---|---|---|
| 3200/6400 + constant/cosine | 训练不足、学习率 schedule | leakage 是否与 base risk 同速、plateau 是否稳定 |
| factorized / dense direct / rank-matched direct | factorization conditioning 与函数 rank 混杂 | dense 只给 capacity upper bound；rank-matched 才是 optimization control |
| random/fixed/learned/low-coherence $E$ | dictionary collision | residual 是否依赖可行的 frame coherence |
| fixed $p$ / fixed $d_h$ / fixed budget | head 数、per-head rank、总容量混杂 | 区分 bottleneck 与 capacity allocation |
| finite QK/OV/FFN/readout suffix | attention heatmap 与 off-manifold patch 假象 | cross-talk 在哪一真实 nonlinear suffix 被放大/抑制 |
| exact population GF-like / SGD / AdamW | mini-batch noise与 adaptive geometry混杂 | order-parameter closure 对哪类动力学成立 |

在 $C=32,d=8$ 中，“orthogonal $E$”在数学上不存在。Welch bound 为

\[
\mu(E)\ge
\sqrt{\frac{32-8}{8(32-1)}}\approx0.3111.
\tag{L8}
\]

因此 hard cell 用近 Welch-bound 的 unit-norm frame；真正 orthogonal control 只在
$C\le d$ 的非压缩 calibration 中运行。

同样，$d=8,H=4,d_h=2$ 时 factorized $B_h,C_h$ 的 rank 至多 2，而 arbitrary direct
$8\times8$ matrix 可达 rank 8。只有 rank-matched direct 或 $H=1,d_h=d$ calibration
能把容量与优化几何分开。

## 6. 为什么先用 Pythia/PolyPythias，再扩到 OLMo/Qwen/Gemma

首个真实模型族固定为
[EleutherAI Pythia](https://huggingface.co/EleutherAI/pythia-70m-deduped)。
[Pythia paper](https://proceedings.mlr.press/v202/biderman23a.html) 与
[官方仓库](https://github.com/EleutherAI/pythia) 提供同一数据顺序、密集训练
checkpoints、多模型大小和 Apache-2.0 权重，因而比只有 final open weights 的模型更适合
研究训练动力学。首轮本地运行 70M，再以 160M/410M 检查 scale transfer。

更关键的是，[PolyPythias](https://arxiv.org/abs/2503.09543) 公开了 5 个大小各 9 条新增
pretraining runs（45 条新轨迹、约 7000 个 checkpoints）。官方模型卡明确给出：原始
非 deduplicated Pythia 使用 seed 1234；`seed1..9` 是另外九条同时改变初始化与 data order
的 runs；160M 另有各三条只改变 data order 或只改变 initialization 的分解控制。
[官方 70M seed 模型卡](https://huggingface.co/EleutherAI/pythia-70m-seed3) 还确认每条 run
都有 154 个 checkpoints 和 Apache-2.0 许可证。

这改变了统计边界：70M 标准模型的 seed 1234 加 seed1--9 可形成 $N=10$ 的真正
pretraining-seed replication；checkpoint 仍只是同一 seed 内的 repeated measure。相反，
deduplicated 70M 没有对应九 seed family，因此只作单轨迹机制描述。两者训练语料处理不同，
不得为了增加 $N$ 而合并。160M 的 data-only/weight-only 每组仅 $N=3$，只作探索性随机性
来源诊断。

注册 revisions：

\[
\{\text{step0},64,512,1000,4000,16000,64000,143000\}.
\tag{L9}
\]

checkpoint 不是独立 seed；因此 checkpoint trajectory 只支持固定训练运行的描述性机制
结论。论文级训练效应使用预先注册的标准 70M 十 seed family，并在 seed-block 内配对
checkpoint 与 prompt；若使用不属于该 family 的模型，则仍需要独立 fine-tuning seeds。

第二外部复核优先
[OLMo 2 1B early training](https://huggingface.co/allenai/OLMo-2-0425-1B-early-training)，
因为模型、训练 recipe 和 checkpoint 流程更开放；但官方说明高频 early-training run
不是原正式 run 的 bitwise trajectory。Qwen3 Base 可测现代 GQA/RMSNorm/RoPE 的迁移，
但没有 Pythia 式密集公开训练轨迹。Gemma 3 + Gemma Scope 适合 feature-level
superposition probe，但本体是 gated/custom license，不能与 Apache 全流程复现混称。

## 7. 当前可发表边界

在完成控制前，只能使用以下分层措辞：

1. **instrument validated：** 解析合同、finite identity、replay 和 provenance 通过；
2. **descriptive phenomenon：** 在固定 seed/checkpoint/prompt population 观察到；
3. **replicated empirical mechanism：** 独立 seeds、第二 optimizer/architecture、校正后方向一致；
4. **empirical open problem candidate：** 协议的十二条升级条件全部通过且事后查重仍无解释；
5. **theorem target：** 明确数据 law、参数化、动力学、order parameters 和反例条件。

当前项目已经处于第 1 层，并拥有 Phase-I 的细粒度描述性/复制证据；它尚未因为一个
$d=8,H=4$ residual 就自动达到第 4 层。Phase-II 六组矩阵的目的正是把“还没调好”与
“现有理论确实缺失”分开。
