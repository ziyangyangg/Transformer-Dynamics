# 文献与开放问题地图（核对截止 2026-08-15）

## 0. 证据标准

本文只用作者论文页、会议正式论文页、arXiv 原文和作者官方代码作为结论依据。三份用户
上传材料逐页核对；网络查重截止 **2026-08-15**。“没有找到”仅表示：在本文列出的一手
来源及其参考文献链中，没有找到同时覆盖指定假设的结果；它不是对所有文献的逻辑穷尽。

分类词有固定含义：

- **已解决**：论文在自己明确写出的模型和假设内给出定理；
- **已有近似理论**：有早期展开、线性化、固定嵌入、特殊初始化、容量构造或降维动力学，
  但没有覆盖这里的联合模型；
- **实验现象已知、理论缺失**：现象被重复观察，但没有对应设定的训练选择定理；
- **真正开放**：扣除已有结果后，仍有一个能被定理或反例关闭的具体缺口。

核心审计原则是：

> “Transformer 会聚类”“attention 变得好看”“低损失”“表示低秩”都不自动回答训练
> 为什么选择了因果 routing，也不自动证明存在 downstream compensation。

---

## 1. 三份上传材料到底提出了什么

### 1.1 Hanin, *Neural Networks: a Primer for Mathematicians*

核对版本为 Boris Hanin 2026-03-28 的 62 页讲义；公开原文为
[作者 PDF](https://boris-hanin.github.io/nn-notes.pdf)。

#### Open Problem 5：数据量和宽度联合增长的 DMFT

原问题在 §4.2.2，页 45。给定 $L$ 层全连接 mean-field 网络、宽度
$N_1,\ldots,N_L\asymp N$、分布
$\rho\in\mathcal P(\mathbb R^{N_0+1})$，训练集

\[
\mathcal D=\{(x^\mu,y^\mu)\}_{\mu=1}^{n}\sim\rho^{\otimes n},
\]

用梯度下降最小化均方误差。问题是：当 $n,N\to\infty$ **同时**发生时，建立
mean-field/DMFT 训练理论。讲义注明：深线性网络是已解决的特殊情形。

本项目固定有限 $C,d,L,H,m$，并用 fresh online batch 近似 population gradient flow；
所以它**不是** OP5 的解。它先找出有限 Transformer 中需要被极限理论解释的 order
parameters，再问这些量在 $d,n\to\infty$ 下是否闭合。

#### Open Problem 6：一般 DAG 的 DMFT

原问题同在 §4.2.2，页 46。把前馈网络写成有向无环图 $G=(V,E)$，训练权重放在边上；
在固定训练集、隐藏宽度增长时，把 DMFT 推广到一般 DAG。

带 residual、multi-head attention 和 FFN 的 Transformer 是参数共享且有乘法 softmax
门控的 DAG；但有限网络的精确梯度流分析最多是 OP6 的一个**原型子问题**。除非进一步取
宽度极限并给出闭合 DMFT，不能声称解决 OP6。

#### §5.3 Superposition 的 Q1/Q2

讲义页 49--50 明确问：

1. **Q1**：对哪些数据生成过程和神经网络，能证明 superposition 会发生？
2. **Q2**：superposition 与 compressed sensing 的关系是什么？

$C>d$ 只说明 concept dictionary 不能全正交；低有效秩只说明压缩几何。只有证明一个
activation 同时携带多个可独立解码的特征，才能称 activation superposition。本项目先
直接检验 **learned compressed dictionary**，再用合法 on-support swap 检验它是否真的
产生 functional cross-talk。

因此本项目是 Hanin Q1/Q2 的一个具体实例，而不是一般答案：它问稀疏 episodic retrieval
分布、有限 causal Transformer、联合训练 $E,Q,K,V,O,\mathrm{FFN},w$ 时，训练是否以及
怎样选择压缩码和查询条件解码器。

---

### 1.2 Geshkovski--Letrouit--Polyanskiy--Rigollet,
*A Mathematical Perspective on Transformers*

核对的是 Bull. AMS 2025 版（427--479 页）；公开版本为
[arXiv:2312.10794](https://arxiv.org/abs/2312.10794)，作者的
[官方实验代码](https://github.com/borjanG/2023-transformers-rotf)也已在本项目固定提交
复现。

文章把层深写成连续时间 $t$，研究给定参数 $Q,K,V$ 后 token 粒子的位置动力学。典型
球面模型是

\[
\dot x_i(t)=P_{x_i(t)}\!\left[
  \frac{\sum_j e^{\beta\langle Qx_i,Kx_j\rangle}Vx_j}
       {\sum_k e^{\beta\langle Qx_i,Kx_k\rangle}}
\right],
\qquad P_x=I-xx^\top .
\]

这里的 $t$ 是**表示穿过层的时间**，不是参数训练时间 $s$。这一区分决定了我们的实验
怎样连接原文 clustering 定理。

#### 原文编号逐项核对

| 原文位置 | 精确问题 | 截至 2026-08-15 的状态 | 与本项目的关系 |
|---|---|---|---|
| Problem 2.15，页 453 | Theorem 2.12--2.13 的高维聚类/相变能否推广到随机 $Q,K,V$？ | [GLPR 2023](https://arxiv.org/abs/2305.05465) 按 $V$ 的谱刻画若干固定权重极限；[Karagodin--Polyanskiy--Rigollet 2024](https://arxiv.org/abs/2411.04990) 在 causal mask 下允许任意 $Q^\top K$、但要求 $V=I$；随机三矩阵仍未一般解决 | 我们学到任务相关矩阵，既非固定 iid 随机矩阵，也不研究同一长时极限；只能给“训练后矩阵”新 ensemble |
| Problem 3.1，页 457 | 除全局最大点外，圆周能量 $E_\beta$ 的所有临界点是否都是 strict saddle？ | 文中只覆盖极端 $\beta$ 区间；一般中间区间仍开放 | 与训练参数 loss landscape 不是同一个景观；前者是 token 配置能量 |
| Problem 3.2，页 458 | 为 BBGKY 方程中的三点项 $g(t,x)$ 构造现实 closure，并证明二点密度 $\psi(t,\cdot)\to\delta_0$ | 高维及完整 self-attention closure 仍开放 | 有限 episode 统计不构成 BBGKY closure |
| Problem 3.5，页 460 | 把 Table 1 的纯 self-attention 聚类推广到更多 $Q,K,V$，并刻画极限形状 | GLPR 2023 的 $V$-spectrum 结果和 causal-mask 2024 的任意 $Q^\top K,V=I$ 结果是严格部分答案；任意 learned matrices 仍开放 | 训练选择可产生应由 3.5 分类的新矩阵族，但不等于已证明其长时极限 |
| Problem 3.7，页 461 | $\beta\to\infty$ 的非唯一 Filippov hard-attention 动力学能否用 entropy/viscosity 型原则选唯一解？ | 仍是奇异动力系统问题 | 本项目温度有限、用 exact softmax；不是直接实验 |
| Problem 3.8，页 461--462 | 加强度 $\sigma>0$ 的扩散后，Transformer 长时极限怎样？ | 一般情况开放 | online SGD 噪声不是文中 token-state diffusion；不能混同 |
| §3.4，页 462 | 文章明确说 training dynamics 是全文没有覆盖的 major challenge | 已有很多特殊任务理论；完整 learned $E/QK/OV/FFN$ 仍缺 | 是“复合 routing kernel 训练选择”最直接的出处 |

一个常见编号错误也在这里澄清：**Theorem 3.4** 是 Cohn--Kumar 的 sharp
configuration 结果；“3.4 training gap”指文章的 **Section 3.4**，不是 “Problem 3.4”。

原文附近另一个边界也已经前移：Geshkovski--Koubbi--Polyanskiy--Rigollet 的
[Dynamic Metastability](https://arxiv.org/abs/2410.06833) 在缩放单位球模型中证明多
cluster 状态可停留指数长时间后再塌缩。它解释“有限深度看见多个 cluster、无限深度只有
一个 cluster”可以并存，但仍固定 interaction weights，也不是参数训练定理。

#### clustering 与我们两个问题的精确关系

Perspective 的 clustering order parameter 可以取

\[
\rho_{\mathrm{global}}^{\ell}
=\mathbb E\frac1{T(T-1)}\sum_{i\ne j}
\frac{\langle x_i^\ell,x_j^\ell\rangle}
{\|x_i^\ell\|\,\|x_j^\ell\|}.
\]

retrieval 需要的则是 target-selective geometry

\[
\Delta\rho^\ell
=\mathbb E\left[
\cos(x_q^\ell,x_J^\ell)
-\frac1{m-1}\sum_{i\ne J}\cos(x_q^\ell,x_i^\ell)
\right]
\]

以及端到端 causal value kernel $\kappa_J$。三者不同：

- $\rho_{\rm global}\uparrow$ 描述所有 token 变得相似；
- $\Delta\rho>0$ 描述 query 对 target 比对 distractor 更接近；
- $\kappa_J\approx1$ 才说明输出因果依赖 target value。

全局 clustering 可能在有限深度形成有用的 metastable groups，也可能继续发展为 rank
collapse 并毁掉检索。我们的训练问题问参数时间 $s$ 怎样选择 interaction kernel，使有限
层 $\ell\le L$ 出现 target selectivity，而不是假定“越聚类越好”。

Isobe--Inoue--Imaizumi 2026 的
[Training-Induced Escape from Token Clustering](https://arxiv.org/abs/2605.07772)
首次把训练和 token mean-field 结合，但只训练带 $L^2$ 正则的**参数线性 FFN**，并在
noisy mean-field、非 causal 的连续模型中证明末层逃离 cluster。它缩小了 §3.4 的空白，
却没有训练 $E,Q,K,V,O$，也没有 retrieval causal kernel。

---

### 1.3 Tai--Liu--Li--Chan,
*A Mathematical Explanation of Transformers for LLMs/GPTs*

核对版本为 arXiv v1；公开原文为
[arXiv:2510.03989](https://arxiv.org/abs/2510.03989)。这篇文章**没有命名或编号 open
problem**。它提供的是控制与约束优化定位，不能把它改写成作者没有提出的 OP。

文章令 $u(x,y,t)$ 表示 token 位置 $x$、feature 坐标 $y$、连续层时间 $t$ 上的状态，把
$W^Q,W^K,W^V$ 写成积分核，把 attention 写成非局部算子，把 layer normalization 写成
到均值/方差约束集的投影，把 FFN 写成 operator-splitting 子步。控制变量为

\[
\theta=\{W^Q,W^K,W^V,(W_j,b_j)_{j=1}^J,\sigma_1,\sigma_2\},
\qquad \mathcal N_\theta:f\mapsto u(\cdot,\cdot,T).
\]

给数据 $(u_i,v_i)_{i=1}^B$，其式 (10)--(11) 是

\[
\min_\theta \frac1B\sum_{i=1}^B\ell(\mathcal N_\theta(u_i),v_i),
\qquad
\mathcal N_\theta(u_i)\ \text{必须满足连续控制方程};
\]

离散后的式 (12) 以离散 Transformer propagator 为约束。论文证明/解释的是“架构可由
连续控制方程离散得到”；它没有分析某个优化算法产生的 $\theta_s$ 轨迹，没有得到
population GF 的闭合 order parameter，也没有回答低风险解怎样被选择。

它对本项目的作用是提供一个承载 $E,QK,OV,\mathrm{FFN}$ 联合控制的连续框架。我们的
两个问题是在这个框架上增加**训练选择、因果可识别性与表示几何**，而不是声称原文已经
提出相同定理。

---

## 2. 逐条查重后的文献边界

### 2.1 Transformer 训练动力学与 routing

| 一手论文 | 已经证明/分析了什么 | 没有覆盖什么 | 分类 |
|---|---|---|---|
| Chen et al., [Unveiling Induction Heads](https://arxiv.org/abs/2409.10559), 2024 | 两层、多头 softmax、相对位置、带 normalization 的 FFN；在 $n$-gram Markov 数据上证明 population GF 收敛到 copier--selector--classifier induction circuit | token 表示与阶段设定高度结构化；不是 learned compressed concept dictionary 或 fresh-value episodic retrieval | 特殊任务内已解决；对本项目是近似 |
| Im et al., [How Do Transformers Learn to Associate Tokens](https://arxiv.org/abs/2601.19208), ICLR 2026 | gradient leading-term 给训练早期各权重闭式表达；基函数来自 bigram、token interchangeability、context mapping | 不是 fresh random-value causal retrieval；不是全程 population GF；没有 learned compressed $E$ 与因果干预定理 | 已有近似理论 |
| Yang et al., [Training Dynamics of Transformers to Recognize Word Co-occurrence](https://proceedings.neurips.cc/paper_files/paper/2024/hash/520416e27d3b0cef3cd70a083e2991c7-Abstract-Conference.html), NeurIPS 2024 | 浅层模型从随机初始化同时训练三个 attention matrices 与线性 MLP；证明 MLP 先对齐、attention 后协同的两阶段 GF | 固定词表示；共现分类；不是 causal retrieval、OV/embedding superposition 或多层 FFN 补偿 | 已有近似理论 |
| He et al., [In-Context Linear Regression Demystified](https://proceedings.mlr.press/v267/he25q.html), ICML 2025 | 多头 exact-softmax 从随机初始化形成 QK 对角同质、OV last-entry/zero-sum 结构，并近似 debiased GD predictor | 高斯线性回归与特定简化；无 learned token dictionary、episodic bit value、FFN 或 on-support swap | 特殊模型内已解决；对本项目是近似 |
| Varre--Flammarion, [Incremental Learning in Transformers for In-Context Associative Recall](https://openreview.net/forum?id=rWQ5PqlROI), 2026 | 简化两层 attention-only；对称性守恒律说明初始化形状/尺度决定 head 激活顺序、plateau 和 stagewise circuit | 无 joint learned $E$、FFN、标准 block 或下游补偿；任务不同于单步 key--bit retrieval | 特殊模型内已解决；对本项目是近似 |
| Nichani--Lee--Bietti, [Understanding Factual Recall via Associative Memories](https://arxiv.org/abs/2412.06538), 2024 | attention value memory 与 MLP memory 的容量 trade-off；简化 linear-attention GF 有 sequential learning | 主要是容量/构造和简化 GF；嵌入不是与 QK/OV/FFN 一起形成的 episodic code | 容量已解决；联合选择开放 |
| Vural et al., [Learning to Recall Beyond Orthogonal Embeddings](https://arxiv.org/abs/2603.15923), ICLR 2026 | 固定非正交随机 embedding、有限样本、单层 retrieval；跟踪 empirical GD 早期并给 $N,d,L$ 乘法容量尺度 | embedding 不学习；one-to-one token-label memory，不是每 episode 新 Rademacher value；只跟踪 early phase | 已有近似理论 |
| Adler, [A Capacity-Based Rationale for Multi-Head Attention](https://arxiv.org/abs/2509.22840), 2026 版 | relational graph recognition 中给 QK channel 近匹配上下界；固定总 key dimension 时，多头降低 embedding interference | 核心是容量与显式构造；未证明 joint GF 选择，也无 OV/FFN compensation | 容量已解决；训练选择开放 |
| Isobe et al., [Training-Induced Escape from Token Clustering](https://arxiv.org/abs/2605.07772), 2026 | noisy mean-field 中训练参数线性 FFN，证明 attention 聚类后可在末层逃离 | QK/OV/E 固定；无 causal mask 和 retrieval | 特殊 mean-field 内已解决 |

文献已经说明“某些简化 Transformer 的 QK/OV 结构可以由 GF 学出”，所以宽泛的
“attention 怎样被训练”不是新问题。真正没有被上述论文覆盖的交集是

\[
\boxed{
\text{learned compressed }E
+\text{causal exact-softmax}
+\text{factorized multi-head QK/OV}
+\text{optional FFN/readout}
+\text{fresh episodic values}
+\text{joint population GF}
}.
\]

目标也不是只求低损失，而是区分 function-level causal routing、直接 key path 和参数
factorization selection。

### 2.2 Superposition、下游选择与 compressed sensing

| 一手论文 | 已经解决的部分 | 对本项目留下的缺口 | 分类 |
|---|---|---|---|
| Cowsik--Dolev--Infanger, [The Persian Rug](https://arxiv.org/abs/2410.12101), 2024 | permutation-symmetric 稀疏数据的极简 ReLU autoencoder；大输入维极限的可解析 loss 与完整算法 | 不是 Transformer，无 query-conditioned routing 或 joint attention training | 特殊 toy model 已解决 |
| Adler--Shavit, [On the Complexity of Neural Computation in Superposition](https://arxiv.org/abs/2409.15318), v3 2026 | permutations、pairwise logic 等显式计算的神经元/参数上下界；区分“表示”和“在叠加中计算” | 复杂度/构造理论不说明 GF 会选何种 code 或哪个模块消串扰 | 复杂度边界已解决 |
| Ravfogel et al., [Geometric Factual Recall in Transformers](https://arxiv.org/abs/2605.12426), 2026 | 随机双射事实构造：$O(\log C)$ 维 subject embedding 可线性叠加属性，小 ReLU MLP 作 relation-conditioned selector；实验观察 GD 找到预言结构 | 固定事实记忆而非每 episode 新 value；理论主结果是存在性/容量，GD 选择主要是实验；无联合 $E/QK/OV/FFN$ population-GF localization | **宽泛 downstream selection 已部分解决；本项目须收窄** |
| Nichani et al., 2024（上表） | OV associative memory 与 MLP associative memory 的容量 trade-off | 不等于在同一 learned compressed activation 上识别 cross-talk 再下游消除 | 容量已解决；机制缺失 |
| Adler 2025（上表） | QK capacity 中 embedding interference 与多头收益 | 不证明 learned $E$ 或 downstream causal compensation | 容量已解决；训练缺失 |

所以“learned superposition 的下游补偿理论”若不加限定，已被 Ravfogel et al. 的结果明显
侵占。仍可诚实主张的问题是：

> 在每个 episode 的 values 都重新随机、答案不能存进 embedding 的 causal retrieval 中，
> 联合梯度流会不会先形成有 functional cross-talk 的 learned compressed concept code，
> 再由 QK、OV 或 FFN 中哪一个模块以可识别的 finite intervention 消除该 cross-talk？

这不是 Ravfogel 的“固定 subject embedding 存若干 attribute，再由 MLP 选择”设定。

### 2.3 Plateau 与失败种子：为什么不能先叫新现象

本项目 momentum-SGD 中一个 seed 在延长训练后仍停在中等风险。现有文献给出足够接近的
解释，因此它目前应分类为**已知 plateau/local-solution 家族的实例**：

- Gopalani--Hu,
  [What Happens During the Loss Plateau?](https://proceedings.neurips.cc/paper_files/paper/2025/hash/4f06c73c45d2625f0e687f7e6a206332-Abstract-Conference.html),
  NeurIPS 2025：plateau 中形成 partial solution、repetition bias、representation collapse，
  且 optimal attention maps 学得慢；
- Song et al.,
  [Unraveling the Gradient Descent Dynamics of Transformers](https://proceedings.neurips.cc/paper_files/paper/2024/hash/a7d36e5cb41a1f21c46db25cb1aafab9-Abstract-Conference.html),
  NeurIPS 2024：单层 softmax Transformer 在某些条件下会落入 suboptimal local solution；
- Gopalani--Lubana--Hu,
  [Abrupt Learning in Transformers: Matrix Completion](https://proceedings.neurips.cc/paper_files/paper/2024/hash/630d293833e09e1ecd892a898a20b074-Abstract-Conference.html),
  NeurIPS 2024：BERT matrix-completion 出现长 plateau 后突降，并伴随 attention/hidden-state
  重组；
- Varre--Flammarion 2026 把简化 associative recall 的 plateau 连接到逐个激活
  sub-circuit 的守恒律。

合理顺序是先做学习率、初始化尺度、优化器和训练时长 remedy。只有这些已知机制都无法
解释、并出现可复制的新 order-parameter 转变时，才升级为新理论问题。

---

## 3. 当前实验支持、反对或尚未识别什么

完整数字见 *MECHANISM_RESULTS_DRAFT.md*。这里仅记录与文献边界有关的结论。

### 已建立/强支持

1. **function-level composite routing**：96 个 AdamW 与 96 个 momentum-SGD 主实验中，
   通过功能门槛的模型具有接近 1 的 queried-value flip effect 与 Walsh target
   coefficient；distractor 与高阶 Walsh 能量很小。网络函数依赖正确随机 value，不能
   靠 concept identity 背答案。
2. **训练后是 target-selective 而非只有 global clustering**：target attention mass 与
   target/distractor 几何间隔可和全局 token 相似度分开记录；这些表示量仍不替代 causal
   kernel。
3. **压缩负载改变 learned dictionary geometry**：$C/d$、head 数和 FFN 条件改变
   embedding effective rank；这只能叫 compressed dictionary geometry。

### 被当前实验反对的具体命题

预注册 QK 局部 suppression statistic 为正才表示 route 把 content chord 缩小。两个优化器
的全部聚合 cell 中，终点值及 init-to-final 增量都为负：route 总体**放大**而非抑制
output-relevant chord。opposition rate 略高于 $1/2$ 不能推翻净效应。

数据反对的是：

> “QK route 普遍作为 content cross-talk 的下游补偿器。”

它不反对 QK 对 retrieval 很重要，也不反对某些 layer/head/任务存在 QK compensation。

### 候选但尚未确认

- **OV**：target direction 相对 distractor direction 的 normalized gain 在 8/8 matched
  cells 中同号，6/8 在两优化器达到预注册方向；这是 robust selectivity 候选，不是 finite
  causal cross-talk removal。
- **FFN**：部分 cell 的 tangent skip/branch opposition 与 cancellation 上升，但 SGD 在
  高 load 条件不稳定；v1 还缺 upstream practical floor 和 finite intervention。因此确认数
  仍为 0。

### 仍未识别

“cross-talk 在 QK、OV 或 FFN 的哪一步被消除”尚未确认。尤其相邻确定性节点的完整
donor patch 必然给同一个下游输出；把 pre-OV 与 post-OV patch 的差当作 OV 贡献是数学
错误。下一轮必须使用 exact route/content/interaction hybrids、normalized direction gain
和 finite skip-vs-branch comparator。

---

## 4. 最终研究地图

| 分类 | 可审计结论 | 对本项目的决定 |
|---|---|---|
| 已解决 | 固定/特殊 $Q,K,V$ 的若干 clustering；特殊 Transformer 任务的 GF；固定事实的 superposed embedding + MLP selector 构造；QK/OV/MLP 容量；superposition computation complexity | 不再把这些宽泛口号当新问题 |
| 已有近似理论 | leading-gradient early phase；固定或随机但不学习的 embeddings；linear attention；特殊初始化/对称不变流形；参数线性 FFN mean-field | 复用其变量、守恒律、小初始化展开和 remedy |
| 实验已知、理论缺失 | 标准 block 中的 plateau、attention 重组、rank collapse；GD 经常找到几何选择器；本实验 OV selectivity 与部分 FFN cancellation | 先做 finite intervention、optimizer replication、loss-landscape/NTK/geometry 对照 |
| 真正开放 A | learned $E$ + factorized multi-head QK/OV + readout/FFN 在 fresh-value causal retrieval 的 joint population-GF closure 与解选择 | 立即推进；先完成低风险强迫的功能定理，再研究参数选择 |
| 真正开放 B | 同一设定中 learned compressed dictionary 是否产生 functional cross-talk，以及哪个模块以 finite、on-support、可复制方式补偿 | 保留第二主问题；接受 QK suppression 当前反证；OV/FFN 只写候选 |
| 更远期开放 | 把有限训练 order parameters 推到 Hanin OP5/OP6 的 $n,d\to\infty$ DMFT，再与 Perspective 的深度/粒子极限耦合 | 不是当前论文首个 theorem claim |

---

## 5. 最小可引用的一手来源

- Hanin, [*Neural Networks: a Primer for Mathematicians*](https://boris-hanin.github.io/nn-notes.pdf),
  2026-03-28。
- Geshkovski et al., [*A Mathematical Perspective on Transformers*](https://arxiv.org/abs/2312.10794),
  Bull. AMS 2025；[官方代码](https://github.com/borjanG/2023-transformers-rotf)。
- Tai et al., [*A Mathematical Explanation of Transformers for LLMs/GPTs*](https://arxiv.org/abs/2510.03989),
  2025。
- Geshkovski et al., [fixed-weight clustering](https://arxiv.org/abs/2305.05465), NeurIPS 2023。
- Karagodin et al., [causal-mask clustering](https://arxiv.org/abs/2411.04990), NeurIPS 2024。
- Geshkovski et al., [dynamic metastability](https://arxiv.org/abs/2410.06833), 2024。
- Chen et al., [induction-head population GF](https://arxiv.org/abs/2409.10559), 2024。
- Im et al., [arXiv:2601.19208](https://arxiv.org/abs/2601.19208), ICLR 2026。
- Yang et al., [NeurIPS 2024](https://proceedings.neurips.cc/paper_files/paper/2024/hash/520416e27d3b0cef3cd70a083e2991c7-Abstract-Conference.html)。
- He et al., [ICML 2025](https://proceedings.mlr.press/v267/he25q.html)。
- Varre--Flammarion, [OpenReview 2026](https://openreview.net/forum?id=rWQ5PqlROI)。
- Nichani et al., [arXiv:2412.06538](https://arxiv.org/abs/2412.06538)。
- Adler, [arXiv:2509.22840](https://arxiv.org/abs/2509.22840)。
- Vural et al., [arXiv:2603.15923](https://arxiv.org/abs/2603.15923)。
- Ravfogel et al., [arXiv:2605.12426](https://arxiv.org/abs/2605.12426)。
- Isobe et al., [arXiv:2605.07772](https://arxiv.org/abs/2605.07772)。
- Cowsik et al., [arXiv:2410.12101](https://arxiv.org/abs/2410.12101)。
- Adler--Shavit, [arXiv:2409.15318](https://arxiv.org/abs/2409.15318)。
