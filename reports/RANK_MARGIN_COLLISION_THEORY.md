# 从低秩 routing margin 到稀有 superposition collision

**状态：** Phase-II discovery 之后的数学工作笔记；不是预注册协议，也不是已证明的论文主定理。  
**证据冻结点：** 2026-08-20；所有经验数字必须回指
`phase2-residual-factorization-noffn-analysis-v3` 或之后通过严格审计的补充实验。  
**核心边界：** 已有文献证明了低秩 attention 的一般表达瓶颈；本笔记研究的是一个更窄的
训练后对象：**联合学习的 exact-softmax causal retrieval 中，每头低秩怎样限制 routing
margin，并怎样产生少数但很大的 on-manifold swap collision。**

---

## 1. 当前数据究竟指向什么

Phase-II hard cell 固定

\[
(C,d,m,L,H,p,d_h)=(32,8,4,2,4,8,2),
\tag{M1}
\]

联合训练 concept dictionary、factorized $Q/K,O/V$、位置/类型向量和 readout。最终的
float64/P19 discovery 结果是：

1. constant-LR 和 cosine 两条轨迹都还在继续下降，因而没有证据支持“不可消除 plateau”；
2. 同 rank 的 direct-composite control 没有修复 residual；
3. full-rank dense $B_h,C_h$ 将 $L_W$ 和 $I_{swap}$ 分别降低约 $8.96$ 和 $9.29$ bits；
4. $H=1,d_h=8$ 的 factorized 模型本身已接近零 residual；
5. 旧的 $b=2048$ swap 估计极重尾：factorized arm 中 top 1% episodes 对总 $I_{swap}$
   的贡献中位数约为 $96.5\%$，有效样本量中位数只有 $5.1/2048$。

第 5 点正在用独立的 nested Monte Carlo 补充重新测量，因此这里只把它当作**机制线索**。
前四点共同排除了一个简单的“factorization conditioning 很差”故事，却与 per-head
rank/function capacity 相容。它们还没有区分：瓶颈发生在 $B_h$、$C_h$，还是二者联合。

---

## 2. 单头的有效 score matrix

先冻结一层、一个 head，并暂时把位置/value 扰动吸收到实际 normalized states 中。对 concept
$c$ 的 query state 和 memory state 分别记作 $z_c^q,z_c^k\in\mathbb R^d$。定义

\[
q_c=Qz_c^q\in\mathbb R^r,
\qquad
k_c=Kz_c^k\in\mathbb R^r,
\qquad r=d_h,
\tag{M2}
\]

以及未归一化 score

\[
s(c,c')=\frac{\beta}{\sqrt r}q_c^\top k_{c'}.
\tag{M3}
\]

于是完整 concept-to-concept score table 可以写成

\[
S=Z_qQ^\top KZ_k^\top,
\qquad \operatorname{rank}(S)\le r.
\tag{M4}
\]

对 hard cell 的 factorized head，$r=2$；dense-direct head 允许 rank 到 $d=8$；$H=1$
factorized calibration 也有 $r=8$。这正是现有三个关键 control 的共同坐标。

### 定义 1：全字典 routing margin

若 concept $c$ 应读取同名 memory，定义

\[
\gamma(c)=s(c,c)-\max_{c'\ne c}s(c,c').
\tag{M5}
\]

这是比 attention heatmap 更稳定的对象：softmax 只依赖 score differences，而不是 score
的共同平移。注意它是全字典 margin；训练 episode 每次只暴露 $m-1$ 个 distractors，所以
全字典中的坏邻居可以表现为稀有而非持续失败。

---

## 3. 一个已经可以严格证明的 packing 上界

### 命题 1：单头 margin-capacity 上界

设 $r\ge1$、$\beta>0$、$\rho_Q>0$。一组 concepts $A$ 都由同一个 rank-$r$ head
以 margin 至少 $\gamma>0$ 路由，并且

\[
\|q_c\|_2\le \rho_Q,
\qquad
\|k_c\|_2\le \rho_K
\quad(c\in A).
\tag{M6}
\]

令 $M=|A|$。则

\[
M\le
\left(
1+\frac{2\beta\rho_Q\rho_K}{\gamma\sqrt r}
\right)^r,
\tag{M7}
\]

当 $M>1$ 时，等价地，

\[
\gamma\le
\frac{2\beta\rho_Q\rho_K}
{\sqrt r\left(M^{1/r}-1\right)}.
\tag{M8}
\]

#### 证明

任取不同的 $c,c'\in A$。由 $c$ 的 margin 条件，

\[
\frac{\beta}{\sqrt r}q_c^\top(k_c-k_{c'})\ge\gamma.
\tag{M9}
\]

Cauchy--Schwarz 给出

\[
\|k_c-k_{c'}\|_2
\ge \delta:=\frac{\gamma\sqrt r}{\beta\rho_Q}.
\tag{M10}
\]

因此，以每个 $k_c$ 为中心、半径 $\delta/2$ 的 $r$ 维球两两不交；它们全部包含在半径
$\rho_K+\delta/2$ 的球中。比较体积，

\[
M(\delta/2)^r\le(\rho_K+\delta/2)^r.
\tag{M11}
\]

代入 $\delta$ 即得式 (M7)--(M8)。$\square$

这个上界不是最优 spherical-code 常数，但它有三个优点：假设透明、有限维、直接包含
$\beta,r$ 和实际 norm。实验可以逐 checkpoint 计算右侧，而不是只说“rank 很低”。

### 推论 1：concept-level 多头 cover

假设每个 concept 被预先分配给至少一个 head，并且被分配的 head 对它达到 margin
$\gamma$。由抽屉原理，某个 head 至少处理

\[
M_\star=\lceil C/H\rceil
\tag{M12}
\]

个 concepts，所以式 (M8) 至少对 $M=M_\star$ 成立。

这个推论的边界很重要：如果 head assignment 随 episode、distractor 或 value 改变，静态
concept cover 不再成立。那不是反驳，而是一个可测的替代机制：我们应看到明显的
episode-conditioned head specialization，而不是把所有 heads 的 attention 先平均。

---

## 4. margin 与 softmax 选择性的精确关系

固定 $0<\varepsilon<1$。某个 episode 中令 target score 为 $s_\star$，最大
competitor score 为
$s_{(2)}$，并记

\[
\Delta=s_\star-s_{(2)}.
\tag{M13}
\]

若 target 之外共有 $n\ge1$ 个可见 competitors，则

\[
a_\star
=\frac1{1+\sum_{j\ne\star}\exp(s_j-s_\star)}.
\tag{M14}
\]

因此：

\[
a_\star\ge1-\varepsilon
\quad\Longrightarrow\quad
\Delta\ge\log\frac{1-\varepsilon}{\varepsilon},
\tag{M15}
\]

而若每个 competitor 都满足 $s_\star-s_j\ge\Gamma$，则

\[
\Gamma\ge\log\frac{n(1-\varepsilon)}{\varepsilon}
\quad\Longrightarrow\quad
a_\star\ge1-\varepsilon.
\tag{M16}
\]

式 (M8) 与式 (M15) 合起来给出一个具体矛盾条件：在 query/key norms 有界时，一个
rank-$r$ head 不可能对任意多 concepts 同时维持任意大的近确定性 softmax margin。

但我们的训练分布只抽取 $m=4$ 个 memory concepts。对 query concept $c$，定义阈值
$\tau$ 下的坏邻居数

\[
b_c(\tau)=
\#\{c'\ne c:s(c,c)-s(c,c')<\tau\}.
\tag{M17}
\]

当 $m-1$ 个 distractors 从其余 $C-1$ 个 concepts 中无放回均匀抽取时，episode 至少包含
一个坏邻居的精确概率是

\[
p_c^{bad}(\tau)
=1-
\frac{\binom{C-1-b_c(\tau)}{m-1}}
{\binom{C-1}{m-1}}.
\tag{M18}
\]

若 $b_c$ 很小，式 (M18) 近似 $(m-1)b_c/(C-1)$。所以低秩约束完全可能同时产生：

- 大多数 episode 几乎完美；
- 少数 query--distractor pairs 具有小 margin；
- population mean 的 swap error 被极少数 collision triads 主导。

这正是 nested-MC 与 triad-stratified follow-up 要检验的结构。

---

## 5. 两 memory 极简模型中的 exact swap 公式

为把“collision”变成可证而非比喻，考虑一个 target 和一个 distractor 的单头标量 value
path。令

\[
a_\star=\sigma(\Delta),
\qquad
f=g\{a_\star v_\star+(1-a_\star)v_d\}+f_0,
\tag{M19}
\]

其中 $g$ 是 OV/readout 的有效 gain，$f_0$ 不受该 swap 影响。这里先施加一个明确的
**value-blind score-path 条件**：给定 concept triad 后，$g,\Delta_0,\Delta_1$ 固定且
不依赖 $v_\star,v_d$。on-manifold swap 只把 distractor concept 从 $c$ 换成 $c'$；其
value $v_d$、target 和 label 保持不变。若前后 margin 是
$\Delta_0,\Delta_1$，则

\[
f(X')-f(X)
=g\{\sigma(\Delta_1)-\sigma(\Delta_0)\}(v_\star-v_d).
\tag{M20}
\]

在上述 value-blind 条件下，对独立 Rademacher values 取条件期望，

\[
\mathbb E_v[(f(X')-f(X))^2]
=2g^2\{\sigma(\Delta_1)-\sigma(\Delta_0)\}^2.
\tag{M21}
\]

所以在这个最小模型里，triad swap effect 不是相关性指标，而是 score-margin chord 和
下游 value gain 的精确乘积。多层多头模型中的 QK finite chord、OV selective gain 和真实
nonlinear suffix，正是式 (M21) 的可审计推广。

若 normalized QK state 本身含 value embedding，则 $g,\Delta_0,\Delta_1$ 可能依赖
Rademacher values，式 (M21) 不能因式分解。此时正确的一般式是

\[
\mathbb E_v\!\left[
g(v)^2\{\sigma(\Delta_1(v))-\sigma(\Delta_0(v))\}^2
(v_\star-v_d)^2
\right],
\tag{M21a}
\]

实现和证明应直接使用式 (M21a)，不得把
$\mathbb E(v_\star-v_d)^2=2$ 提出期望。

式 (M21) 还说明为什么只回归 embedding cosine 不够：相同 $E$-Gram collision 可以被
$B_h$ 放大或分离，也可以被 $C_h$ 和 suffix 放大或抑制。

---

## 6. 接下来必须计算的 seed-level objects

对每个训练 seed、arm、checkpoint，保存而不是只画图：

1. 每层每头的完整 concept score table $S_{\ell h}$ 及其数值 rank；
2. 每个 concept 的 $\gamma_{\ell h}(c)$、$b_{\ell h,c}(\tau)$ 和式 (M18)；
3. $\rho_Q,\rho_K$ 及 packing slack

   \[
   \mathcal P_{\ell h}(\gamma)
   =\left(1+\frac{2\beta\rho_Q\rho_K}{\gamma\sqrt{d_h}}\right)^{d_h};
   \tag{M22}
   \]

4. episode-conditioned winning head 与 head specialization entropy；
5. ordered triad

   \[
   T_{q,c\to c'}
   =\mathbb E[D\mid q,\text{ old distractor }c,
   \text{ new concept }c'];
   \tag{M23}
   \]

6. $T$ 的 Gini、top-$k$ mass、out-of-seed predictability，以及 E-gram/QK/OV/suffix
   regressors；
7. mixed-capacity arms：dense-$B$ only、dense-$C$ only、dense-both、rank-$B$/rank-$C$，
   全部从相同 step-0 composites 和非 attention 参数出发。

训练 seed 仍是统计单位。episodes、blocks、heads 和 triads 都不能伪装成额外的 $N$。
triad regression 是探索性机制模型；IID population $I_{swap}$ 仍是主 estimand。

---

## 7. 可证的论文主问题与反例目标

### 主定理候选 A：风险--margin--rank 下界

在一层 attention-only 或受控两层模型中，加入可检查的 value-path 条件：OV/readout gain
有上下界、不同 heads 不作任意符号抵消、query self path 不携带 label。目标是证明

\[
R\le\varepsilon
\Longrightarrow
\Pr\{\Delta_{episode}<\tau(\varepsilon)\}
\le g(\varepsilon),
\tag{M24}
\]

并把式 (M7)、(M18) 代入，得到依赖

\[
(C,m,H,d_h,\rho_Q,\rho_K,\beta)
\tag{M25}
\]

的必要容量条件或风险下界。真正的新内容不是“rank 有上限”，而是把 rank packing、随机
episode law、softmax margin 和功能风险连成同一个有限样本定理。

### 主定理候选 B：训练选择哪一种 margin cover

对 population GF 下的 $B_h=Q_h^\top K_h,C_h=O_hV_h$，研究

\[
\dot B_h=-G_{B_h}K_h^\top K_h-Q_h^\top Q_hG_{B_h},
\qquad
\dot C_h=-G_{C_h}V_h^\top V_h-O_hO_h^\top G_{C_h},
\tag{M26}
\]

何时会：

1. 均匀提高所有 concept margins；
2. 形成 head-specialized cover；
3. 牺牲少数 concept pairs，得到低平均风险但重尾 $I_{swap}$；
4. 让 OV/后续层补偿 QK 无法分离的 collision。

一个有力的反例应给出两组完整参数，具有相同 population risk 或相同低阶 order
parameters，却有不同的 triad tail 或 $\dot\gamma(c)$。这会告诉我们闭合动力学缺少哪个
order parameter。

### 必须先排除的普通解释

在把上述问题称为 open problem 之前，至少要完成：更长训练与 scheduler、mixed $B/C$
capacity、fixed/learned low-coherence codebook、三种 head-capacity control、第二 optimizer、
float64 finite localization 和高精度 nested-MC。任何一个标准 control 能解释现象，就应把
它写成已解决机制，而不是制造“神秘 residual”。

---

## 8. 与 clustering 定理的严格关系

Perspective 一类结果研究固定 interaction kernel 后 token 随 depth 聚集。设 normalized
states 满足

\[
\max_{i,j}\|z_i-z_j\|_2\le\eta,
\qquad \|z_t\|_2\le\rho_z,
\qquad \|B_h\|_{op}\le M.
\tag{M27}
\]

则固定 query $t$ 的任意两个 key scores 满足

\[
|s_{ti}-s_{tj}|
\le\frac{\beta\rho_z M}{\sqrt{d_h}}\eta.
\tag{M28}
\]

所以当 $\eta\to0$，且 $M$ 与 query-norm 上界 $\rho_z$ 都一致有界时，所有 attention
ratios 趋于 $1$，attention 趋于在可见 tokens 上均匀。换言之，

\[
\text{global clustering}
\not\Rightarrow
\text{task-selective routing};
\tag{M29}
\]

若 $\rho_z\le R$ 一致有界而模型在 clustering 时仍保持固定正 margin $\gamma$，则
$\|B_h\|_{op}\ge\gamma\sqrt{d_h}/(\beta R\eta)$，因而必须至少按 $1/\eta$ 发散；
否则模型必须保留一个不聚集的任务相关子空间。这给出本项目与 fixed-parameter clustering
理论之间最具体的桥：训练动力学是否制造一个有界但选择性的 interaction kernel，还是以
norm blow-up 对抗 representation consensus。

---

## 9. 与最近文献的边界

- Bhojanapalli et al., *Low-Rank Bottleneck in Multi-head Attention Models*（ICML 2020）
  已证明 head size 带来一般表达瓶颈；因此“attention 低秩”不是本文新发现。
- Nichani, Lee & Bietti, *Understanding Factual Recall in Transformers via Associative
  Memories*（ICLR 2025）给出 attention/MLP associative-memory 容量和简化 linear-attention
  GF；它没有给出本项目联合 learned $E$、exact softmax、随机 episodic values、逐 slot
  causal routing 和 finite collision tail 的训练选择定理。
- Vural et al., *Learning to Recall with Transformers Beyond Orthogonal Embeddings*（ICLR
  2026）分析随机非正交 embeddings、有限样本和早期 GD 的容量缩放；我们的边界必须放在
  完整训练轨迹、factorized QK/OV、multi-head rank cover 与自然 swap tail，而不是重新声称
  “非正交 embedding 很难”。
- Xiong et al., *In-context Superposition*（arXiv 2026）在预训练 LLM 中观察到 overlapping
  working-memory representations 与下游重组/抑制。我们的可增量目标是把这种现象连接到
  一个已知训练 law、精确 on-support counterfactual 和可证 rank-margin 条件。

因此，最准确的论文定位不是“发现低秩”或“发现 superposition”，而是：

> **训练怎样在 per-head rank 约束下选择一个 margin cover；这个 cover 为什么把平均
> retrieval 做得很好，却把残余误差集中到少数可预测的 concept collisions；以及何时
> downstream paths 能、不能补偿这些 collisions。**

这句话只有在 nested-MC、mixed $B/C$ controls、representation/head matrices 和真实
checkpoint 迁移都完成后，才有资格从 theorem target 升级为论文主张。
