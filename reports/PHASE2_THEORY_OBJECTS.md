# Phase II：从实验量到可证明命题

**状态：** 这是 production 结果解盲前冻结的理论对象说明。它不把实验现象预先写成
定理，而是说明每个实验量能支持什么、不能支持什么，以及真正需要证明或构造反例的地方。

## 1. 一句话研究对象

我们研究一个有限 causal Transformer 怎样从随机初始化学会下面的函数：给定若干张
“概念—符号”卡片，再给出其中一个概念作为 query，输出它对应的符号。

它看起来像查字典，但同时包含三个困难：

1. 概念数 $C$ 可以大于表示维度 $d$，所以概念必须共享方向（superposition）；
2. 注意力只允许 query 读取过去的 memory slots（causality）；
3. $E,Q,K,V,O,$ FFN 和 readout 可以共同改变，因此相同的低 loss 可能由不同内部机制实现。

我们的理论问题不是“网络能否拟合”，而是：

> 在什么结构条件下，低 population risk 必然意味着任务相关的 causal routing；当这些
> 条件不成立时，网络怎样用下游模块补偿上游 superposition cross-talk？

---

## 2. 完整概率空间

固定概念全集 $[C]=\{1,\ldots,C\}$ 和 memory 长度 $m$。一次 episode 由

\[
(c_1,\ldots,c_m),\quad J,\quad(v_1,\ldots,v_m)
\tag{T1}
\]

组成，其中：

- $(c_1,\ldots,c_m)$ 是从 $[C]$ 中无放回、按顺序均匀抽取的概念；
- $J\sim\mathrm{Unif}\{1,\ldots,m\}$；
- $v_i\overset{iid}{\sim}\mathrm{Unif}\{-1,+1\}$；
- query 是 $q=c_J$，label 是 $y=v_J$。

因此完整有限 population 的大小是

\[
|\Omega|=\frac{C!}{(C-m)!}\,m\,2^m.
\tag{T2}
\]

例如 $(C,m)=(4,2)$ 时只有 $96$ 个 episode；$(6,3)$ 时有 $2880$ 个。对这些小问题，
我们可以真的把所有数据列出来，而不把 mini-batch 噪声误认为训练动力学。

population risk 统一定义为

\[
R(\theta)=\frac12\,\mathbb E_{X\sim\Omega}
\left[f_\theta(X)-y(X)\right]^2.
\tag{T3}
\]

前面的 $1/2$ 使梯度公式更干净；报告中的 MSE 恒等于 $2R$。

---

## 3. 实现中的 causal Transformer

概念字典为

\[
E=(e_1^\top,\ldots,e_C^\top)^\top\in\mathbb R^{C\times d}.
\tag{T4}
\]

memory token 和 query token 的初态是

\[
x_i^0=e_{c_i}+v_i u+t_{mem}+p_i,\qquad
x_{m+1}^0=e_{c_J}+t_{query}+p_{m+1},
\tag{T5}
\]

其中 $u,t_{mem},t_{query},p_i\in\mathbb R^d$ 都可训练。第 $\ell$ 层先做 RMSNorm，
记 $z_i^\ell=N(x_i^\ell)$。第 $h$ 个 head 的 composite maps 是

\[
B_{\ell h}=Q_{\ell h}^\top K_{\ell h}\in\mathbb R^{d\times d},
\qquad
C_{\ell h}=O_{\ell h}V_{\ell h}\in\mathbb R^{d\times d}.
\tag{T6}
\]

令 $d_h=p/H$。从 query position $t$ 到过去 position $i\le t$ 的 causal attention 为

\[
a_{\ell h,t i}
=\frac{\exp\{\beta z_t^{\ell\top}B_{\ell h}z_i^\ell/\sqrt{d_h}\}}
{\sum_{r\le t}\exp\{\beta z_t^{\ell\top}B_{\ell h}z_r^\ell/\sqrt{d_h}\}},
\quad a_{\ell h,t i}=0\ \text{if }i>t.
\tag{T7}
\]

这里“因果”是一个矩阵约束，不是口头上的“有影响”：任何 token 都不能直接读取未来
token。query 位于最后，所以它能读取全部 memory；memory 之间仍服从下三角 mask。

attention 和 FFN residual update 为

\[
\widetilde x_t^\ell=x_t^\ell+
\frac1{\sqrt L}\sum_{h=1}^H C_{\ell h}
\sum_{i\le t}a_{\ell h,t i}z_i^\ell,
\tag{T8}
\]

\[
x_t^{\ell+1}=\widetilde x_t^\ell+
\frac1{\sqrt L}W_{2\ell}\operatorname{GELU}
\left(W_{1\ell}N(\widetilde x_t^\ell)\right).
\tag{T9}
\]

若该 arm 没有 FFN，式 (T9) 的第二项为零。最后

\[
f_\theta(X)=w^\top N(x_{m+1}^{L})+b.
\tag{T10}
\]

---

## 4. “低风险强迫 routing”最小定理

先看一个学生可以直接展开平方验证的模型：假设网络输出对随机 values 是线性的，且系数
不依赖 values，

\[
f(X)=\sum_{i=1}^m w_i(c_{1:m},J)v_i.
\tag{T11}
\]

因为不同 $v_i$ 是独立、均值为零的 Rademacher 随机变量，交叉项期望全为零，于是

\[
2R=
\mathbb E_{c,J}\left[
(w_J-1)^2+\sum_{i\ne J}w_i^2
\right].
\tag{T12}
\]

这给出第一个可证明命题。

### 命题 A（功能 routing forcing）

若式 (T11) 成立且 $R\le\varepsilon$，则

\[
\mathbb E(w_J-1)^2\le2\varepsilon,
\qquad
\mathbb E\sum_{i\ne J}w_i^2\le2\varepsilon.
\tag{T13}
\]

直观上：想在所有随机正负号上都答对，不能靠猜测 distractors 的平均值；target 的系数必须
接近 $1$，所有 distractor 系数必须接近 $0$。

但 $w_i$ 是整个网络从 value $v_i$ 到输出的**复合功能系数**。式 (T13) 单独不能推出
某个 attention head 的 $a_{Ji}$。OV、后续层、FFN 和 readout 都可能放大、旋转或抵消。

### 定理目标 A1（attention routing identifiability）

在一层 attention-only 模型中，寻找最弱的可检查条件，例如：

1. value direction 与 concept subspace 正交或可分离；
2. $C_h$ 和 readout 在 value subspace 上有有界 condition number；
3. 不同 head 的 value paths 不发生任意符号抵消；
4. query self path 不携带 label。

在这些条件下证明

\[
R\le\varepsilon
\quad\Longrightarrow\quad
\mathbb E[1-a_{target}]\le K\sqrt\varepsilon,
\quad
\mathbb E\sum_{i\ne J}a_i^2\le K'\varepsilon,
\tag{T14}
\]

其中 $K,K'$ 显式依赖上述 condition numbers。实验中的 dense/rank-matched、fixed-$E$、
head-width controls 正是在检验哪些条件不可缺。

### 反例目标 A2

若删去哪一条条件，可以构造 $R=0$ 但 attention 不选择 target 的参数？一个有效反例必须
给出所有参数矩阵，而不是只说“下游可以补偿”。这会精确划出“低风险强迫 causal
routing”定理的边界。

---

## 5. 非线性网络：Walsh 展开是正确的功能坐标

一般 Transformer 的 attention 本身也可能依赖 $v$，所以式 (T11) 未必成立。固定
concept skeleton 和 target 后，$f(v_1,\ldots,v_m)$ 是超立方体上的任意函数。它有唯一
Walsh 展开

\[
f(v)=\sum_{S\subseteq[m]}\widehat f_S\chi_S(v),
\qquad
\chi_S(v)=\prod_{i\in S}v_i.
\tag{T15}
\]

Parseval 恒等式给出

\[
2R=
(\widehat f_{\{J\}}-1)^2
+\sum_{i\ne J}\widehat f_{\{i\}}^2
+\sum_{|S|\ge2}\widehat f_S^2
+\widehat f_\varnothing^2
\tag{T16}
\]

我们把四项依次记为

\[
E_T,\quad L_D,\quad L_H,\quad L_0,
\qquad L_W=L_D+L_H+L_0.
\tag{T17}
\]

其中 $L_H$ 按注册定义包含**所有**二阶及以上项；是否包含 target 只能作为额外的探索性
细分，不能改变 headline partition。

因此 $2R=E_T+L_W$ 是逐 skeleton 的精确恒等式，而不是拟合关系。

### 命题 B（一般功能 routing forcing）

对任意 Transformer，$R\le\varepsilon$ 必然推出

\[
E_T\le2\varepsilon,
\qquad L_W\le2\varepsilon.
\tag{T18}
\]

这回答“网络是否在函数上学会 target-value routing”，但仍不定位是哪一层、哪个 head、
哪个模块完成的。因此每次使用“routing”一词都必须加限定：functional、attention-mass、
direct-edge causal，或 total causal path。

再令 $v^{\oplus J}$ 只翻转 queried value，定义

\[
\Xi_{value}
=\frac12\mathbb E\left[v_J\{f(c,v,J)-f(c,v^{\oplus J},J)\}\right].
\tag{T18a}
\]

在完整 Boolean cube 上，Walsh 正交性逐 skeleton 给出

\[
\Xi_{value}=\mathbb E\kappa_J=K_{target},
\qquad
|1-\Xi_{value}|\le\sqrt{E_T}\le\sqrt{2R}.
\tag{T18b}
\]

第一等式在代码中由 intervention 与 Walsh 两条独立路径复算；第二个不等式来自 Jensen/
Cauchy--Schwarz。这说明低风险强迫 **functional value routing**，但完全没有推出下一节的
direct-edge $S_{key}$：下游间接路径仍可实现同一个函数。

---

## 6. 什么叫 direct-edge causal routing

对同一个 episode，令 $f(X)$ 为正常输出；$f(X;A_{q\leftarrow i}=0)$ 表示在每一层、
每个 head 中，屏蔽 final query 到 memory slot $i$ 的直接 attention edge，再重新归一化并
完整运行后代节点。定义 label-aligned effect

\[
\delta_i(X)=y\left[f(X)-f(X;A_{q\leftarrow i}=0)\right].
\tag{T19}
\]

注册的 slot selectivity 是

\[
S_{key}=\mathbb E\left[
\delta_J-\frac1{m-1}\sum_{i\ne J}\delta_i
\right].
\tag{T20}
\]

这才是我们所说的“任务相关 direct causal routing”。仅仅看到 target attention mass 更大，
或只屏蔽 target edge，均不等于式 (T20)。注意它仍只测 direct query-to-memory edges；经由
中间 token 的间接路径属于 total-effect 问题。

### 定理目标 B1（functional-to-edge bridge）

在显式结构假设下，证明一个形如

\[
L_W\le\varepsilon,\quad R\le\varepsilon,
\quad\mathcal I(\theta)\le\kappa
\Longrightarrow S_{key}\ge s_0-c(\kappa)\sqrt\varepsilon
\tag{T21}
\]

的结论；$\mathcal I$ 是可计算的“间接路径/下游抵消预算”。若实验显示 $L_W\to0$ 而
$S_{key}$ 不增，重点应是构造式 (T21) 的反例或补充条件，而不是宣称训练失败。

---

## 7. factorization 为什么改变训练动力学

令

\[
G_B=\nabla_B R,\qquad G_C=\nabla_C R.
\tag{T22}
\]

若直接训练 composites，Euclidean population gradient flow 是

\[
\dot B=-G_B,\qquad \dot C=-G_C.
\tag{T23}
\]

若训练 factors $B=Q^\top K$、$C=OV$，链式法则给出

\[
\dot Q=-KG_B^\top,\qquad \dot K=-QG_B,
\tag{T24}
\]

\[
\dot O=-G_CV^\top,\qquad \dot V=-O^\top G_C.
\tag{T25}
\]

所以 composite 自身满足

\[
\dot B=-G_BK^\top K-Q^\top QG_B,
\tag{T26}
\]

\[
\dot C=-G_CV^\top V-OO^\top G_C.
\tag{T27}
\]

同一个 $B,C$，不同 factor gauge 仍可能有不同速度；factorized training 相当于一个
随时间变化、左右不对称的预条件系统。并且

\[
QQ^\top-KK^\top=\text{constant},
\qquad
O^\top O-VV^\top=\text{constant}
\tag{T28}
\]

沿连续 GF 保持不变。它们是检查数值轨迹和构造闭合动力学时的重要守恒量。

### 定理目标 C（conditioning 与容量分离）

比较三条从相同 $B(0),C(0)$ 出发的轨迹：factorized、rank-matched direct、dense direct。

- rank-matched direct 修复 residual，而 dense/direct 都修复：支持 factorization geometry；
- 只有 dense direct 修复：支持 rank/function capacity；
- 三者都不修复：不能把问题归因于 factorization。

理论目标是用 $G_B,G_C$、factor imbalance 和 composite singular values 给出 residual 衰减率
的上/下界，并解释什么时候式 (T26)–(T27) 比式 (T23) 慢。

---

## 8. superposition 的不可避免性与可识别补偿

当 $C>d$ 且 $\|e_c\|=1$ 时，不可能让所有概念两两正交。最大 coherence

\[
\mu(E)=\max_{c\ne c'}|e_c^\top e_{c'}|
\tag{T29}
\]

至少满足 Welch bound

\[
\mu(E)\ge
\sqrt{\frac{C-d}{d(C-1)}}.
\tag{T30}
\]

对 $C=32,d=8$，下界约为 $0.3111$。因此 hard cell 的正确对照不是“不可能的正交
32 个向量”，而是 random unit codebook 与接近 Welch bound 的 tight frame；真正正交的
negative control 只能用 $C\le d$。

给定一个 label-preserving on-manifold swap $X\to X'$：只把非 target 概念换成当前
episode 未出现的合法概念，values、target、label 和 slot 数不变。自然 cross-talk 为

\[
I_{swap}=\mathbb E[f(X')-f(X)]^2.
\tag{T31}
\]

若 $I_{swap}$ 随 low-coherence fixed $E$ 大幅下降，最节约的解释是 dictionary collision；
若不下降，才继续检查 head bottleneck 或下游路径。

### finite suffix 定义

对模块 $M$ 的实际输入状态 $z_{M,e}$ 和 swap chord $\Delta_{M,e}$，其后代网络记作
$G_{M,e}$。真实 finite response 是

\[
p_{M,e}(\Delta)=
G_{M,e}(z_{M,e}+\Delta)-G_{M,e}(z_{M,e}).
\tag{T32}

它不是 Jacobian 乘向量。tangent $J_G(z)\Delta$ 只能作为小扰动近似，并须和式 (T32)
分别报告。

### 定理目标 D（learned downstream compensation）

寻找一个 module-wise 分解，使相同 on-manifold swap 的输出变化满足

\[
\Delta f=p_{QK}+p_{OV}+p_{FFN}+p_{readout}+p_{interaction},
\tag{T33}

\]

其中每一项由同一 base state 上的真实 counterfactual hybrid 定义，interaction remainder
精确闭合。要称某模块是 compensator，必须同时看到：其输入 finite energy 非零；其 signed
response 反向；实际 nonlinear suffix 衰减；跨 pairs、seeds 和第二 optimizer/architecture
复制。相邻两个 coherent donor patches 的输出等价不能作为衰减证据。

---

## 9. 训练时间、层深时间与 population GF

训练时间记为 $s$，层索引记为 $\ell$：

\[
\frac{d\theta_s}{ds}=-\nabla_\theta R(\theta_s),
\qquad
x_s^{\ell+1}=\Phi_{\theta_s^\ell}(x_s^\ell).
\tag{T34}

固定 $s$ 研究 $\ell$，得到的是给定参数下 token 如何聚类或传播；固定有限架构研究 $s$，
得到的是训练如何改变传播算子。二者不能混称一个“动力学”。

我们在小 population 上以 Euler 法

\[
\theta_{k+1}=\theta_k-\eta\nabla R(\theta_k)
\tag{T35}

逼近式 (T34)，并用 $\eta,\eta/2,\eta/4$ 的 step-halving 检验，而不是把一个任意 full-batch
学习率称作 gradient flow。

候选低维 order parameter 向量为

\[
z=(R,E_T,L_D,L_H,L_0,S_{key},r_{eff}(E),
\|B\|_F,\|C\|_F,QQ^\top-KK^\top,O^\top O-VV^\top).
\tag{T36}

### 定理/反例目标 E（closure）

判断是否存在 $F$ 使

\[
\dot z=F(z)
\tag{T37}

在注册分布和参数化下近似闭合。真正有力的反例是找到两组完整参数 $\theta,\theta'$，
满足 $z(\theta)=z(\theta')$，但 $\dot z(\theta)\ne\dot z(\theta')$。这会说明必须增加哪个
order parameter，而不是笼统说“动力学复杂”。

---

## 10. 与 fixed-parameter clustering 结果的精确关系

Transformer Perspective 中的 clustering 结果主要固定 $Q,K,V$，研究 token 随层深/连续
depth 聚集。我们的 Phase-I reproduction 甚至出现了：token 几乎完全聚到同一点，同时
attention 变成均匀分布。这说明

\[
\text{global representation clustering}
\centernot\Longrightarrow
\text{task-selective causal routing}.
\tag{T38}

本项目接在其上游：训练怎样选择 $B_s,C_s$，使固定 $s$ 时的层动力学具有任务相关结构。
若未来能从式 (T24)–(T27) 推出某类 learned interaction kernel 再满足 clustering theorem 的
条件，才真正建立

\[
\text{parameter training dynamics}
\Longrightarrow
\text{learned interaction kernel}
\Longrightarrow
\text{depth-wise representation dynamics}.
\tag{T39}

当前实验直接测第一条箭头；Perspective 的证明主要约束第二条箭头之后的行为。

---

## 11. 什么结果可以升级成论文主问题

### 主问题 1：复合 routing kernel 的训练选择

只有当长训练、cosine、rank-matched direct、fixed low-coherence $E$、真正的 head-capacity
controls 和第二 optimizer 都无法消除 residual，且注册的 per-slot $S_{key}$ 与非对称 finite
QK chord 均已测量，才研究：

> 在联合学习 $E$ 和 factorized $Q/K,O/V$ 时，式 (T24)–(T27) 在什么条件下选择满足
> 式 (T14)/(T21) 的 causal routing kernel？

### 主问题 2：learned superposition 的下游补偿

只有当 swap 在上游产生非平凡 finite energy，而最终输出 cross-talk 很小，并能以式
(T32)–(T33) 在模块级稳定定位时，才研究：

> 梯度训练为什么选择一个 upstream non-orthogonal representation，再由特定 OV/FFN/
> readout path 抵消其 cross-talk；该选择相对直接去相关有哪些容量或隐式正则优势？

如果 upstream energy 本来就近零，这不是“神秘补偿”，而是补偿假说的反例。如果 effect
在模块间不可唯一分配，应研究 distributed/non-identifiable compensation，而不能挑一张
activation-patching 图命名机制。

---

## 12. 面向实验读者的判读表

| 观察 | 最小结论 | 不能直接说 |
|---|---|---|
| $R,L_W\to0$ | 功能上正确 retrieval | 某个 head 做了 causal routing |
| $S_{key}>0$ | direct query-memory edges 有 target-selective effect | 所有间接 causal paths 已定位 |
| rank-matched direct 修复 | factorization conditioning 候选 | dense capacity 是原因 |
| 仅 dense direct 修复 | rank/function capacity 候选 | 纯优化几何 |
| low-coherence fixed $E$ 修复 | dictionary collision 候选 | learned E 一定差 |
| fixed $d_h$ 修复 | per-head bottleneck 候选 | head 数本身有害 |
| finite FFN response 抵消 | FFN compensation 候选 | tangent 或 coherent patch 已证明补偿 |
| frozen Pythia checkpoint 有差异 | 该 checkpoint 的描述性机制 | 训练诱导，除非有 checkpoint trajectory |

这个表是后续报告的 claim ladder。任何 headline 都必须能回指一条公式、一个 seed-level
estimand、一个校正 family 和一个明确反事实。
