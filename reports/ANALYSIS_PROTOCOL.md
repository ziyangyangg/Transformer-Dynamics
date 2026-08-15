# Confirmatory analysis protocol

本协议固定“一个训练随机种子到底贡献一个什么观测量”、哪些比较是配对的、
何时可以使用 `causal` / `compensation` / `gradient-flow-like` 等词。任何阈值、
方向或站点都必须在读取确认性随机种子的结果之前写入配置和分析清单；改变这些定义
会生成新的实验版本，而不是覆盖旧结果。

## 1. 概率空间、因果语义与独立实验单位

一次 episode 的外生随机变量为

\[
U=(c_{1:m},v_{1:m},J),\qquad
c_{1:m}\sim\operatorname{Unif}\{\text{有序无重复的 }m\text{-tuple}\},
\]

\[
v_i\overset{\mathrm{iid}}\sim\operatorname{Unif}\{-1,+1\},\quad
J\sim\operatorname{Unif}[m],\quad q=c_J,\quad Y=v_J.
\]

网络中的 score、attention、mixture、residual 和 prediction 都是给定参数
\(\theta\) 与 \(U\) 后的确定性结构方程。本文只把以下三类量称为因果量：

1. **外生变量干预**：固定 \((c_{1:m},v_{-i},J)\)，执行
   \(\operatorname{do}(v_i=+1)\) 与 \(\operatorname{do}(v_i=-1)\)；
2. **结构内路径干预**：把明确指定的内部结构方程替换为常量，例如
   \(\operatorname{do}(s^{\ell h}_{qJ}=-\infty)\)，并重算所有后代；
3. **activation patch**：从同一模型的另一个合法 episode 取得 donor 节点值，
   替换 recipient 的指定内部节点并重算后代。这是一个跨 episode 的内部节点干预，
   不是自然直接效应，也不自动具有现实世界语义。

embedding chord、JVP、梯度、probe 与相关性均只称为**局部机械诊断**，不能单独称为
因果效应。

独立推断单位是训练种子 \(r\)，不是 evaluation episode。每个种子必须从一个 master
seed 派生互不重叠的

```text
init_seed, train_data_seed, eval_base_seed, eval_value_seed,
eval_swap_seed, patch_seed, diagnostic_seed
```

子流。相同 seed id 的不同 \((C,H,\text{optimizer},\text{architecture})\) cell 共用同一批
可比较的 evaluation episode；这构成 blocking，而不意味着不同形状的参数初始化相同。

## 2. Evaluation 样本与 seed-level 汇总

每个 checkpoint 至少使用 8192 个从未参与训练的 base episode。以 32 个 chunk、每个
chunk 256 个 episode 计算 Monte Carlo 误差。训练种子 \(r\) 的任何 estimand 都先在其
evaluation episode 内平均成一个标量 \(T_r\)，再跨 seed 推断。

对于 \(m\leq 8\) 的 primary/scaling cell，先抽取 concept tuple 和 target \((c_{1:m},J)\)，
然后枚举全部 \(2^m\) 个 \(v\in\{-1,+1\}^m\)。这使 value Fourier/Walsh 分解在给定
concept tuple 后是精确的，而不是由随机 value 造成的近似。

每个 cell 的基础 seed-level 量为

\[
\widehat R_r=\frac1{2n}\sum_{e=1}^n(f_{re}-y_e)^2,
\qquad
\widehat A_r=\frac1n\sum_{e=1}^n
\mathbf 1\{\operatorname{sign}(f_{re})=y_e\}.
\]

预测恰为零时记为错误。所有平方量使用 float64 accumulation；模型 forward 可以使用
float32，但每个 patch replay 必须先通过确定性重放测试。

### 2.1 Monte Carlo 充分性

对 seed-level 均值 \(T_r=n^{-1}\sum_e t_{re}\)，用 32 个 evaluation chunk 的方差
估计 \(\operatorname{SE}_{MC}(T_r)\)。只有当

\[
\operatorname{median}_r\operatorname{SE}_{MC}(T_r)
\leq 0.2\,
\frac{\operatorname{sd}_r(T_r)}{\sqrt{R}}
\]

时，evaluation 样本量才算足够；否则只增加 held-out episode，不增加训练 seed。
若跨 seed 方差接近零，则改用绝对规则
\(\operatorname{SE}_{MC}(T_r)\leq 10^{-3}\)（prediction-scale 指标）或相应预注册尺度。

## 3. 复合 routing 的可操作定义

对固定 \((c_{1:m},J)\)，把完整网络输出视作 value hypercube 上的函数
\(f(v_1,\ldots,v_m)\)。其 Walsh 系数是

\[
\widehat f_S(c,J)=2^{-m}\sum_{v\in\{-1,+1\}^m}
f(c,v,J)\prod_{i\in S}v_i,
\qquad S\subseteq[m].
\]

定义第 \(i\) 个 slot 的**端到端复合 routing kernel**

\[
\kappa_i(c,J):=\widehat f_{\{i\}}(c,J)
=\frac12\,\mathbb E_{v_{-i}}
\left[f(\operatorname{do}(v_i=+1))-f(\operatorname{do}(v_i=-1))\right].
\]

它包括 embedding、所有层 QK、OV、FFN、residual 与 readout 的乘积/组合，因此不等同于
任一 attention probability。精确 Parseval 恒等式给出

\[
\mathbb E_v[(f(v)-v_J)^2\mid c,J]
=(\kappa_J-1)^2
+\sum_{i\ne J}\kappa_i^2
+\widehat f_\varnothing^2
+\sum_{|S|\ge2}\widehat f_S^2. \tag{1}
\]

这四项分别叫 target coefficient error、distractor routing leakage、bias leakage 与
nonlinear/higher-order leakage。式 (1) 是数值单元测试：四项之和与枚举得到的条件 MSE
相对误差必须小于 \(10^{-6}\)（float64 analysis）。

每个 seed 的确认性 routing 量为

\[
K^{\rm target}_r=\mathbb E_{c,J}[\kappa_J],\quad
L^{\rm distractor}_r=\mathbb E_{c,J}\sum_{i\ne J}\kappa_i^2,
\]

\[
L^{\rm high}_r=\mathbb E_{c,J}\sum_{|S|\ge2}\widehat f_S^2,
\quad
E^{\rm route}_r=\mathbb E_{c,J}
\left[(\kappa_J-1)^2+\sum_{i\ne J}\kappa_i^2\right].
\]

注册的 value-flip statistic

\[
\Xi_{{\rm value},r}
=\frac12\mathbb E\left[Y\{f(X)-f(\operatorname{do}(v_J=-v_J))\}\right]
\]

在该分布上恰等于 \(K^{\rm target}_r\)。两种独立实现的绝对差必须小于 \(10^{-5}\)。
因此，低风险模型的 \(\Xi_{\rm value}\approx1\) 是必要的功能核验，不是独立的新发现。

### 3.1 直接 key 路径与 target selectivity

对 episode \(U=(X,Y)\) 和 slot \(i\)，在所有层和 head 同时执行
\(s^{\ell h}_{qi}\leftarrow-\infty\)，重算 softmax 与全部后代：

\[
\delta_{i,r}(U)=Y\left\{f(X)-
f\bigl(\operatorname{do}(s^{\ell h}_{qi}=-\infty,\ \forall\ell,h)\bigr)
\right\}.
\]

主要的 key-path selectivity 是

\[
S_{{\rm key},r}=\mathbb E_U\left[
\delta_{J,r}(U)-\frac1{m-1}\sum_{i\ne J}\delta_{i,r}(U)
\right]. \tag{2}
\]

\(\mathbb E\delta_J\) 不是总 value 因果效应：target value 可能先进入其他 memory token，
再间接到 query。
因此只有 (2) 显著为正时才可以说“直接 query-to-target key 路径具有任务选择性”；
\(\Xi_{\rm value}>0\) 只允许说“输出因果依赖 queried value”。

**发布偏差。** 当前 evaluator 只保存了 target-edge blocking effect
\(\mathbb E\delta_J\)，没有逐个阻断 distractor edges，因而没有估计 (2)。结果表中的
`target-edge + attention screen` 还结合了描述性的 target-minus-distractor attention mass；
它不是 causal key selectivity，注册的 \(S_{\rm key}\) 在本批实验中尚未评估。

## 4. 表示几何与 superposition 的 seed-level estimands

令训练后 concept matrix \(E\in\mathbb R^{C\times d}\)，奇异值为 \(\sigma_j\)，
归一化行 \(u_c=E_c/\|E_c\|_2\)。主要几何量是

\[
r_{{\rm eff},r}(E)=
\frac{(\sum_j\sigma_j^2)^2}{\sum_j\sigma_j^4},\qquad
\widetilde r_{{\rm eff},r}=r_{{\rm eff},r}/d,
\]

\[
\mu_r(E)=\max_{c\ne c'}|u_c^\top u_{c'}|,\qquad
D_{c,r}=\frac{\|E_c\|_2^2}{\sum_{c'}(u_c^\top E_{c'})^2}.
\]

每个 checkpoint 必须验证

\[
\sum_cD_{c,r}\leq \operatorname{rank}(E)+10^{-5}\leq d+10^{-5}.
\]

`C>d`、低 \(r_{eff}\) 或高 coherence 只支持“compressed dictionary geometry”。只有当
第 5 节的 on-support functional cross-talk 显著，并且第 6 节定位出下游抑制时，才进入
“learned superposition with downstream compensation”的候选证据层级。

### 4.1 与 fixed-parameter clustering 对照的逐层表示量

在注册 site \(s\) 记一个 episode 的 residual states 为
\(X^{s}_{b}=(x^{s}_{b1},\ldots,x^{s}_{bT})\in\mathbb R^{T\times d}\)，其中
\(T=m+1\)。本任务**没有 padding**：\(1,\ldots,m\) 全部是真实 memory card，\(T\) 固定是
query。因此任何全 token 统计都使用全部 \(T\) 行，不进行 mask、长度加权或跨 episode
pooling。令

\[
u^s_{bi}=\frac{x^s_{bi}}{\|x^s_{bi}\|_2},\qquad
G^s_{b,ij}=(u^s_{bi})^\top u^s_{bj},
\]

并约定精确零向量的 \(u=0\)。在每个 seed 的固定 evaluation batch 内记录

\[
\rho^s_{target}=\mathbb E_b G^s_{b,TJ_b},\qquad
\rho^s_{distractor}=\mathbb E_b\frac1{m-1}\sum_{i\ne J_b}G^s_{b,Ti},
\]

\[
\Delta\rho^s=\rho^s_{target}-\rho^s_{distractor},\qquad
\rho^s_{global}=\mathbb E_b\frac1{T(T-1)}\sum_{i\ne j}G^s_{b,ij}. \tag{R1}
\]

token covariance 必须先在每个 episode 内中心化：

\[
\widetilde X^s_b=X^s_b-\mathbf1_T\bar x_b^{s\top},\qquad
\Sigma^s_b=T^{-1}\widetilde X_b^{s\top}\widetilde X^s_b,
\]

\[
r^s_{token}=\mathbb E_b
\frac{\operatorname{tr}(\Sigma^s_b)^2}
{\operatorname{tr}((\Sigma^s_b)^2)}, \tag{R2}
\]

其中 \(\Sigma=0\) 时该 episode 的 rank 约定为 0。不能先把不同 episode 的 token pool
起来再求 covariance；那会把不同 concept identities 的变化误当成单个序列内部的维度。

这些量在 `input_embeddings` 以及每层的 `post_attention_residual`、
`post_ffn_residual` 都记录。较高的 \(\rho_{global}\) 配合较低的 \(r_{token}\) 是全局
clustering/collapse 的描述证据；较高的 \(\Delta\rho\) 才是 target-selective geometry。
二者都不是因果 routing 结论，仍须由第 3 节的 value/key intervention 和第 5--6 节的
on-support swap/localization 验证。

## 5. On-support swap 与内部 patch 量

从 base episode \(X\) 选 \(K\ne J\)，再选不在 \(c_{1:m}\) 中的 \(c_{new}\)，构造

\[
X'=\operatorname{swap}(X; c_K\leftarrow c_{new}),
\]

其中 value、target index、query 与 label 不变。每个 pair 在 forward 前必须断言：memory
concept 仍互异、\(K\ne J\)、新 concept 原先缺席、\(q'=q\)、\(Y'=Y\)。

最终函数的 swap sensitivity 为

\[
I_{{\rm output},r}=\mathbb E[(f(X')-f(X))^2]. \tag{3}
\]

对注册节点 \(Z_s\)，recipient 为 \(X\)、donor 为 \(X'\)：

\[
p_{s,e}=f(X_e;\operatorname{do}(Z_s=Z_s(X'_e)))-f(X_e),\qquad
I_{s,r}=\mathbb E_e[p_{s,e}^2]. \tag{4}
\]

主要方向固定为 \(X\leftarrow X'\)。反向 \(X'\leftarrow X\) 单独报告；若两个方向的
seed-level effect 符号结论不一致，则降级为方向敏感的 exploratory observation。

### 5.1 注册 patch sites

每层 \(\ell\) 和 head \(h\) 保存以下 query-row 节点：

```text
qk_scores[ell,h]          [visible_tokens]
attention[ell,h]          [visible_tokens]
pre_ov_mixture[ell,h]     [d_head or d, according to implementation]
post_ov_update[ell,h]     [d]
post_attention_query[ell] [d]
ffn_branch_query[ell]     [d]
post_ffn_query[ell]       [d]
prediction                scalar
```

`qk_scores` patch 后必须由 recipient 重新 softmax；`attention` patch 必须检查和为 1 且
不含 causal-mask 外位置；per-head patch 只替换一个 head，其余 head 保持 recipient；
residual patch 默认只替换 query row。补充的 all-token patch 是 secondary，因为它把
memory-to-memory 间接路径也一起改动。

### 5.2 禁止的相邻节点推断

若 \(Z_{after}=g(Z_{before})\) 且两次 patch 都完整替换 donor 值并重算同一个确定性
\(g\)，则

\[
f(\operatorname{do}(Z_{before}=Z'_{before}))
=f(\operatorname{do}(Z_{after}=g(Z'_{before}))). \tag{5}
\]

例如同一 head 的完整 pre-OV mixture patch 与对应 post-OV update patch 理论上等价。
式 (5) 的差应小于 \(10^{-6}\)，它是 patch implementation test，**不是** OV
compensation test。任何违反 (5) 的结果先按 instrumentation bug 处理。

## 6. 补偿的可识别分解与 localization ratios

### 6.1 QK route、content 与 finite interaction

在同一层/head，base trace 为 \((a_i,z_i,m)\)，donor trace为
\((a'_i,z'_i,m')\)。使用下列精确有限差分：

\[
\delta m_{route}=\sum_i(a'_i-a_i)z_i,
\quad
\delta m_{content}=\sum_i a_i(z'_i-z_i),
\]

\[
\delta m_{interaction}=\sum_i(a'_i-a_i)(z'_i-z_i),
\quad
m'-m=\delta m_{route}+\delta m_{content}+\delta m_{interaction}. \tag{6}
\]

令 \(\delta u_{h,p}=C_{\ell h}\delta m_{h,p}\)，
\(p\in\{route,content,interaction\}\)。式 (6) 在每个 example 上的相对重构误差必须
小于 \(10^{-5}\)。它回答“swap 造成的 query update 是来自 QK 权重改变、被混合内容
改变，还是二者的有限交互”，避免把所有 attention 变化都叫 routing。

在该 attention residual 之后取 downstream adjoint
\(r=\nabla_{x^{att}_{q}}f\)。一阶 output-relevant 分量为

\[
t_{h,p}=L^{-1/2}r^\top\delta u_{h,p}.
\]

跨 head/component 的精确一阶抵消率定义为

\[
\eta^{tangent}_{att}
=1-\frac{|\sum_{h,p}t_{h,p}|}
{\sum_{h,p}|t_{h,p}|+10^{-12}}\in[0,1]. \tag{7}
\]

QK-route 对 content cross-talk 的特定抑制 contrast 为

\[
C_{QK,r}=\mathbb E_e\left[
\log\frac{(t_{content}+t_{interaction})^2+10^{-12}}
{(t_{route}+t_{content}+t_{interaction})^2+10^{-12}}
\right], \tag{8}
\]

并同时报告 opposition rate

\[
O_{QK,r}=\Pr_e[t_{route}(t_{content}+t_{interaction})<0].
\]

只有 (8) 为正、\(O_{QK}>1/2\)，且第 5 节 finite patch 方向一致，才把 QK 称为候选
compensator。

#### 6.1.1 实现偏差：当前快照使用对称 midpoint split

首轮机制快照没有保存式 (6) 的三个非对称 endpoint 项。实现采用了另一个同样精确、
但**不等价于本节预注册 estimand** 的对称双线性恒等式。令
\(\bar a=(a+a')/2\)、\(\bar z=(z+z')/2\)，则实现记录

\[
\delta m_{content}^{sym}=\sum_i\bar a_i(z'_i-z_i)
=\delta m_{content}+\tfrac12\delta m_{interaction},
\]

\[
\delta m_{route}^{sym}=\sum_i(a'_i-a_i)\bar z_i
=\delta m_{route}+\tfrac12\delta m_{interaction},
\qquad
m'-m=\delta m_{content}^{sym}+\delta m_{route}^{sym}. \tag{8a}
\]

当前 CSV 中沿用的字段名 `qk_suppression_log_ratio` 实际计算

\[
C_{QK}^{sym}=\mathbb E_e\log
\frac{(t_{content}^{sym})^2+10^{-12}}
{(t_{content}^{sym}+t_{route}^{sym})^2+10^{-12}}, \tag{8b}
\]

而不是式 (8)。有限 interaction 被一半分给 content、一半分给 route，因此二者甚至可能
符号相反。例如标量 \(t_{content}=1,t_{route}=0.05,t_{interaction}=-0.2\) 时，式 (8) 的
log-ratio 为负，而式 (8b) 为正。故现有 midpoint 结果只能作为**探索性 protocol
deviation**：它可以反对一个朴素的 midpoint suppression story，但没有检验、更没有反驳
预注册的式 (8)。正式确认性重放必须同时输出三个 endpoint 项、式 (8) 与 finite hybrid
validation；不得事后把式 (8b) 改名为预注册结果。

### 6.2 OV 的方向选择性（不是相邻 patch attenuation）

由于 OV 是线性映射，pre/post-OV coherent patch 不能识别补偿。对 swap cross-talk
方向 \(\delta m\) 计算

\[
g_{swap}=\frac{\|C_{\ell h}\delta m\|_2^2}
{\|\delta m\|_2^2+10^{-12}},\qquad
g_{iso}=\frac{\|C_{\ell h}\|_F^2}{\dim(m)},
\]

\[
A_{OV,\ell h,r}=\mathbb E_e
\log\frac{g_{iso}+10^{-12}}{g_{swap}+10^{-12}}. \tag{9}
\]

正的 (9) 表示 OV 对实际 swap 方向的增益小于各向同性方向的平均增益。必须进一步
报告从初始化到训练后的配对变化

\[
\Delta A_{OV,r}=A_{OV,r}^{final}-A_{OV,r}^{init};
\]

只有 \(\Delta A_{OV}>0\) 才支持“训练选择了与 cross-talk 对齐的低增益方向”。(9) 是
方向选择性证据；除非再通过第 6.4 节的 finite output test，否则不能单独称为因果补偿。

多 head post-OV cancellation 另记为

\[
\eta^{tangent}_{heads}
=1-\frac{|\sum_h r^\top C_h\delta m_h|}
{\sum_h|r^\top C_h\delta m_h|+10^{-12}}. \tag{10}
\]

### 6.3 FFN residual cancellation

令同一层 attention 后的 base/donor query state 为 \(x,x'\)：

\[
\delta x_{skip}=x'-x,\qquad
\delta x_{ffn}=L^{-1/2}\left[
F(\operatorname{RMSNorm}(x'))-
F(\operatorname{RMSNorm}(x))\right],
\]

\[
\delta x_{post}=\delta x_{skip}+\delta x_{ffn}. \tag{11}
\]

在 FFN residual 之后取 \(r=\nabla_{x_{post}}f\)，定义

\[
t_{skip}=r^\top\delta x_{skip},\quad
t_{ffn}=r^\top\delta x_{ffn},
\]

\[
C_{FFN,r}=\mathbb E_e\log
\frac{t_{skip}^2+10^{-12}}
{(t_{skip}+t_{ffn})^2+10^{-12}},\qquad
O_{FFN,r}=\Pr_e[t_{skip}t_{ffn}<0]. \tag{12}
\]

`before` cross-talk 的 practical floor 是
\(\mathbb E[t_{skip}^2]\ge10^{-4}\operatorname{Var}(Y)=10^{-4}\)。低于该值时
`C_FFN` 记为 `not-identifiable`，不能用极小分母制造很大的 log ratio。

### 6.4 Finite on-support validation of tangent localization

一阶抵消必须用 finite intervention 验证。令 \(G_\ell\) 是从某 residual 输出到最终
prediction 的固定 downstream suffix，并在 base residual state \(z\) 注入 (11) 的分量：

\[
p_{skip}=G_\ell(z+\delta x_{skip})-G_\ell(z),
\]

\[
p_{ffn}=G_\ell(z+\delta x_{ffn})-G_\ell(z),
\]

\[
p_{joint}=G_\ell(z+\delta x_{skip}+\delta x_{ffn})-G_\ell(z),
\quad
p_{nonlin}=p_{joint}-p_{skip}-p_{ffn}.
\]

有限抵消率为

\[
\eta^{finite}_{FFN,r}=\mathbb E_e\left[
1-\frac{|p_{joint}|}
{|p_{skip}|+|p_{ffn}|+|p_{nonlin}|+10^{-12}}
\right]. \tag{13}
\]

attention 的 route/content/interaction 以相同方法定义 finite \(p_p\)。候选 compensator
必须满足：tangent contrast 的 simultaneous 95% CI 在抑制方向、finite contrast 同号、
且至少 60% evaluation pair 同号。60% 是预注册的异质性门槛，不可在看过结果后改变。

## 7. Primary 与 secondary estimands

### 7.1 Primary family A：representation/function decoupling

固定 \(d=16,L=2,m=4\)，对每个 matched seed 定义

\[
\Delta^{rank}_r=
\{r_{eff,r}(C=64,H=4)-r_{eff,r}(C=64,H=1)\}
-\{r_{eff,r}(C=16,H=4)-r_{eff,r}(C=16,H=1)\}. \tag{14}
\]

主要 estimand 是 \(I_{rank}=\mathbb E_r[\Delta^{rank}_r]\)。预注册方向是
\(I_{rank}<0\)：在高 load 下，多 head 相对单 head 选择更低 effective-rank 的 concept
geometry。只有第 9 节 functional matching gate 通过，(14) 才解释为不同内部表示实现了
匹配的 retrieval function。

同一个 2×2 paired interaction 也用于

\[
E^{route},\ L^{distractor},\ L^{high},\ I_{output},\ S_{key}
\]

作为功能不变性/机制 secondary endpoint。特别地，`causal interaction approximately
zero` 必须通过等效性检验，不能由“不显著”推出。

### 7.2 Primary family B：compensation localization

对每个存在且通过 practical floor 的 layer/module，primary seed-level contrast 是

\[
T_{M,\ell,r}\in
\{C_{QK,\ell,r},\ \Delta A_{OV,\ell,r},\ C_{FFN,\ell,r}\}. \tag{15}
\]

headwise 数值先在 seed 内按 head 等权平均；head specialization 的离散分布另作
secondary，不能把 head 当独立样本。第 10 节对 (15) 的所有 layer/module 做 family-wise
校正。只有某个具体 \((M,\ell)\) 通过校正、finite validation 与 replication gate，才允许
说 cross-talk “在该 module/layer 被抑制”。

### 7.3 Secondary estimands

以下均为 secondary：

- \(\widetilde r_{eff}\)、coherence、\(\sum_cD_c\)、embedding Gram spectrum；
- 每个 head 的 target attention、logit margin
  \(s_{qJ}-\log\frac1{m-1}\sum_{i\ne J}e^{s_{qi}}\)；
- \(B_{\ell h}=Q_{\ell h}^\top K_{\ell h}\)、
  \(C_{\ell h}=O_{\ell h}V_{\ell h}\) 的谱、有效秩和训练漂移；
- head ablation、single-block freeze/only-train、all-token patch；
- checkpoint trajectory、NTK、Hessian、loss landscape；
- 未预注册的 layer/head/site 与三阶以上 interaction。

它们可以解释 primary effect，但不能替代 primary estimand。

## 8. 配对估计、bootstrap 与标准化效应

所有 primary comparison 以 seed id 为 block。若共有 \(R\) 个完整配对，

\[
\widehat I_{rank}=R^{-1}\sum_{r=1}^R\Delta^{rank}_r,\qquad
d_z=\frac{\overline\Delta}{s_\Delta},
\]

其中 \(s_\Delta\) 使用 \(R-1\) 分母。报告每个 \(\Delta_r\)、均值、标准差、\(d_z\) 与
CI；不把 episode 数代入 \(d_z\)。

使用 20,000 次 paired seed block bootstrap。第 \(b\) 次从
\(\{1,\ldots,R\}\) 有放回抽取 \(R\) 个 seed id，抽中的 seed 在所有 cell、site、layer
一起重复：

\[
\widehat I^{*(b)}=R^{-1}\sum_{j=1}^R\Delta_{r_j^{(b)}}.
\]

未校正的 95% percentile CI 是
\([q_{.025}(I^*),q_{.975}(I^*)]\)。bootstrap RNG seed 固定在 analysis manifest。
若 \(R<10\)，不做确认性结论，只显示所有 seed 点与 exploratory interval。

缺失 cell 时，primary 使用完整配对交集；同时报告所有有限 seed 的 unpaired sensitivity
analysis 与缺失原因。禁止用不同 seed 拼成伪配对。function-qualified 分析和
intention-to-train（所有计划 seed，包括失败率）必须同时报告。

## 9. 功能、因果与可比性 gates

### 9.1 单 seed function gate

一个训练 seed 只有同时满足

\[
\widehat A_r\ge0.95,\qquad
\widehat R_r\le0.05,\qquad
\Xi_{{\rm value},r}\ge0.90 \tag{16}
\]

才进入 `successful-solution` 机制比较。每个 cell 至少有 10 个通过 seed，且通过比例至少
80%；否则 cell failure 仍计入 intention-to-train，并触发第 12 节 remedy。

对 compensation claim，donor endpoint 还必须满足

\[
A_r(X')\ge0.95,\qquad I_{{\rm output},r}\le2.5\times10^{-3}. \tag{17}
\]

(17) 表示 RMS output swap sensitivity 不超过 label scale 的 5%。若 (17) 不通过，正确
结论是模型本身未对无关 distractor identity 保持函数不变，不能说下游“成功消除”了
cross-talk。

### 9.2 cell 之间 functional matching

表示比较的两个 cell 必须通过 bootstrap TOST 等效性。等效边界预注册为

\[
|\Delta A|<0.02,\qquad |\Delta\Xi_{value}|<0.05,\qquad
|\Delta E^{route}|<0.02. \tag{18}
\]

具体实现要求对应差异的 paired 90% bootstrap CI 完全落在
\((-0.02,0.02)\)、\((-0.05,0.05)\)、\((-0.02,0.02)\) 内。`p>0.05` 不等于匹配。
若 (18) 不通过，(14) 只能表述为 performance-confounded geometry difference。

### 9.3 causal routing gate

“成功使用 queried value”要求 (16)。更强的“直接 target-key selective routing”要求

\[
\mathbb E_r S_{key,r}>0
\]

的 paired/bootstrap 95% CI 下界大于零，并且 \(D_J>0\) 的 CI 下界大于零。若
\(\Xi_{value}\) 高但 \(S_{key}\) 不通过，应报告 alternative/indirect routing，而不是把
它归为训练失败。

## 10. Multiplicity 与 trajectory inference

`I_rank` 是唯一方向性 primary geometry test，使用双侧 family alpha 0.05，并同时报告
预注册方向。所有 \((M,\ell)\) localization tests 构成一个 family，使用 paired-seed
studentized max-\(T\) bootstrap：

1. 对每个站点 \(k=(M,\ell)\) 计算
   \(T_k=\bar T_k/(s_k/\sqrt R)\)；
2. 在 seed block 内中心化，并以 Rademacher sign flip 生成 100,000 个 null replicate；
3. 每次保存 \(\max_k|T_k^*|\)，其 0.95 quantile 为 \(c_{.95}\)；
4. simultaneous CI 为
   \(\bar T_k\pm c_{.95}s_k/\sqrt R\)。

practical-floor 不通过的站点在看方向前标记为不可识别，不能先删除不利站点再校正。
未注册的 secondary scalar family 使用 Benjamini-Hochberg \(q=0.10\)，并明确标为
exploratory。图中的逐 checkpoint 轨迹使用同一 seed-block bootstrap 的
\(\max_t|T_t^*|\) simultaneous band；禁止从许多 pointwise CI 中挑一个 checkpoint
声称 phase transition。

## 11. Optimizer、architecture 与 gradient-flow replication gates

### 11.1 Optimizer replication

AdamW（weight decay 0）为发现/工程优化器，tuned momentum SGD 为首个优化器控制。对
一个带方向的 primary effect \(T\)，optimizer replication 通过需满足：

1. 两个 optimizer 各自通过第 9 节 cell/function gate；
2. 两个估计同号；
3. SGD 的未校正 95% paired bootstrap CI 排除零；
4. optimizer×effect difference 的 CI 不支持符号翻转；
5. 每个 optimizer 至少 10 个完整 successful-solution seed pairs。

若只满足同号而 SGD CI 包含零，状态为 `qualitative optimizer agreement`，不是 replicated。

### 11.2 Architecture replication/control

对 representation/function decoupling，attention-only 与 FFN-width-\(2d\) 必须各自满足
(14)、(16)、(18) 同号且各自 CI 排除零，才称 architecture replicated。

对“存在下游补偿”这一宽命题，两种 architecture 可定位到不同 module，但各自必须在
其可用 module family 内至少一个站点通过第 6、9、10 节。对“FFN 是 compensator”这一
窄命题，只能在包含 FFN 的架构中跨 optimizer/width/depth 复制；attention-only 是
negative architecture control，不是假装具有同一个 module 的 replication。

### 11.3 Population-gradient-flow claim gate

AdamW 或 momentum SGD trajectory 不直接等同于
\(\dot\theta=-\nabla R(\theta)\)。要把 order-parameter trajectory 称为
`gradient-flow-like`，必须额外运行无 momentum、无 weight decay 的 large-batch plain
gradient descent，并同时通过：

\[
\nu_t=\frac{\mathbb E\|g_{batch,t}-g_{reference,t}\|_2^2}
{\|g_{reference,t}\|_2^2+10^{-12}}\le0.1
\]

（初始化、routing onset、loss midpoint、final 四个 checkpoint），以及 step-halving
curve convergence

\[
D(z)=\frac{\{\sum_t[z_{\eta}(t)-z_{\eta/2}(t)]^2\}^{1/2}}
{\{\sum_tz_{\eta/2}(t)^2\}^{1/2}+10^{-12}}\le0.10 \tag{19}
\]

for \(z\in\{R,K^{target},E^{route},r_{eff},\|B\|_F,\|C\|_F\}\)，时间按
\(s=\eta\times\text{step}\) 对齐。没有 (19) 时，轨迹只能称为 discrete optimizer
dynamics。

## 12. 失败分类与预注册 remedy 顺序

每个计划 seed 都必须进入 `failures.jsonl` 或正常结果表。失败类型固定为：

```text
NUMERICAL_NAN_INF
PATCH_REPLAY_MISMATCH
SUPPORT_ASSERTION_FAILURE
FOURIER_IDENTITY_FAILURE
TANGENT_RECONSTRUCTION_FAILURE
OPTIMIZATION_GATE_FAILURE
EVALUATION_MC_INSUFFICIENT
RESOURCE_INTERRUPTION
CONFIGURATION_ERROR
```

遇到失败按以下顺序处理，且 remedy 对同一 comparison 的所有 cell 对称应用：

1. instrumentation/support/Fourier/JVP 失败：停止 grid，修复并递增 code/config version；
   旧 seed 全部作废但保留，不把 bug run 混入新版；
2. Monte Carlo 不足：只将 evaluation episode 加倍，checkpoint 不变；
3. optimization gate 失败：训练步数依次扩为 2×、4×；
4. 仍失败时，在独立 remedy seeds 上按固定集合
   \(\eta\times\{0.25,0.5,1,2\}\) 选择 validation risk 最低且稳定的 learning rate；
5. 用从未参与调优的新 confirmatory seeds 重跑所有比较 cell；
6. 仍有超过 20% seed 失败，则报告 optimization phase boundary；不得只删除失败 seed。

optimizer remedy 只能依据 risk、NaN 与 gate pass rate，不能依据希望得到的 rank、patch、
NTK 或 compensation 方向。任何 gradient clipping、warmup、初始化尺度、batch size 或模型
结构改变都形成新 protocol version。禁止对同一 seed 反复重启直到成功。

本项目已经完成的 b=2,048 follow-up **没有执行第 5 步**：它是在先前筛出的困难
cells 3/7（另含边界 cells 6/11）上，继续使用训练 seeds 0--9 比较固定的延长/降学习率
schedule。因此这些 paired bootstrap 区间只量化这批已选择 trajectories 上的
targeted remedy effect；它们是高精度、同 seed 的**探索性诊断**，不是独立 remedy seeds
或 never-tuned confirmatory seeds 上的确认性推断。任何正式的 optimization phase-boundary
命题仍须按第 4--6 步重新采样并做 family-level correction。

## 13. Loss-landscape diagnostics

这些诊断用于排除优化伪象，不证明 routing 或 compensation。

### 13.1 局部二维切片

在 \(\theta_*\) 取两个由固定 diagnostic seed 生成的高斯方向 \(d_1,d_2\)。对每个参数
tensor \(k\) 做 filter normalization：

\[
d_{j,k}\leftarrow d_{j,k}
\frac{\|\theta_{*,k}\|_2}{\|d_{j,k}\|_2+10^{-12}}.
\]

在 \((\alpha,\beta)\in[-1,1]^2\) 的 41×41 grid 评估
\(R(\theta_*+\alpha d_1+\beta d_2)\)，所有点使用同一固定 65,536-example population
probe。图必须同时标出中心截面、色标上限与超过上限的点，不能由每张图各自缩放制造
视觉差异。

### 13.2 Seed-to-seed path barrier

不同 seed 的 head 先根据 held-out trace 上的 attention vector 与 post-OV update 的联合
cost 用 Hungarian matching 对齐；同时报告未对齐 sensitivity。对齐后的线性路径
\(\theta(t)=(1-t)\theta_A+t\theta_B\)，\(t\in\{0,.02,\ldots,1\}\)，barrier 为

\[
B_{AB}=\max_t\left[R(\theta(t))-{(1-t)R(\theta_A)+tR(\theta_B)\}\right]. \tag{20}
\]

若测试 one-bend polygonal path，bend 只可在独立 construction set 上优化，在固定
evaluation set 上报告；不能用 evaluation loss 同时找路径和评估 barrier。

### 13.3 Hessian

在初始化、routing onset 与 final checkpoint，用同一 8192-example probe 的 HVP 做
Lanczos top-20 eigenvalues，并用 64 个固定 Rademacher probes 做 Hutchinson trace。
报告 negative eigenvalue、largest positive eigenvalue、trace 与 probe standard error。
单 seed 的负曲率或极小 eigenvalue 只称诊断；必须跨 seed/optimizer 复制才可关联 phase。

## 14. Empirical NTK diagnostics

固定 \(n_{NTK}=256\) 个从未训练的 episode，标量输出 Jacobian
\(J_t\in\mathbb R^{n_{NTK}\times P}\)，定义

\[
K_t=P^{-1}J_tJ_t^\top,\qquad
D_{NTK}(t)=\frac{\|K_t-K_0\|_F}{\|K_0\|_F+10^{-12}},
\]

\[
A_{NTK}(t)=\frac{\langle K_t,K_0\rangle_F}
{\|K_t\|_F\|K_0\|_F+10^{-12}},\qquad
r_{eff}(K_t)=\frac{\operatorname{tr}(K_t)^2}{\operatorname{tr}(K_t^2)}. \tag{21}
\]

同时按参数组 `E`, `QK raw factors`, `VO raw factors`, `FFN`, `readout` 计算
\(K_t^{(g)}=J_t^{(g)}J_t^{(g)\top}/P_g\)。组合矩阵 \(Q^\top K\) 不是独立参数组，不得
用它代替 raw-parameter Jacobian。

在相同 probe 上记录 initialization linearization

\[
e_{lin}(t)=
\frac{\|f_{\theta_t}-[f_{\theta_0}+J_0(\theta_t-\theta_0)]\|_2}
{\|f_{\theta_t}-f_{\theta_0}\|_2+10^{-12}}. \tag{22}
\]

若某 checkpoint 的 \(D_{NTK}\) 超过该 seed trajectory median 加 5 MAD，先依次检查
gradient norm、parameter norm、finite precision、checkpoint replay 与 probe identity。
只有 spike 在至少 5 个 seed、第二 optimizer、加倍 precision/probe 下重现，才进入
现象报告；否则归类为 numerical/local optimization diagnostic，不能升级为 open problem。

## 15. 输出表、图与审计字段

分析实现必须生成以下不覆盖的版本化产物：

```text
results/<study_id>/seed_metrics.parquet
results/<study_id>/trajectory_metrics.parquet
results/<study_id>/patch_pair_metrics.parquet
results/<study_id>/failures.jsonl
results/<study_id>/analysis_manifest.json
results/<study_id>/analysis_summary.json
figures/<study_id>/*.png
```

`seed_metrics` 一行对应 `(config_hash, seed, checkpoint)`；`patch_pair_metrics` 至少保留
`base_episode_id, donor_episode_id, layer, head, site, direction, raw_delta_output`，使平方、
log ratio 与异质性可重算。manifest 必须包含 git commit、完整 config、依赖版本、设备、
determinism flags、所有阈值、bootstrap seed、scheduled/completed/failed seed id。

每张主图显示所有 seed 点，不只画 bar；trajectory 显示 simultaneous band；补偿图同时
显示 before-floor、opposition rate、finite 与 tangent effect。失败率与 pass-rate 必须和
successful-solution effect 放在同一结果章节。

## 16. Claim status ladder

所有结论必须标记以下最高可达级别；不得跨级：

| Level | 名称 | 必要条件 | 允许的表述 |
|---:|---|---|---|
| 0 | implementation-validated | support、replay、Fourier、JVP、serialization tests 全过 | “实现复现了注册恒等式” |
| 1 | descriptive | 有限 seed 的 raw points 与 MC error；未必过 multiplicity/gate | “在这些运行中观察到” |
| 2 | confirmatory finite-model effect | \(R\ge10\) 配对 seed、primary CI/多重校正通过、失败完整报告 | “在注册有限模型/分布中存在统计效应” |
| 3 | optimizer/architecture replicated | 第 11 节相应 replication gate 通过 | “效应不依赖单一优化器/该架构选择” |
| 4 | localized internal causal mechanism | on-support donor、明确 SCM patch、practical floor、tangent+finite 一致、模块 family 校正通过 | “干预定位到指定层/模块的抑制路径” |
| 5 | empirical dynamical law | gradient-flow gate、scaling grid、simultaneous trajectory band、独立确认 seeds 通过 | “在注册缩放范围内符合该动力学规律” |
| 6 | mathematical theorem | 假设、量词、误差界与证明独立于实验；实验只检验假设范围 | “在所列假设下证明” |

低风险与式 (1) 可以证明复合 value kernel 接近 target coefficient，但它不推出某个 head
attention 接近 1，也不推出 QK、OV 或 FFN 中任一个 raw factor 是唯一机制。实验的核心
任务正是：在功能由 (1)、(16)、(18) 匹配后，检验训练是否系统性选择不同的
\((E,Q^\top K,OV,w)\) 分解，并用第 5--6 节把 on-support distractor cross-talk 的抑制
定位到可复查的层和模块。
