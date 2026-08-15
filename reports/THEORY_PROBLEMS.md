# 两个主问题的严格数学定义、定理目标与反例目标

## 0. 先把研究对象说成一句没有歧义的话

我们研究一个**固定、有限、标准 softmax 的 causal Transformer**，它每次看到 $m$ 张
“concept--random-bit”记忆卡和一个 query concept；答案是与 query 同 concept 的卡片上的
随机 bit。所有参数从随机初始化联合训练。我们问：

1. 梯度流怎样选择出“把正确随机 bit 路由到 query 输出”的复合计算？
2. 当 concept 数多于隐藏维数时，learned concept code 的几何重叠会不会造成真实
   cross-talk；如果会，是 QK、OV 还是 FFN 在下游消掉它？

问题 1 的**函数层必要条件**可以精确证明；参数选择仍开放。问题 2 的宽泛版本已经被已有
文献部分解决；这里保留的是 fresh episodic value、联合训练、finite causal localization
这一更窄版本。

---

## 1. 两个问题共用的完整概率模型

### 1.1 数据分布

固定整数

\[
C\ge m\ge2,\qquad d\ge1,\qquad L\ge1,\qquad H\mid d ,
\]

其中 $C$ 是 concept vocabulary，$m$ 是 memory slots，$d$ 是 residual width，$L$ 是层数，
$H$ 是 head 数，$d_h=d/H$。

一次 episode 的外生随机变量为

\[
(c_1,\ldots,c_m)
\sim\operatorname{Unif}
\{(a_1,\ldots,a_m)\in[C]^m:a_i\ne a_j\text{ for }i\ne j\},
\]

\[
v_i\overset{\mathrm{iid}}{\sim}\operatorname{Unif}\{-1,+1\},\qquad
J\sim\operatorname{Unif}[m],
\]

\[
q=c_J,\qquad Y=v_J .
\tag{1}
\]

$v_{1:m}$、$J$ 与 $c_{1:m}$ 相互独立。训练和测试都每次重新抽 $v_i$，所以

\[
\Pr(Y=1\mid q)=\Pr(Y=-1\mid q)=\frac12 .
\]

因此 concept embedding 不能把答案背下来。网络必须根据当前 episode 找到位置 $J$，再
读取该位置当前才出现的 bit $v_J$。

### 给大一学生的直觉

每道题有四张左右的卡片；卡片左边是名字，右边是现场掷硬币得到的 $\pm1$。最后问“名字
$q$ 的硬币是什么”。名字每次可以重复出现在不同题，但硬币每次重掷。只记住名字没有用。

---

## 2. 固定有限 Transformer

序列长度 $T=m+1$。前 $m$ 个位置是 memory，最后位置是 query。所有下面出现的向量和矩阵
都是 learned parameters 的一部分。

### 2.1 输入表示

\[
x_i^0=E_{c_i}+v_i e_v+e_{\rm mem}+p_i,\quad 1\le i\le m,
\]

\[
x_T^0=E_q+e_{\rm qry}+p_T,
\tag{2}
\]

其中 $E\in\mathbb R^{C\times d}$ 是 concept dictionary，$e_v,e_{\rm mem},
e_{\rm qry}\in\mathbb R^d$，$p_i\in\mathbb R^d$。

### 2.2 一层 multi-head causal attention

令 $z_i^\ell=\operatorname{RMSNorm}_\ell(x_i^\ell)$。对层 $\ell$、head $h$，

\[
Q_{\ell h},K_{\ell h},V_{\ell h}\in\mathbb R^{d_h\times d},
\qquad O_{\ell h}\in\mathbb R^{d\times d_h}.
\]

定义两个 gauge-invariant 复合矩阵

\[
B_{\ell h}=Q_{\ell h}^{\top}K_{\ell h}\in\mathbb R^{d\times d},
\qquad
C_{\ell h}=O_{\ell h}V_{\ell h}\in\mathbb R^{d\times d}.
\tag{3}
\]

位置 $t$ 只能看 $i\le t$：

\[
s_{ti}^{\ell h}
=\frac{\beta}{\sqrt{d_h}}\,(z_t^\ell)^\top B_{\ell h}z_i^\ell,
\qquad
a_{ti}^{\ell h}
=\frac{e^{s_{ti}^{\ell h}}}
{\sum_{r\le t}e^{s_{tr}^{\ell h}}},
\tag{4}
\]

\[
u_t^{\ell h}=\sum_{i\le t}a_{ti}^{\ell h}z_i^\ell,
\qquad
x_t^{\ell,\mathrm{att}}
=x_t^\ell+\frac1{\sqrt L}\sum_{h=1}^H C_{\ell h}u_t^{\ell h}.
\tag{5}
\]

“causal”在这里首先是一个**图结构条件**：

\[
a_{ti}^{\ell h}=0\qquad\text{对所有 }i>t.
\tag{6}
\]

它不等于“attention 是因果解释”；内部因果效应还必须用下面的结构干预定义。

### 2.3 可选 FFN、readout 与输出

若 FFN width 为 $r$，

\[
F_\ell(z)=U_{\ell,2}\,
\operatorname{GELU}(U_{\ell,1}z+b_{\ell,1})+b_{\ell,2},
\]

\[
x_t^{\ell+1}
=x_t^{\ell,\mathrm{att}}
+\frac1{\sqrt L}
F_\ell(\operatorname{RMSNorm}_{\ell,\rm ffn}(x_t^{\ell,\mathrm{att}})).
\tag{7}
\]

attention-only control 取 $F_\ell\equiv0$。最终标量预测为

\[
f_\theta(X)=w^\top\operatorname{RMSNorm}_{\rm out}(x_T^L)+b.
\tag{8}
\]

模型实现中 RMSNorm 的 coordinatewise gains、所有 bias、type/position/value vectors 也
都训练；它们没有在符号 $\theta$ 中省略。

### 2.4 loss 与训练时间

理论对象是 population square risk

\[
\mathcal R(\theta)
=\frac12\,\mathbb E_{X,Y}\bigl(f_\theta(X)-Y\bigr)^2
\tag{9}
\]

及 population gradient flow

\[
\frac{d\theta_s}{ds}=-\nabla_\theta\mathcal R(\theta_s).
\tag{10}
\]

$s$ 是**训练时间**；$\ell$ 是**层深**。Perspective 的 clustering 主要研究固定
$\theta_s$ 后随 $\ell$ 变化的表示；这里首先固定 $C,d,L,H,m$，研究 $\theta_s$ 随 $s$
改变。

实验以 fresh online mini-batch 估计 (10)。AdamW（weight decay 为 0）和 momentum-SGD
是两种离散优化器 robustness checks；不能把它们与连续 GF 声称为完全相同。

---

## 3. “因果”在本项目中的数学含义

给定训练后固定参数，网络中每个 score、attention、mixture、residual、prediction 都是
外生变量

\[
U=(c_{1:m},v_{1:m},J)
\]

的确定性结构方程。

### 3.1 外生 value 干预

固定 $(c_{1:m},v_{-i},J)$，只替换结构方程中的 $v_i$：

\[
\kappa_i(c,v_{-i},J)
=\frac12\left[
f_\theta(\operatorname{do}(v_i=+1))
-f_\theta(\operatorname{do}(v_i=-1))
\right].
\tag{11}
\]

这是 slot $i$ 的 end-to-end causal finite difference。

### 3.2 内部 key-path 干预

把每层、每 head 的 query-to-slot-$i$ logit 改成 $-\infty$ 并重算全部后代。先定义每个
episode、每个 memory slot 的 signed blocked-edge effect：

\[
\delta_i(U)
=
Y\left\{
f_\theta(X)-
f_\theta\bigl(
\operatorname{do}(s_{Ti}^{\ell h}=-\infty,\ \forall\ell,h)
\bigr)
\right\}.
\tag{12}
\]

直接 key-path selectivity 是

\[
S_{\rm key}
=\mathbb E_U\left[
\delta_J(U)-\frac1{m-1}\sum_{i\ne J}\delta_i(U)
\right].
\tag{13}
\]

$\mathbb E\delta_J$ 是指定路径效应，不是 queried value 的总因果效应：在多层网络中，
target value
可以先流到另一个 memory token，再间接流到 query。

### 3.3 不是因果效应的量

embedding cosine、JVP、gradient、linear probe、attention correlation、loss-landscape
slice 都是描述或局部机械诊断。除非明确替换一个结构方程并重算后代，本文不把它叫
causal effect。

---

# 主问题 A：复合 routing kernel 的训练选择理论

## 4. 函数层问题：低风险到底强迫了什么

固定 $(c_{1:m},J)$，把完整网络看成 Boolean cube 上的函数
$f(v_1,\ldots,v_m)$。定义 Walsh 系数

\[
\widehat f_S(c,J)
=2^{-m}\sum_{v\in\{-1,+1\}^m}
f(c,v,J)\prod_{i\in S}v_i,\qquad S\subseteq[m].
\tag{14}
\]

平均 (11) 的 $v_{-i}$ 后，

\[
\kappa_i(c,J)
=\mathbb E_{v_{-i}}\kappa_i(c,v_{-i},J)
=\widehat f_{\{i\}}(c,J).
\tag{15}
\]

### 4.1 已经可以完整证明的命题

Walsh characters 是均匀 Boolean cube 上的正交基，因此 Parseval 给出

\[
\begin{aligned}
2\mathcal R(\theta)
=\mathbb E_{c,J}\Big[
&(\widehat f_{\{J\}}-1)^2
+\sum_{i\ne J}\widehat f_{\{i\}}^2\\
&+\widehat f_{\varnothing}^2
+\sum_{|S|\ge2}\widehat f_S^2
\Big].
\end{aligned}
\tag{16}
\]

这是**恒等式**，不依赖宽度、深度或优化算法。于是若
$\mathcal R(\theta)\le\varepsilon$，

\[
\mathbb E(\kappa_J-1)^2\le2\varepsilon,\qquad
\mathbb E\sum_{i\ne J}\kappa_i^2\le2\varepsilon,
\tag{17}
\]

且 bias leakage 与 higher-order leakage 也各自不超过 $2\varepsilon$。

定义 registered target flip statistic

\[
\Xi_{\rm value}
=\frac12\mathbb E\left[
Y\{f_\theta(X)-f_\theta(\operatorname{do}(v_J=-v_J))\}
\right].
\tag{18}
\]

变量替换立即给出

\[
\Xi_{\rm value}=\mathbb E_{c,J}\widehat f_{\{J\}},
\qquad
|\Xi_{\rm value}-1|\le\sqrt{2\varepsilon}.
\tag{19}
\]

所以“低 population risk 强迫任务相关的**函数级 causal value routing**”已经是可证明的
命题。它不能被包装成尚未解决的主要定理。

### 4.2 它没有证明什么

(16)--(19) 没有证明：

- attention mass 必须集中到 $J$；
- 直接 query-to-$J$ edge 是唯一或必要路径；
- $Q$ 与 $K$ 的某个 raw factor 被唯一选中；
- OV 或 FFN 中哪一个实现了 $\kappa_J$；
- 训练从随机初始化一定到达低风险区域。

一个简单非可识别性来源是 OV/readout amplification：很小的 target attention 可以乘很大
的 value gain，仍令 $\kappa_J=1$。另一个来源是多层 indirect path。第三个来源是 gauge：

\[
Q\mapsto AQ,\quad K\mapsto A^{-\top}K
\]

保持 $Q^\top K$ 不变；raw factor 并非函数上可识别。

---

## 5. 参数层问题：联合梯度流怎样选择实现

令

\[
G_{B_{\ell h}}=\frac{\partial\mathcal R}{\partial B_{\ell h}},
\qquad
G_{C_{\ell h}}=\frac{\partial\mathcal R}{\partial C_{\ell h}}.
\]

虽然 full state dynamics 不闭合在 $B,C$ 上，factorization 本身给出以下**精确** GF
恒等式。

### 5.1 QK 复合矩阵及 balance Grams

对一个 head 省略 $\ell,h$，令

\[
B=Q^\top K,\qquad S_Q=Q^\top Q,\qquad S_K=K^\top K.
\]

则

\[
\dot B=-G_BS_K-S_QG_B,
\tag{20}
\]

\[
\dot S_Q=-G_BB^\top-BG_B^\top,
\qquad
\dot S_K=-G_B^\top B-B^\top G_B.
\tag{21}
\]

这说明同一个 composite score form 的训练速度受 raw-factor balance 控制。直接把 $B$
当成独立欧氏参数训练，会得到不同动力学。

### 5.2 OV 复合矩阵及 balance Grams

令

\[
C=OV,\qquad S_O=OO^\top,\qquad S_V=V^\top V.
\]

则

\[
\dot C=-G_CS_V-S_OG_C,
\tag{22}
\]

\[
\dot S_O=-G_CC^\top-CG_C^\top,
\qquad
\dot S_V=-G_C^\top C-C^\top G_C.
\tag{23}
\]

### 5.3 embedding 与 readout

令 $r_\theta(X)=\operatorname{RMSNorm}_{\rm out}(x_T^L)$。精确地，

\[
\dot w=-\mathbb E[(f_\theta-Y)r_\theta(X)],
\qquad
\dot b=-\mathbb E[f_\theta-Y],
\tag{24}
\]

\[
\dot E_c
=-\mathbb E\left[
(f_\theta-Y)\nabla_{E_c}f_\theta(X)
\right].
\tag{25}
\]

$e_v$、type/position vectors、RMS gains 和 FFN 参数有相同链式法则形式。

(20)--(25) 还不是低维 closure，因为 $G_B,G_C$ 与
$\nabla_{E_c}f$ 依赖所有层的状态分布。**真正开放的训练问题**是：在数据 law (1) 的
置换对称性和小随机初始化下，哪些任务相关投影足以近似或精确关闭这些梯度？

---

## 6. 需要关闭和解释的 order parameters

每一个量都以训练时间 $s$ 为自变量：

### 6.1 功能量

\[
K_{\rm target}(s)=\mathbb E\kappa_J(s),\qquad
L_{\rm distract}(s)=\mathbb E\sum_{i\ne J}\kappa_i(s)^2,
\]

\[
L_{\rm high}(s)=\mathbb E\sum_{|S|\ge2}\widehat f_S(s)^2.
\tag{26}
\]

### 6.2 attention routing 量

\[
\gamma_{\ell h}(s)
=\mathbb E\left[
s_{TJ}^{\ell h}
-\frac1{m-1}\sum_{i\ne J}s_{Ti}^{\ell h}
\right],
\]

\[
\alpha_{\ell h}(s)=\mathbb E\,a_{TJ}^{\ell h},\qquad
S_{\rm key}(s)\ \text{由 (13) 定义}.
\tag{27}
\]

### 6.3 concept geometry

令 $u_c=E_c/\|E_c\|$，

\[
g_E(s)
=\frac1C\sum_c\|E_c\|^2
-\frac1{C(C-1)}\sum_{c\ne c'}E_c^\top E_{c'},
\tag{28}
\]

并记录 coherence、Gram spectrum 和

\[
r_{\rm eff}(E)
=\frac{(\sum_j\sigma_j^2)^2}{\sum_j\sigma_j^4}.
\tag{29}
\]

### 6.4 factorization、feature learning 与表示深度

记录 $B_{\ell h},C_{\ell h}$ 的 spectrum、$S_Q-S_K$ 等 balance modes，及

\[
\Delta_{\rm NTK}(s)
=\frac{\|K_{\theta_s}-K_{\theta_0}\|_F}
{\|K_{\theta_0}\|_F+10^{-12}}.
\tag{30}
\]

逐层表示同时记录

\[
\rho_{\rm global}^{\ell}(s),\qquad
\Delta\rho^\ell(s)
=\mathbb E\!\left[
\cos(x_T^\ell,x_J^\ell)
-\frac1{m-1}\sum_{i\ne J}\cos(x_T^\ell,x_i^\ell)
\right].
\tag{31}
\]

(30) 区分 lazy/NTK-like 与 feature-learning；(31) 区分 global clustering 与
target-selective geometry。

---

## 7. 主问题 A 的可证命题与反例目标

### A0：函数级强迫定理

**状态：已解决。** 完整证明就是 (14)--(19)。实验中的 exhaustive Walsh cube 是对该
恒等式和实现的数值核验。

### A1：早期复合模式 closure

给定初始化尺度 $\delta$ 与明确的随机初始化 law，计算 (20)--(25) 在 $s=0$ 的首个非零
Taylor 项。目标不是写“梯度会学习 attention”，而是证明存在显式向量

\[
z(s)=
(g_E,\gamma_{\ell h},\alpha_{\ell h},
\text{value/readout alignments},
\text{factor-balance modes})
\]

及显式 $F$，使某个非平凡时间窗内

\[
\dot z(s)=F_{C,m,d,L,H,\beta}(z(s))+\mathcal E(s),
\qquad
\sup_{s\le s_0}\|\mathcal E(s)\|\le\eta(\delta,d),
\tag{32}
\]

并给出 $\eta$ 的确定界。可走两条严格路线：

1. 在精确对称初始化子流形上证明 finite-dimensional exact closure；
2. 对一般 iid 随机初始化证明高概率 approximate closure。

若 naive $z$ 不闭合，反例同样有价值：构造两组参数具有相同已注册 $z$，但
$\dot z$ 不同，从而证明必须加入哪个 higher-order correlator。

### A2：从函数 routing 到直接 key routing 的最小条件

一般网络中该结论为假。目标应写成二选一：

- **定理路线**：在单层、无 indirect memory-to-memory path、scores 对 values 不敏感、
  OV/readout gain 有界且 value channels 单调非负的可识别子类中，从
  $\mathcal R\le\varepsilon$ 推出显式的 target-attention 或 $S_{\rm key}$ 下界；
- **反例路线**：在尽可能小的标准 block 中，构造
  $\mathcal R=0$ 但 target attention 不集中，或 $S_{\rm key}$ 很小却
  $\Xi_{\rm value}=1$ 的参数族。

无论哪条关闭，都比未经条件的“低风险强迫 attention routing”正确。

### A3：训练选择和架构分岔

在达到相同低风险的条件下，证明或反驳

\[
(C/d,H,\mathrm{FFN\ width},L)
\longmapsto
\{r_{\rm eff}(E),\operatorname{rank}_{\rm eff}(B),
\operatorname{rank}_{\rm eff}(C),\text{head specialization}\}
\tag{33}
\]

存在稳定的 selection/bifurcation law。实验已给出 $H\times(C/d)$ 的 paired interaction；
理论要解释的是 factorized GF 为什么选择不同内部实现，而不是再次证明网络能拟合。

### 大一学生直觉

A0 说：如果答题几乎全对，输出当然必须跟正确硬币一起翻转。这是“函数事实”。A1--A3
才问：网络内部是先把名字排好、先学会搬运硬币，还是先把最后读数器对准；多个同样能答对
的内部方案里，梯度下降选哪一个。这是“学习机制”。

---

# 主问题 B：learned compressed dictionary 的下游补偿

## 8. 先严格收窄 novelty

[Geometric Factual Recall](https://arxiv.org/abs/2605.12426) 已经证明：在固定事实和共享
attribute 集合中，低维 subject embeddings 可以叠加属性，小 ReLU MLP 可作
relation-conditioned selector；GD 也在实验中找到这种结构。因此以下宽泛命题不再开放：

> “低维 embedding 能不能 superpose 多个属性，MLP 能不能下游选择？”

本项目剩余问题必须同时包含：

1. 每个 episode 的 values 都重新随机，不能存进 $E$；
2. $E,QK,OV,\mathrm{FFN},w$ 联合训练；
3. 先证明 compressed geometry 产生**功能 cross-talk**；
4. 用 on-support finite intervention 定位 cross-talk 在哪个模块被缩小；
5. 描述这一机制怎样随训练时间形成。

---

## 9. “superposition”与“cross-talk”的可检验定义

### 9.1 compressed dictionary 不是 activation superposition

当 $C>d$，concept rows 不可能两两正交。定义

\[
\mu(E)=\max_{c\ne c'}|u_c^\top u_{c'}|,
\qquad
r_{\rm eff}(E)\ \text{由 (29) 定义},
\tag{34}
\]

\[
D_c
=\frac{\|E_c\|^2}
{\sum_{c'}(u_c^\top E_{c'})^2},
\qquad
\sum_cD_c\le\operatorname{rank}(E)\le d.
\tag{35}
\]

这里假设 $E_c\ne0$；零 row 约定 $D_c=0$。不等式不是经验猜测：令
$G=E^\top E$、leverage score $\tau_c=E_cG^+E_c^\top$，Cauchy--Schwarz 给
$D_c\le\tau_c$，而 $\sum_c\tau_c=\operatorname{rank}(E)$。

(34)--(35) 只测 compressed dictionary geometry。若要声称 activation superposition，
还必须在同一个 hidden state 上定义多个稀疏 feature variables，并证明它们可被独立 decoder
恢复；当前主实验没有把 rank 单独叫作 activation superposition。

### 9.2 on-support distractor swap

要求 $C>m$。从 base episode $X$ 选 $K\ne J$，再从当前 memory 未出现的 concepts 中均匀
选 $c_{\rm new}$，构造

\[
X'=\operatorname{swap}(X;c_K\leftarrow c_{\rm new}).
\tag{36}
\]

$v_{1:m},J,q,Y$ 全部不变，且 $X,X'$ 都在原数据分布支持上。因此理想 retrieval 函数应
满足 $f^\star(X')=f^\star(X)$。

最终函数 cross-talk 为

\[
I_{\rm out}
=\mathbb E\bigl(f_\theta(X')-f_\theta(X)\bigr)^2.
\tag{37}
\]

对注册内部节点 $Z_r$，用 donor $X'$ 替换 recipient $X$ 的该节点并重算后代：

\[
I_r
=\mathbb E\left[
\left\{
f_\theta(X;\operatorname{do}(Z_r=Z_r(X')))
-f_\theta(X)
\right\}^2
\right].
\tag{38}
\]

只有某个 upstream $I_r$ 实际非零，而后续模块的 finite effect 显著变小，才有
“downstream compensation”可识别。

---

## 10. QK、OV、FFN 的精确局部分解

### 10.1 QK 的 exact finite route/content/interaction chord

对同一层/head 的 query row，base 为 $(a_i,z_i)$，donor 为 $(a_i',z_i')$，

\[
\delta m_{\rm route}=\sum_i(a_i'-a_i)z_i,
\qquad
\delta m_{\rm content}=\sum_i a_i(z_i'-z_i),
\]

\[
\delta m_{\rm interaction}
=\sum_i(a_i'-a_i)(z_i'-z_i),
\tag{39}
\]

\[
m'-m
=\delta m_{\rm route}
+\delta m_{\rm content}
+\delta m_{\rm interaction}.
\tag{40}
\]

(40) 是逐 episode 精确恒等式。乘 $C_{\ell h}$ 并取 attention residual 后的 downstream
adjoint $r$，

\[
t_p=L^{-1/2}r^\top C_{\ell h}\delta m_p,
\quad
p\in\{\mathrm{route,content,interaction}\}.
\tag{41}
\]

预注册 QK 局部 suppression contrast 是

\[
C_{QK}
=\mathbb E\log
\frac{(t_{\rm content}+t_{\rm interaction})^2+\epsilon_0}
{(t_{\rm route}+t_{\rm content}+t_{\rm interaction})^2+\epsilon_0}.
\tag{42}
\]

$C_{QK}>0$ 才表示 route 减小 content+interaction；还必须由 finite hybrid 输出同方向
验证。仅有

\[
\Pr[t_{\rm route}(t_{\rm content}+t_{\rm interaction})<0]>\tfrac12
\]

不够，因为 route 虽常反向，却可能绝对值过大，最终放大扰动。

### 10.2 OV：线性方向选择，而不是相邻 patch 差

对实际 swap direction $\delta m$，

\[
g_{\rm swap}
=\frac{\|C_{\ell h}\delta m\|^2}
{\|\delta m\|^2+\epsilon_0},
\qquad
g_{\rm iso}=\frac{\|C_{\ell h}\|_F^2}{d},
\]

\[
A_{OV}
=\mathbb E\log\frac{g_{\rm iso}+\epsilon_0}
{g_{\rm swap}+\epsilon_0}.
\tag{43}
\]

若 $\Delta A_{OV}=A_{OV}^{\rm final}-A_{OV}^{\rm init}>0$，说明训练让 OV 对实际 cross-talk
方向的 gain 相对各向同性平均变小。它仍是方向选择性，需 finite output test 才能升级为
因果补偿。

同一 donor 的完整 pre-OV mixture patch 与 post-OV update patch 必然满足

\[
f(\operatorname{do}(m=m'))
=f(\operatorname{do}(Cm=Cm')).
\tag{44}
\]

所以二者之差应是数值零；把它当 OV compensation 是 instrumentation error。

### 10.3 FFN residual 的 skip/branch cancellation

令同层 attention 后 base/donor query states 为 $x,x'$，

\[
\delta x_{\rm skip}=x'-x,
\]

\[
\delta x_{\rm ffn}
=F_\ell(\operatorname{RMSNorm}(x'))
-F_\ell(\operatorname{RMSNorm}(x)),
\]

\[
\delta x_{\rm post}
=\delta x_{\rm skip}+\delta x_{\rm ffn}.
\tag{45}
\]

在 FFN residual 后取 adjoint $r$：

\[
t_{\rm skip}=r^\top\delta x_{\rm skip},
\qquad
t_{\rm ffn}=r^\top\delta x_{\rm ffn},
\]

\[
C_{FFN}
=\mathbb E\log
\frac{t_{\rm skip}^2+\epsilon_0}
{(t_{\rm skip}+t_{\rm ffn})^2+\epsilon_0}.
\tag{46}
\]

必须先通过 practical floor

\[
\mathbb E\,t_{\rm skip}^2\ge10^{-4}\operatorname{Var}(Y)=10^{-4};
\tag{47}
\]

否则极小分母可以制造虚假的大 log ratio。

令 $z=x+F_\ell(\operatorname{RMSNorm}(x))$ 是 base episode 在该 FFN residual 后的状态。
finite suffix map $G_\ell$ 给

\[
p_{\rm skip}=G_\ell(z+\delta x_{\rm skip})-G_\ell(z),
\]

\[
p_{\rm ffn}=G_\ell(z+\delta x_{\rm ffn})-G_\ell(z),
\]

\[
p_{\rm joint}
=G_\ell(z+\delta x_{\rm skip}+\delta x_{\rm ffn})-G_\ell(z).
\tag{48}
\]

候选 FFN compensator 必须在 (46) 与 (48) 上同方向，而不是只看 tangent。

---

## 11. “补偿器”的确认标准

对任一模块 $M$，只有同时满足以下条件才使用 compensator：

1. (36) 的两个 endpoints 都在支持上，且 label 不变；
2. upstream cross-talk 通过事先固定的 practical floor；
3. 令 $C_M\in\{C_{QK},A_{OV},C_{FFN}\}$；seed-level $C_M$ 的 simultaneous 95% CI
   在抑制方向；
4. finite output contrast 与 tangent/local contrast 同号，且达到预注册 pair consistency；
5. 第二优化器或架构控制复制；
6. accuracy、$\Xi_{\rm value}$ 与功能门槛匹配；
7. layer/head family 做 multiplicity correction，head 不是独立样本。

训练 seed 是统计独立单位；512 或 8192 个 held-out episodes 只减小每个 seed 内 Monte Carlo
误差，不把样本量变成 512 或 8192。

---

## 12. 主问题 B 的可证命题与反例目标

### B0：几何重叠是否真的产生 upstream functional cross-talk

要证明的不是 $C>d$，而是某个已注册内部 site $r$ 满足

\[
\mathbb E I_r^{\rm final}\ge\tau>0
\quad\text{同时}\quad
\mathbb E I_{\rm out}^{\rm final}\le\varepsilon.
\tag{49}
\]

若所有 upstream $I_r$ 都接近 0，则网络可能在进入共享 residual stream 前已经做了无串扰
的 query-conditioned hashing；这会反驳“压缩几何必然需要下游补偿”。

### B1：训练诱导的 module localization

在明确的 load regime $\rho=C/d>1$、初始化 law 和 architecture 下，证明存在至少一个
$M\in\{QK,OV,\mathrm{FFN}\}$ 使

\[
\Delta C_M=C_M^{\rm final}-C_M^{\rm init}>0
\tag{50}
\]

并满足第 11 节 finite criteria；或者给出反例，证明低风险与 compressed $E$ 可以在所有
这些 $C_M\le0$ 时实现。

### B2：哪个参数比决定机制分岔

寻找显式 phase diagram

\[
(C/d,H,L,r_{\rm ffn}/d,\beta,\delta_{\rm init})
\longmapsto
\{\text{QK route},\text{OV filter},\text{FFN cancellation},
\text{no compensation needed}\}.
\tag{51}
\]

可先在 $L=1,H=1$ 上证明，再增加 head 和深度。若同一宏观 order parameters 对应不同
localization，反例应指出缺少哪一个 order parameter，而不是把未闭合称为噪声。

### B3：与 compressed sensing 的具体连接

为了与 compressed sensing 做**可检验但不冒充原网络恒等式**的对照，先定义一个线性汇总
surrogate。把一次 episode 的稀疏 value memory 写成 $x\in\mathbb R^C$：

\[
x_c=
\begin{cases}
v_i,&c=c_i,\\
0,&c\notin\{c_1,\ldots,c_m\}.
\end{cases}
\qquad \|x\|_0=m.
\tag{52}
\]

在这个 surrogate 中，learned dictionary 产生压缩 measurement
$h=E^\top x\in\mathbb R^d$。真实 Transformer 保留分开的 memory tokens，并通过 attention
做 query-dependent 汇总，所以 $h=E^\top x$ 不是对 (2)--(7) 的等式；它是需要与真实
attention decoder 对照的最小 compressed-sensing 基线。任务不是恢复整个 $x$，只需在
给定 query $q$ 后恢复坐标 $x_q$：

\[
\mathcal D_\theta(h,q)\approx x_q.
\tag{53}
\]

经典 compressed sensing 用 RIP/coherence 保证统一稀疏恢复；这里 decoder
$\mathcal D_\theta$ 是由 QK/OV/FFN 实现的 query-conditioned nonlinear map，而且 measurement
matrix $E$ 也由梯度流学习。具体理论目标是：

- 给 (53) 的充分 coherence/RIP-like 条件与必要容量下界；
- 证明或反驳 population GF 会把 $E_s$ 推向这些条件；
- 解释多 head 是增加 measurement channels、降低 collision，还是仅改变 factorized
  optimization。

这才是 Hanin Superposition Q2 在本任务中的可证版本。

### 大一学生直觉

把很多名字压进较少维度，就像把很多人的声音混到少数麦克风里。几何重叠只说明声音会
叠在一起；它不证明听错了。on-support swap 是把一个不相关说话人换掉，看中间信号和最终
答案是否变化。如果中间明显变、最后几乎不变，才有东西在下游消噪。然后用 finite
intervention 判断消噪器究竟是“先选对人”的 QK、“只放大有用方向”的 OV，还是“加一个
反相信号”的 FFN。

---

## 13. 当前实验对两个问题已经说了什么

### 13.1 对主问题 A

- AdamW 主网格 96 runs、momentum-SGD 复制网格 96 runs；绝大多数 seed 达到近零风险；
- queried-value flip 与 Walsh target coefficient 接近 1，支持 (16)--(19) 的功能级
  composite routing；
- attention target mass、direct-key intervention、表示几何和 Walsh kernel 分开记录，
  没有从 attention 图跳到因果结论；
- 一个延长训练仍未逃出的 SGD seed 先按已有 plateau/local-solution 文献处理，不命名为
  新 open problem；
- 参数级早期 closure、收敛和 factor selection 仍未证明。

### 13.2 对主问题 B

- learned $E$ 的 effective rank 随 load、head、FFN 改变；这证明表示选择不同，不证明
  activation superposition；
- natural distractor swap 的输出 MSE 多数很小，说明最终函数大致保持任务不变量；
- **QK 特定 suppression 命题被当前聚合实验反对**：两个优化器的所有 cell 中，
  (42) 的终点和 init-to-final 方向均为负，route 净放大而不是缩小 content chord；
- 当前 OV 指标主要是 target-vs-distractor gain，方向一致，是候选，但还不是 (43) 的
  isotropic-vs-swap finite compensation；
- FFN tangent cancellation 只在部分 cell/优化器复制；在 practical floor 与 finite
  validation 完成前，不能确认；
- 所以当前“已确认 compensator”数量是 **0**。这不是失败：它排除了一个宽泛 QK 故事，
  并把下一步集中到 OV direction filtering、FFN finite residual cancellation，或“上游根本
  没有足够 cross-talk”这一反例。

---

## 14. 与 Perspective clustering 的最终关系

三个时间/对象必须始终分开：

\[
\underbrace{s}_{\text{训练改变参数}}
\quad\longrightarrow\quad
\underbrace{\theta_s}_{\text{interaction kernel}}
\quad\longrightarrow\quad
\underbrace{x_i^\ell(s)}_{\text{表示随层深变化}}.
\tag{54}
\]

Perspective 主要固定 $\theta_s$，研究 $\ell$ 的粒子动力学及 long-time clustering。主问题
A 研究 $s$ 怎样选择 $B_{\ell h}(s),C_{\ell h}(s),E(s)$，从而产生 target-specific
routing。主问题 B 研究同一选择怎样在有限层里容纳压缩表示并控制 distractor cross-talk。

因此两主问题不是 Perspective clustering 的同义改写，而是它缺失的训练层：

- global clustering 由 $\rho_{\rm global}$ 描述；
- task-selective geometry 由 $\Delta\rho$ 描述；
- causal retrieval 由 $\kappa_J,\Xi_{\rm value},S_{\rm key}$ 描述；
- downstream compensation 由 on-support finite module contrasts 描述。

只有联合画出这些量随 $(s,\ell)$ 的轨迹，才可能回答“训练出来的 interaction kernel 为何
在有限深度有用，而不是最终把所有 token 无差别压成一个 cluster”。

---

## 15. 当前最值得写成论文的两个准确标题

### 首选主问题

**Training Selection of Composite Causal Routing in a Finite Softmax Transformer**

最小论文闭环：证明 Parseval causal forcing；推导 factorized QK/OV exact GF identities；
在单层/单头可识别子类给 early closure 或明确 closure counterexample；用多 seed 实验验证
order-parameter 轨迹、NTK drift、loss landscape 与架构分岔。

### 第二主问题

**Does Learned Compression Require Downstream Compensation in Episodic Retrieval?**

最小论文闭环：证明或反驳 compressed $E$ 会产生 upstream functional cross-talk；接受当前
QK suppression 反证；用 finite on-support tests 区分 OV filtering、FFN cancellation 与
“no compensation needed”；最后给出至少一个可证明简化模型或容量/反例边界。

这两个标题都比“解释 Transformer clustering”或“解释 superposition”窄，但每个变量、
数据 law、网络、loss、动力学、estimand 与成败条件都已经固定，因此可以真正被完成。
