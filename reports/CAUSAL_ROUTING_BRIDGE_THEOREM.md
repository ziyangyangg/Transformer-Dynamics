# 低风险到 direct-edge routing：一个可识别子类与精确反例

**边界。** 本页只处理单层、value-linear 的 exact-softmax 子类。它不给多层、RMSNorm、
FFN 或一般 residual Transformer 下的结论；这些模型仍需要式 (T21) 的间接路径/抵消预算。

## 1. 模型与干预

一次 episode 的 skeleton 为

\[
\omega=(c_{1:m},J),\qquad m\ge2,
\]

values 独立且均匀：\(v_i\in\{-1,+1\}\)，label 为 \(y=v_J\)。给定 \(\omega\)，第
\(h\) 个 head 的 memory 权重为 \(a_{hi}>0\)，query-self 权重为 \(a_{h0}>0\)，且

\[
a_{h0}+\sum_{i=1}^m a_{hi}=1.
\tag{B1}
\]

这些权重可以依赖 concepts、target 和 slot，但不能依赖 values。self path 在所测标量
value channel 中为零。没有 memory-to-memory relay、FFN、额外 value bypass 或输出
非线性。模型为

\[
f_\omega(v)=b_\omega+
\sum_{h=1}^H g_{h\omega}\sum_{i=1}^m a_{hi\omega}v_i,
\qquad g_{h\omega}\ge0.
\tag{B2}
\]

其中 \(b_\omega\) 不依赖 values；\(g_h\) 是 OV 与 readout 在该 value channel 上的有效
增益。阻断 slot \(i\) 的 score 后，对其余可见位置重新归一化：

\[
a_{hk}^{(-i)}=\frac{a_{hk}}{1-a_{hi}}\ (k\ne i),
\qquad a_{hi}^{(-i)}=0.
\tag{B3}
\]

query-self 也按式 (B3) 重标，但因其 value 为零，不出现在输出和以下公式中。定义

\[
\delta_i=v_J\{f_\omega(v)-f_\omega^{(-i)}(v)\},
\quad
s_{key}(\omega)=\mathbb E_v\!\left[
\delta_J-\frac1{m-1}\sum_{i\ne J}\delta_i\right].
\tag{B4}
\]

总体 \(S_{key}=\mathbb E_\omega s_{key}(\omega)\)，风险

\[
R=\frac12\mathbb E_{\omega,v}(f_\omega(v)-v_J)^2.
\tag{B5}
\]

## 2. 定理

**定理 B1（可识别子类中的 bridge）。** 在式 (B1)--(B5) 下，

\[
S_{key}\ge 1-\sqrt{2R}.
\tag{B6}
\]

因此 \(R\le\varepsilon\) 时，\(S_{key}\ge1-\sqrt{2\varepsilon}\)。

**证明。** 令 target 的函数系数

\[
\kappa_J(\omega)=\sum_h g_h a_{hJ}.
\tag{B7}
\]

由 Rademacher 正交性，逐 skeleton 有

\[
\mathbb E_v\delta_J=\sum_hg_ha_{hJ}=\kappa_J,
\tag{B8}
\]

而对任意 \(i\ne J\)，

\[
\mathbb E_v\delta_i
=-\sum_hg_h\frac{a_{hi}a_{hJ}}{1-a_{hi}}\le0.
\tag{B9}
\]

故 \(s_{key}(\omega)\ge\kappa_J(\omega)\)。另一方面，Parseval 给出

\[
\mathbb E_\omega(\kappa_J-1)^2\le2R.
\tag{B10}
\]

对式 (B10) 用 Cauchy--Schwarz，

\[
S_{key}\ge\mathbb E\kappa_J
\ge1-\mathbb E|\kappa_J-1|
\ge1-\sqrt{2R}.
\]

证毕。这里不需要 attention mass 接近一；结论约束的是 blocking effect。若还要从低风险
推出大 attention margin，必须另加有效增益上界。

## 3. signed-gain 两头反例

删去 \(g_h\ge0\) 后，一层、\(m=2\)、\(H=2\) 已足以使 bridge 失败。对每个 skeleton，
按 `(target, distractor, self)` 排列两头的 softmax 权重：

\[
a_1=\left(\frac15,\frac3{10},\frac12\right),
\qquad
a_2=\left(\frac8{35},\frac25,\frac{13}{35}\right),
\tag{B11}
\]

并取

\[
g_1=35,\qquad g_2=-\frac{105}{4},\qquad b=0.
\tag{B12}
\]

三项都严格为正且各行和为一；令 logits 等于对应概率的对数，exact softmax 就精确产生
式 (B11)。concept-match 与 type features 可以让 target、distractor、self 分别得到这三个
logit。具体地，取相互正交的 concept basis \(e_c\) 与 type basis \(q_0,m_0\)，令
\(z_q=e_c+q_0,z_k=e_{c'}+m_0\)。记三类 logit 为
\(\ell_{hT},\ell_{hD},\ell_{hS}\)，则 composite

\[
B_h=(\ell_{hT}-\ell_{hD})\sum_c e_ce_c^\top
+\ell_{hD}q_0m_0^\top
+\{\ell_{hS}-(\ell_{hT}-\ell_{hD})\}q_0q_0^\top
\tag{B11a}
\]

对 match memory、mismatch memory、query-self 分别给出这三个 logit。value direction
取为与上述 basis 正交且被 \(B_h\) 消去，故 scores 不依赖 values；softmax temperature
与 \(1/\sqrt{d_h}\) 可吸收到 \(B_h\)。

target 与 distractor 的函数系数分别为

\[
35\frac15-\frac{105}{4}\frac8{35}=7-6=1,
\qquad
35\frac3{10}-\frac{105}{4}\frac25
=\frac{21}{2}-\frac{21}{2}=0.
\tag{B13}
\]

所以对全部四个 value assignments，\(f(v)=v_J\)、\(R=0\)，且
\(\Xi_{value}=1\)。但式 (B8)--(B9) 的等式部分不需要 \(g_h\ge0\)，故

\[
\mathbb E_v\delta_J=1,
\tag{B14}
\]

\[
\mathbb E_v\delta_D
=-35\frac{(3/10)(1/5)}{1-3/10}
-\left(-\frac{105}{4}\right)
\frac{(2/5)(8/35)}{1-2/5}
=-3+4=1.
\tag{B15}
\]

因此

\[
S_{key}=1-1=0.
\tag{B16}
\]

反例不是“attention 没学到 target”：两头共同实现了正确函数；失败来自 signed OV/readout
cancellation 令 target 与 distractor 的单边阻断效应相同。它证明一般 bridge 至少要控制
signed cancellation，或把它显式放进式 (T21) 的预算 \(\mathcal I(\theta)\)。

## 4. 尚未关闭的标准网络目标

下一步的全参数反例/定理必须在注册网络中明确给出
\(E,Q,K,V,O,w\)（以及 RMSNorm/FFN，若启用），并二选一：

\[
R=0,\quad\Xi_{value}=1,\quad S_{key}=0,
\tag{B17}
\]

或在可计算的 signed-cancellation、indirect-path 和 gain-conditioning 预算下证明式 (T21)。
式 (B11)--(B16) 只解决可识别子类的边界，不能直接外推到多层标准 Transformer。
