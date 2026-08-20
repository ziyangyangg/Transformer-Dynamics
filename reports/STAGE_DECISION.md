# 阶段判断：trained routing、rank 与稀有 collision

> **证据冻结：2026-08-20。** toy、localization gate 与 Pythia-70M float64 calibration
> 均已审计；本报告不包含未完成实验。
>
> 本文是历史证据判断，不再规定下一阶段主目标。后续研究以
> [RESEARCH_CHARTER.md](RESEARCH_CHARTER.md) 为唯一方向约束；rank--collision 仅保留为候选机制。

## 当时的候选判断（现已降级）

当时证据曾把候选机制收缩为：

\[
\boxed{
\text{factorized population GF}
\longrightarrow\text{rank-constrained margin cover}
\longrightarrow\text{rare routing collisions}
}
\tag{1}
\]

“superposition 的特定下游补偿器”暂不作主问题；没有共同基准的 factorial/Shapley 闭合时，
QK、OV、FFN patch 是重叠 local hybrids，归因在方法上 non-identifiable；这不等于已经
观察到 distributed compensation。

## 对象与已经关闭的部分

数据为无放回 concepts、独立 Rademacher values，\(q=c_J,y=v_J\)。hard toy 固定
\((C,d,m,L,H,d_h)=(32,8,4,2,4,2)\)，联合训练 \(E,Q,K,V,O\) 与 readout：

\[
B_h=Q_h^\top K_h,\quad C_h=O_hV_h,\quad
\dot\theta=-\nabla R,\quad
R=\tfrac12\mathbb E(f_\theta-y)^2.
\tag{2}
\]

固定 skeleton 后的 Walsh--Parseval 恒等式

\[
2R=E_T+L_W,\qquad
L_W=L_D+L_H+L_0,\qquad
|1-\Xi_{value}|\le\sqrt{2R}
\tag{3}
\]

已经完整回答：低风险强迫**函数级 value routing**。这不是创新，也不能推出某个 attention
edge 必要。逐 slot direct effect 是

\[
S_{key}=\mathbb E\!\left[\delta_J-\frac1{m-1}\sum_{i\ne J}\delta_i\right],
\quad
\delta_i=y\{f-f^{(-i)}\}.
\tag{4}
\]

这里 superposition 只指 \(E\in\mathbb R^{C\times d},C>d\) 的 learned compressed
dictionary；它本身不等于功能干扰。功能 cross-talk 由 label-preserving on-manifold swap

\[
I_{swap}=\mathbb E\{f(X^{swap})-f(X)\}^2
\tag{4a}
\]

定义。

无条件的 \(R\to0\Rightarrow S_{key}>0\) 在 exact-softmax attention 模型类中为假；对
注册的多层 RMSNorm 网络仍需全参数反例或条件定理。在单层、value-linear、scores
value-blind、无 bypass 且各 head 有效 value gain 非负的可识别子类中，已证明

\[
S_{key}\ge1-\sqrt{2R}.
\tag{5}
\]

一般网络的 T21 只能是条件目标：

\[
R,L_W\le\varepsilon,\quad
\mathcal I_{indirect}+\mathcal I_{signed}\le\kappa
\Longrightarrow
S_{key}\ge s_0-c(\kappa)\sqrt{\varepsilon},
\tag{T21}
\]

其中 \(L_W\le2R\) 由式 (3) 已知，不是独立证据；真正未知的是可计算的 indirect-path、
signed-cancellation 与 gain-conditioning 预算。

允许 signed head cancellation 后，两头 exact-softmax 反例满足
\(R=0,\Xi_{value}=1,S_{key}=0\)。完整模型、证明与精确分数审计见
[CAUSAL_ROUTING_BRIDGE_THEOREM.md](CAUSAL_ROUTING_BRIDGE_THEOREM.md)。

Perspective 固定 \(B,C\) 后研究 layer/depth dynamics；本项目研究训练怎样产生 \(B_s,C_s\)。
复现中 global clustering 与均匀 attention 同时出现，故

\[
\text{global clustering}\not\Rightarrow\text{selective causal routing}.
\tag{6}
\]

两者的合法桥只有“训练出的 kernel 是否满足固定-kernel 定理的条件”。

## 观察与边界

toy discovery 使用 12 个训练 seeds；seed 是推断单位。所有比较臂 12/12 通过功能门槛。
更长训练与 cosine 使 \(R,L_W,I_{swap}\) 继续下降；预注册 stable-residual gate 失败，因而
排除了“不可消除 plateau”。同 rank direct-composite 没修复 residual；full-rank dense
composite 相对 factorized 使 \(L_W\) 降低 \(8.96\) bits、使高精度
\(I_{swap}\) 降低 \(9.79\) bits（hierarchical simultaneous 95%：
\([-11.33,-8.24]\)）。所以纯 factorization-conditioning 解释被反对；当前分类只是
**rank/function capacity**。\(H=1,d_h=8\) 近零 residual 与此相容，但同时改变了 head 数和
capacity allocation，不能单独证明 per-head rank 因果。

高精度 swap 使用每 checkpoint 524,288 个 on-support pairs。方向保持，但 all-state precision
gate 仍失败；median Gini 为 \(0.976\)，median effective-sample fraction 为 \(0.005\)。因此
“残差集中于少数 episodes”是强线索，“精确 tail law/特定 collision identities”尚未确认。
composite loss plane 排除了 raw-factor gauge flatness 的误读；toy empirical NTK 相对初始
值显著漂移，只说明该 probe 不是 lazy 描述，不能跨 parameterization 比较绝对 NTK。

localization-v2 仅完成 62/72，root gate 失败；10 个拒绝已定位为近零尺度下把
row-wise \((abs>10^{-8})\land(rel>10^{-5})\) 错接成 \(\max(rel)>10^{-5}\) 的 runner bug。
修复已测试但未重放。因此 paired module estimand 为 N/A，当前 confirmed compensator
数量为 **0**；这表示证据未闭合，不表示 QK/OV/FFN effect 为零。

Pythia-70M-v4 通过 8/8 checkpoints 与 32/32 checkpoint-template rows；最大
parallel-residual closure error 为 \(2.04\times10^{-14}\)。但信号短暂且模板异质：最高
accuracy \(0.715\)、最低 risk \(0.379\)、最高 \(S_{key}=0.223\) 同在 step4000 的单一
template；final 四模板 accuracy 仅 \(0.531\)–\(0.562\)，risk \(0.486\)–\(0.510\)，
\(S_{key}\) 为 \(-5.88\times10^{-5}\)–\(0.0569\)。四模板均值的 accuracy/\(S_{key}\)
从 step16000 的 \(0.630/0.104\) 降到 final 的 \(0.543/0.0299\)，但 observation-only
best-head selectivity 同期从 \(0.0483\) 升到 \(0.133\)：direct-edge routing 是
transient/non-monotonic，且不等价于 attention-mass selectivity。因此
“diffuse \(\to\) selective routing \(\to\) sparse collision/downstream reorganization”
**不成立为当前结论**。自然 swap 未保存 episode-level tail，三类 patches 又是重叠
hybrids，故不能检验稀疏 collision 或唯一 module compensator。这里始终只有一条
pretraining trajectory；checkpoint、template、layer、head 都是 repeated measures。

## 研究地图

| 分类 | 当前结论 |
|---|---|
| 已解决 | 式 (3) 的 functional routing；式 (5) 的可识别子类 bridge；factorized \(B,C\) 的精确 GF 恒等式；norm-controlled rank packing 与 softmax/hypergeometric 关系。 |
| 已有近似理论 | fixed-\(QKV\) clustering、low-rank attention bottleneck、简化 token-selection max-margin、linear-attention/非正交 embedding 的早期 GD；均未共同覆盖 learned \(E\)+exact softmax+joint QK/OV+finite slot intervention。 |
| 现象已知、理论缺失 | toy 中 dense 修复而 same-rank direct 不修复；\(I_{swap}\) 重尾；codebook/query geometry、NTK 与 leakage 共变。Pythia 仅见短暂、模板依赖的 routing 信号；localization 未过 gate，不列作现象。 |
| 当前支持的 theorem target | 在 norm/gain/cancellation 有界、无 label bypass、静态或显式 episode-conditioned head assignment 下，把风险、margin、random episode law 与 rank packing 连成有限容量定理；再解释 GF 为何选择均匀 cover 或牺牲少数 pairs。 |

## 主定理包与唯一下一实验

风险--margin 目标（M24）必须带可识别性。若对每个 concept \(c\) 存在固定 head
\(h(c)\)，它在每个 episode 满足
\(g_{h(c)}a_{h(c),J}\ge\lambda\kappa_J\) 且 \(g_{h(c)}\le G\)，定义
\(\Delta_{h(c)}=s_{h(c),J}-\max_{i\ne J}s_{h(c),i}\)，令
\(\alpha_\eta=\lambda(1-\eta)/G>1/2\)，则应先证明

\[
R\le\varepsilon
\Longrightarrow
\Pr\!\left\{\Delta_{h(c)}<
\log\frac{\alpha_\eta}{1-\alpha_\eta}\right\}
\le\frac{2\varepsilon}{\eta^2},
\tag{7}
\]

再与

\[
M_h\le\left(1+\frac{2\beta\rho_Q\rho_K}
{\gamma\sqrt{d_h}}\right)^{d_h},\qquad
p_c^{bad}=1-\frac{\binom{C-1-b_c}{m-1}}{\binom{C-1}{m-1}}
\tag{8}
\]

合成显式风险/容量下界。norm bound、nonnegative/controlled cancellation、无 indirect label
path 和 head-assignment 不能省略；否则式 (7) 有反例。

训练选择目标（M26）的起点是精确恒等式

\[
\dot B_h=-G_{B_h}K_h^\top K_h-Q_h^\top Q_hG_{B_h},\qquad
\dot C_h=-G_{C_h}V_h^\top V_h-O_hO_h^\top G_{C_h}.
\tag{9}
\]

要证明的是式 (9) 何时使 \(\{\gamma_h(c),b_{h,c}(\tau)\}\) 均匀改善、head-specialize，
或把风险集中到少数 pairs；若低阶状态不闭合，就构造相同状态、不同
\(\dot\gamma\)/collision tail 的反例。

下一阶段只做一个实验：在同一 hard cell、相同初始化与新 confirmation seeds 上运行
\(\text{rank/dense }B\times\text{rank/dense }C\) 的 \(2\times2\) 矩阵，同时保存完整
concept score margins 与高精度 triad swap。它一次区分 QK-rank、OV-rank 和 joint-rank；
在它完成前，不扩模型，也不把 rank--collision 候选升级为论文事实。

核心图只保留四张：toy paired capacity forest；nested-MC tail/precision；composite
loss/representation geometry；通过 gate 后的 Pythia 8-checkpoint panel。raw-factor gauge、
完整 head grids 放附录；失败的 localization 不作科学结果图。
