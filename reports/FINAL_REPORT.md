# 固定有限 Transformer 中 routing、superposition 与训练动力学

## 完整研究闭环与可复现实验报告（文献边界截至 2026-08-15）

## 0. 先给结论

这轮工作把原先两个容易产生多种解释的问题，拆成了三个不同层级：

1. **函数层的 causal routing 必要条件**：已经有精确答案。对 fresh random-value
   retrieval，低 population risk 必然强迫输出的一阶 Walsh 系数集中在 queried slot；
   distractor 系数和高阶交互能量必须小。这不依赖 attention 热图。
2. **参数层的训练选择**：仍是真正开放的问题。我们还没有理论说明联合训练的
   learned embedding、factorized QK、OV、FFN 和 readout 为什么选择某一种内部实现，
   也不能从低风险推出某个特定 head 必须看 target。
3. **learned compressed representation 的下游补偿**：宽泛说法已被现有论文部分覆盖，
   但本项目关心的窄版本仍开放：在 episodic random values、learned compressed
   embedding 与完整 QK/OV/FFN 联合训练下，cross-talk 是否、何时、在哪个模块被有限地
   消除。

实验没有支持一个简单的探索性故事——“对称 midpoint 分解中的 QK route 项在下游抑制
content cross-talk”。但这一实现与预注册的非对称 content/route/interaction estimand 不同，
两者甚至可能给出相反符号；因此当前结果是 protocol deviation，只能保留为探索性负结果，
**不能声称检验或反驳了预注册 QK 命题**。OV 学到稳定的方向选择性，但还没有达到 causal compensation 的
证据门槛；FFN 的 signed cancellation 只在部分低负载配置出现，也没有通过完整的有限
干预、能量下限和跨优化器门槛。当前 **confirmed compensator = 0**。

与此同时，实验明确显示了一个值得继续追究、但尚不能叫作新 open problem 的现象：
在 $d=8,C=32,m=4,H=4$ 的高负载多头配置中，base retrieval 已接近完美，target-selective
表示几何也已形成，但 on-manifold distractor swap 仍有残余 cross-talk。保持较优学习率并
延长训练时，未校正 pointwise 区间支持残差下降；把学习率降到 .001 并没有成为可靠
remedy——cell 3 的 pointwise 区间完全高于零，cell 7 的 swap-MSE 区间跨零。由于这些
cells 与 seeds 已参与筛选，现阶段只有 targeted exploratory evidence 把慢收敛或优化路径
列为优先候选；它还不能确认该解释，更不支持不可消除的表示容量障碍。

本项目最终包含：659 条已完成、分别执行的 architecture×optimizer×seed 训练轨迹（其中
remedy 会与 baseline 配对复用 seed 编号，故 659 是工作量清单而不是推断样本量）、超过
2,100 个机制 snapshot 评估、两个优化器、
完整 $2^4$ 架构网格、定向 remedy、固定参数 clustering 复现，以及 loss landscape、分组
NTK、初始化线性化和 Hessian 个案。所有 claim 都按 training seed 统计；episode、head、层和
checkpoint 都不被伪装成独立样本。

---

## 1. 到底训练了什么网络、解决什么任务

### 1.1 数据分布

固定 concept vocabulary 大小 $C$ 和 memory slots 数 $m$。每个 episode 先无放回抽取

\[
(c_1,\ldots,c_m)\in[C]^m,\qquad c_i\neq c_j\quad(i\neq j),
\]

再独立抽取随机值和 target 位置

\[
v_i\overset{\mathrm{iid}}\sim\operatorname{Unif}\{-1,+1\},
\qquad J\sim\operatorname{Unif}[m],
\]

最后令

\[
q=c_J,\qquad Y=v_J. \tag{1}
\]

给大一学生的版本：题目给出几张“名字—硬币”卡片，硬币每道题重新掷成 $\pm1$；最后问
某个名字对应的硬币。因为硬币每题重掷，网络不能把答案背进名字 embedding，只能在当前
序列里找到正确卡片并读取它的值。

### 1.2 输入和完整模型

序列长度 $T=m+1$。memory 与 query 的输入为

\[
x_i^0=E_{c_i}+v_i e_v+e_{\rm mem}+p_i,\quad i\le m,
\]

\[
x_T^0=E_q+e_{\rm qry}+p_T. \tag{2}
\]

$E\in\mathbb R^{C\times d}$、value/type/position vectors 都学习。实验网络是有限、标准
softmax、pre-RMSNorm causal Transformer；默认 $L=2$，$d\in\{8,16,32\}$，
$H\in\{1,4\}$，并比较 attention-only 与宽度 $2d$ 的 GELU FFN。

对层 $\ell$、head $h$，定义 gauge-invariant composites

\[
B_{\ell h}=Q_{\ell h}^{\top}K_{\ell h},
\qquad
C_{\ell h}=O_{\ell h}V_{\ell h}. \tag{3}
\]

令 $z_t^\ell=\operatorname{RMSNorm}(x_t^\ell)$，则

\[
s_{ti}^{\ell h}
=\frac{\beta}{\sqrt{d_h}}(z_t^\ell)^\top B_{\ell h}z_i^\ell,
\qquad
a_{ti}^{\ell h}
=\frac{e^{s_{ti}^{\ell h}}}{\sum_{r\le t}e^{s_{tr}^{\ell h}}}, \tag{4}
\]

\[
x_t^{\ell,\mathrm{att}}
=x_t^\ell+\frac1{\sqrt L}\sum_h
C_{\ell h}\sum_{i\le t}a_{ti}^{\ell h}z_i^\ell. \tag{5}
\]

如果有 FFN，继续更新

\[
x_t^{\ell+1}=x_t^{\ell,\mathrm{att}}
+\frac1{\sqrt L}\left[
U_{\ell,2}\operatorname{GELU}\!\left(U_{\ell,1}
\operatorname{RMSNorm}(x_t^{\ell,\mathrm{att}})+b_{\ell,1}\right)+b_{\ell,2}
\right]. \tag{6}
\]

输出是 query token 的 scalar readout：

\[
f_\theta(X)=w^\top\operatorname{RMSNorm}(x_T^L)+b. \tag{7}
\]

理论对象是 population risk 和 gradient flow

\[
\mathcal R(\theta)=\frac12\mathbb E(f_\theta(X)-Y)^2,
\qquad
\dot\theta_s=-\nabla_\theta\mathcal R(\theta_s). \tag{8}
\]

实验每一步重新生成 online mini-batch，以 AdamW（weight decay 为 0）或 momentum-SGD
近似 population dynamics。训练时间记作 $s$；层深记作 $\ell$，二者绝不混用。

---

## 2. “causal routing”在这里到底是什么意思

### 2.1 causal mask 只是图结构，不是解释

网络满足

\[
a_{ti}^{\ell h}=0\quad\text{when }i>t. \tag{9}
\]

这表示未来 token 不能影响过去 token；它不表示某张 attention 热图已经是因果解释。

### 2.2 end-to-end value causal kernel

固定 concept、target 和其余 values，只翻转第 $i$ 个外生 value：

\[
\kappa_i(c,v_{-i},J)
=\frac12\left[
f_\theta(\operatorname{do}(v_i=+1))
-f_\theta(\operatorname{do}(v_i=-1))
\right]. \tag{10}
\]

这是真正的 finite causal effect；它会穿过所有 QK、OV、residual、FFN 和 readout 路径。

### 2.3 Walsh–Parseval 给出的精确 forcing theorem

固定 $(c,J)$，把网络视为 Boolean cube 上的函数 $f(v_1,\ldots,v_m)$，定义

\[
\widehat f_S(c,J)
=2^{-m}\sum_{v\in\{-1,+1\}^m}
f(c,v,J)\prod_{i\in S}v_i. \tag{11}
\]

因为标签就是 $v_J$，正交性给出精确恒等式

\[
2\mathcal R
=\mathbb E_{c,J}\left[
(\widehat f_{\{J\}}-1)^2
+\sum_{S\neq\{J\}}\widehat f_S^2
\right]. \tag{12}
\]

所以若 $\mathcal R\le\varepsilon$，则 target direct coefficient 必须接近 1，bias、所有
distractor direct coefficients 和所有高阶 interactions 的总能量必须至多为
$O(\varepsilon)$。同样，注册的 value-flip statistic

\[
\Xi=\frac12\mathbb E\left[Y(f(X)-f(X^{\rm flip\,J}))\right] \tag{13}
\]

满足

\[
|\Xi-1|\le\sqrt{2\varepsilon}. \tag{14}
\]

这解决了“低风险是否强迫任务相关 causal value dependence”这一**函数层**问题。

它没有解决“哪个 head、哪个 QK/OV factorization 被 gradient flow 选择”。同一个函数可由
多种内部电路实现，甚至可以先把 target value 搬到别的 memory token 再传给 query。

### 2.4 target-key mask 是路径效应，不是总效应

内部 key-path 干预把 query-to-slot-$i$ 的 logits 在所有层和 heads 中改为 $-\infty$，然后
重算全部后代。其 signed effect 可写成

\[
\delta_i=Y\{f(X)-f(\operatorname{do}(s_{Ti}^{\ell h}=-\infty,\ \forall\ell,h))\}. \tag{15}
\]

这测量指定 attention edges 的路径效应。它不能代替 (10) 的总因果效应。

当前发布 evaluator 只保存 target-edge blocking effect，没有逐个阻断 distractor edges；
所以注册的 direct-path selectivity $S_{\rm key}$ 尚未评估。机制表中的正值门槛只是
`target-edge effect + descriptive attention selectivity` 的探索性 screen，不能叫 causal
direct-key gate。

---

## 3. “superposition”与“补偿”也必须精确定义

### 3.1 本实验直接测的是 compressed dictionary geometry

learned concept vectors 为 $e_c=E_c$，Gram matrix 为

\[
G_E=EE^\top. \tag{16}
\]

我们记录：

\[
\operatorname{coh}(E)=
\max_{c\neq c'}
\frac{|\langle e_c,e_{c'}\rangle|}{\|e_c\|\|e_{c'}\|}, \tag{17}
\]

以及由奇异值能量 $\lambda_k$ 定义的 participation/effective rank

\[
r_{\rm eff}(E)=
\frac{(\sum_k\lambda_k)^2}{\sum_k\lambda_k^2}. \tag{18}
\]

factorial 图表使用的 normalized embedding rank 是

\[
\widetilde r_E=\frac{r_{\rm eff}(E)}{d}\in[0,1]. \tag{18a}
\]

当 $C>d$ 时，所有 concept 不可能彼此正交；这是有限维压缩几何。它并不自动证明某个单一
activation 同时编码了多个可独立解码 feature。因此报告使用“learned compressed
dictionary geometry”，不会把低 rank 单独宣传成完整 activation superposition 证明。

### 3.2 逐层表示几何

在 input、每层 post-attention 和 post-FFN residual 上，以每个 episode 为单位记录：

- query–target cosine；
- query–distractor mean cosine；
- 二者差 $S_\ell^{\rm repr}$；
- 所有 token 的 global off-diagonal cosine；
- episode-centered covariance participation rank。

这里最重要的是把**全局聚拢**与**任务选择性**分开：

\[
S_\ell^{\rm repr}
=\cos(x_q^\ell,x_J^\ell)
-\frac1{m-1}\sum_{i\neq J}\cos(x_q^\ell,x_i^\ell). \tag{19}
\]

### 3.3 on-manifold distractor swap

选择 $K\neq J$，把 $c_K$ 换成当前 memory 中未出现的新 concept $c'_K$；保持 values、query、
target index 和 label 不变。原输入和替换后输入都来自数据支持。

\[
X' = X^{K:c_K\mapsto c'_K},
\qquad Y'=Y. \tag{20}
\]

输出 cross-talk 是

\[
\chi_{\rm out}
=\mathbb E\left[(f_\theta(X')-f_\theta(X))^2\right]. \tag{21}
\]

只有 base retrieval 很好、donor 输入也正确、$\Xi$ 足够大且 $\chi_{\rm out}$ 足够小，才通过
严格 causal-robustness gate。注册阈值是 accuracy $\ge0.95$、risk $\le0.05$、
$\Xi\ge0.90$、donor accuracy $\ge0.95$、swap MSE $\le2.5\times10^{-3}$。

### 3.4 finite attention chord 的精确局部分解

对一个 attention head，令两个 on-manifold 端点的中点量用横线表示，差用 $\Delta$ 表示。
输出差精确分解为

\[
\Delta U
=C\sum_i\bar a_i\Delta z_i
+C\sum_i\Delta a_i\bar z_i. \tag{22}
\]

第一项是 content，第二项是 route。它不是一阶近似，而是 midpoint bilinear identity。
对应 tangent JVP 为

\[
DU[\delta z]
=C\sum_i a_i\delta z_i
+C\sum_i a_i(z_i-\bar z)\delta s_i. \tag{23}
\]

代码同时用手算、autograd JVP 和中心差分验证 (23)。

相邻确定性节点使用同一个 coherent donor state 做 patch 时，理论上应得到相同后代输出；
所以不能用“pre-OV patch 比 post-OV patch 小”声称 OV 在补偿。真正可接受的证据是有限
content/route 分解、OV 对 task 与 distractor chord 的方向选择性、FFN residual branch 在
downstream adjoint 下的 signed cancellation，以及完整 on-support 输出验证共同成立。

---

## 4. 实验矩阵：做了什么、为什么做

| 实验 | 变量和规模 | 回答的问题 |
|---|---:|---|
| instrumentation smoke | 2 cells $\times$ 2 seeds | 数据、causal mask、patch、checkpoint、metrics 是否闭环 |
| primary AdamW | 8 cells $\times$ 12 seeds，400 steps | $C\times H\times$ FFN 下 routing/geometry 怎样形成 |
| momentum-SGD replication | 同样 8 cells $\times$ 12 seeds，800 steps | 机制方向是否只是 AdamW 特例 |
| SGD stuck-seed extension | 1 seed，1600 steps | plateau 是否只是训练不足 |
| high-LR $2^4$ scaling | width $\times$ load $\times$ heads $\times$ FFN，160 runs | 架构主效应与交互；先暴露 failure |
| pilot/remedy pilots | 72 seed-runs | 先用已有优化手段排查高 LR failure |
| tuned $2^4$ scaling | 16 cells $\times$ 10 seeds，800 steps | base-function/gate 的预注册验证；rank secondary contrasts 仅探索性 |
| targeted low-LR remedy | 4 cells $\times$ 10 seeds，1600 steps | 边界/困难 cross-talk 是否被更小 LR 修复 |
| same-LR extension | 3 cells $\times$ 10 seeds，1600 steps | 把 LR 与训练时长分开 |
| mechanism replay | 固定 held-out episodes；多批次共 $>2100$ rows | attention、Walsh、swap、OV、FFN、表示轨迹 |
| dynamics case studies | 4 controlled cell/seed cases | loss landscape、NTK、linearization、Hessian |
| Perspective baseline | 64 particles，固定 $Q=K=V=I$ | clustering 与训练所得 selective routing 的关系 |

总计 659 条已完成训练轨迹；这是跨主实验、pilot 与配对 remedy 的工作量清单，不是 659 个
彼此独立的推断样本。大网格的统计单位始终是 training seed；16 个 architectures、
多个 checkpoints、heads 和 8,192 个 held-out episodes 都只提供 seed 内精度，不增加
$n$。主要区间使用 20,000 次 whole-seed paired bootstrap。

训练失败或新现象的处理顺序是：

1. 检查数据/实现/数值和 estimator；
2. 查找最接近的训练理论与已知优化手段；
3. 做同 seed、同初始化或同 held-out stream 的配对 remedy；
4. 只有在合理 remedy 后仍稳定、且文献没有覆盖，才升级为候选 open problem。

完整 keep/discard 账本在 `autoresearch/classic-260815-0735/results.tsv`。

---

## 5. 主要实验结果

### 5.1 tuned 后，base function-level routing 是稳定的

高学习率初始 scaling 网格有 7/160 个 seed-runs 没通过预注册 base gate。降低学习率并延长
训练后，tuned 网格 160/160 都满足：accuracy $\ge0.95$、population risk $\le0.05$、
value-flip effect $\ge0.90$。

这说明早期失败首先是优化失败，不是 width/load/head/FFN 的容量反例。Walsh decomposition
同时显示 target coefficient 接近 1，distractor 与 interaction energy 小；Parseval numerical
gap 最大约 $4.77\times10^{-6}$。

### 5.2 “base retrieval 成功”不等于“cross-talk 已消除”

以 256 episodes 的固定-stream机制网格评估，严格 full gate 为 136/160 seed-runs；
2,048-episode 高精度复验为 132/160，逐 cell 的严格 10/10 gate 为 9/16。这里的逐 seed
threshold crossing 是描述性 screen，不是 cell mean 显著性检验。7 个 cells 至少有一个 seed
越过阈值，但其中 5 个 cell 的 mean 95% CI 完全低于 $2.5\times10^{-3}$，属于单 seed tail
或 evaluation-stream sensitivity；只有 cells 3/7 有 material mean residual。材料残差因此
更具体地集中在 $d=8$、load 4、4-head 两个配置，而不是所有压缩模型或所有高负载多头模型。

最困难两个配置在 tuned step 800 的 2,048-episode 均值为：

| $d,C,m,H,$ FFN | base MSE | donor MSE | swap MSE | Walsh distractor direct | Walsh interaction |
|---|---:|---:|---:|---:|---:|
| $8,32,4,4,$ none | 0.02426 | 0.02370 | 0.02175 | 0.00777 | 0.00842 |
| $8,32,4,4,$ 16 | 0.02341 | 0.02159 | 0.02143 | 0.00685 | 0.00768 |

这不是“阈值设得略严”造成的假象：base/donor/Walsh energy 也仍有实质残差。但它也不是
global collapse：对应表示已表现出明显 query–target selectivity，而 global token cosine
更低。

### 5.3 remedy 判定：定向探索支持慢收敛候选，但尚未独立确认

三组高精度评估使用相同 seed 映射和 evaluation offset，每个终点 2,048 episodes，0 failures。
每个比较先在同一个 architecture×seed 内作差，再对 10 维 seed-difference vector 做 20,000 次
percentile bootstrap；head、episode 和 cell 都没有被当成额外独立样本。

这是在先前结果中定向选出的困难 cells 上、复用训练 seeds 0--9 的 exploratory remedy
follow-up。它违反协议中“独立 remedy seeds + 从未调优 confirmatory seeds”的最终确认步骤，
也没有对 7 个 pointwise contrasts 做 family-wise 校正。因此下表的区间用于判断“现有优化
伪象是否仍然可能”，不能作为预注册 phase-boundary 检验或新样本确认性显著性结论。

| schedule | cell | baseline→follow-up swap MSE | paired $\Delta$ [95% CI] | gate before→after |
|---|---:|---:|---:|---:|
| same LR=.003, 1600 | 3: $d8,C32,H4,$ no FFN | 0.021754→0.008685 | $-0.013069$ [$-0.017047,-0.009953$] | 0/10→0/10 |
| same LR=.003, 1600 | 7: $d8,C32,H4,$ FFN16 | 0.021428→0.004853 | $-0.016575$ [$-0.025496,-0.009016$] | 0/10→4/10 |
| same LR=.003, 1600 | 11: $d32,C128,H4,$ no FFN | 0.000955→0.001343 | $+0.000388$ [$-0.000966,+0.001783$] | 7/10→8/10 |
| lower LR=.001, 1600 | 3 | 0.021754→0.033209 | $+0.011455$ [$+0.005938,+0.017867$] | 0/10→0/10 |
| lower LR=.001, 1600 | 6 | 0.000844→0.000883 | $+0.000040$ [$-0.000212,+0.000328$] | 9/10→9/10 |
| lower LR=.001, 1600 | 7 | 0.021428→0.028428 | $+0.007000$ [$-0.000065,+0.014697$] | 0/10→0/10 |
| lower LR=.001, 1600 | 11 | 0.000955→0.000220 | $-0.000735$ [$-0.001588,+0.000060$] | 7/10→10/10 |

保持 LR=.003 延长训练，在这批定向复用的 seeds 上把两个困难配置的 swap MSE 分别降低约
60% 和 77%；相应 base MSE 也降至 0.00763 和 0.00404，两个未做 family correction 的
pointwise paired CI 都完全低于零。这是支持“继续训练可减小残差”的探索性证据，而不是
独立确认。较低 LR=.001 对 cell 3 的 pointwise CI 完全高于零；cell 7 的 swap CI 跨零，
不能仅据该 endpoint 声称显著恶化，不过它的 base、donor 与
Walsh leakage 也同时变差。Cell 11 的 10/10 threshold crossing 同样不等于 mean effect 已显著，
因为 paired CI 仍跨零。因而“学习率越小越能消除 cross-talk”被否定；“仍在慢收敛/依赖
优化路径”只得到 targeted exploratory support。

但 step 1600 后 strict passes 仍只有 0/10 和 4/10，不能写成已经解决。当前正确分类是：

> 一个经优化后仍可测、且在定向复用的 seeds 上随合理延长训练明显衰减的实验现象；
> pointwise 区间尚未经 family correction 或新 seeds 确认，所以它没有资格成为不可消除
> 容量 open problem。下一步需要独立 confirmatory seeds、训练极限与模块归因。

### 5.4 对称 midpoint QK 诊断不支持简单 suppression 故事；预注册命题尚未检验

预注册协议把有限 chord 非对称地分为
\(a\,\Delta z\)、\(\Delta a\,z\) 和 \(\Delta a\,\Delta z\)，并用
content+interaction 对 total 的 log-ratio。当前 evaluator 实际使用对称 midpoint 恒等式：

\[
\delta m_{content}^{sym}=\bar a\,\Delta z
=\delta m_{content}+\tfrac12\delta m_{interaction},\qquad
\delta m_{route}^{sym}=\Delta a\,\bar z
=\delta m_{route}+\tfrac12\delta m_{interaction}. \tag{25a}
\]

它没有单独保存 interaction，因而与预注册 estimand 不等价，极端情况下可给出相反符号。

对这个**探索性 midpoint 统计量**，所有 16 个 optimizer$\times$cell 聚合的终点和
init-to-final 增量均为负；8 个 AdamW/SGD 匹配 cells 中 0/8 支持其 suppression 方向。
这说明 midpoint route 分量通常放大而不是抑制这条 chord。

这是有价值的探索性诊断：它反对最直接、也最容易从 attention 图误读出来的 midpoint
补偿故事。但预注册统计量尚未计算，不能把 0/8 写成确认性反证。下一轮必须同时保存三个
非对称 endpoint 项、预注册 log-ratio 与 finite hybrid output。

### 5.5 OV 有方向选择性，但还不是补偿定理

比较 $\|C_{\ell h}\delta\|/\|\delta\|$ 在 target-value chord 与 distractor-concept chord
上的增益。8/8 匹配 cells 的 target-vs-distractor selectivity 都随训练增强；6/8 在两个
优化器中置信区间支持同一方向。

这说明 OV composite 不只是各向同性缩放，确实学到 task-relevant directional geometry。
但方向选择性本身不证明它在真实 forward path 上抵消了 swap cross-talk；尤其 QK route
项已经与简单抵消故事相反。因此它被标为“候选机制”，不是 confirmed compensator。

### 5.6 FFN compensation 没有通过完整门槛

在 downstream adjoint 下，我们分别记录 skip、FFN branch 和 total 的 signed contribution，
要求：

- skip 与 branch 反号；
- cancellation fraction 足够大；
- 两项能量都超过 practical floor；
- tangent 与 finite on-support intervention 一致；
- 在 AdamW 和 SGD 中重复。

低负载的部分 cells 出现反号/抵消，高负载则不稳定；finite validation 和 energy floor 没有
同时通过。因此不能声称“FFN 已经学会纠错”。

### 5.7 learned embedding geometry 的 exploratory factorial 结果

在完整 $2^4$ tuned 网格上，先在每个 seed 内计算架构对比，再对 10 个 seeds 做 20,000 次
paired bootstrap。这里所谓 width contrast 实际是在固定 load $C/d$ 下同时把
$(d,C)$ 从 $(8,8)$ 扩到 $(32,32)$（load 1），或从 $(8,32)$ 扩到 $(32,128)$
（load 4）；因此它是 **fixed-$C/d$ scale contrast**，不是控制 $C$ 不变的 isolated-width
effect。协议把 normalized rank 与未注册 interaction 归为 secondary family，并要求 BH
$q=.10$；当前表只给出 7 个选择后 contrasts 的未校正 pointwise percentile intervals，
没有执行该 family adjustment。因此它们是 architecture-pattern discovery，不能作为
confirmatory factorial effects。normalized embedding rank
$\widetilde r_E=r_{\rm eff}(E)/d$ 的探索性结果为：

| contrast | paired effect | 95% CI |
|---|---:|---:|
| scale $d:8\to32$ at fixed $C/d$ | -0.0298 | [-0.0359, -0.0231] |
| load 4 minus 1 | +0.1404 | [0.1152, 0.1636] |
| heads 4 minus 1 | -0.0491 | [-0.0564, -0.0425] |
| FFN on minus off | +0.0280 | [0.0172, 0.0390] |
| heads$\times$load | -0.0586 | [-0.0849, -0.0343] |
| heads$\times$width | -0.0197 | [-0.0343, -0.0041] |
| FFN$\times$load | +0.0234 | [-0.0018, 0.0477] |

coherence 的主要效应为：width 增大约 -0.187、load 增大约 +0.118、heads 增大约 +0.022；
FFN 对 coherence 的效应接近 0。

这些是 learned dictionary geometry 的证据，不是“一个 neuron 表示多个 concept”的直接
证据。它为下一步的容量/动力学理论提供 order parameters：$G_E(s)$、coherence、normalized
rank 以及 head$\times$load interaction。

### 5.8 global average alignment 与 label-conditioned selectivity 明确分离

训练通常降低 global off-diagonal token cosine，同时让 query–target minus
query–distractor cosine 沿深度显著升高。例如 load 4 的聚合结果中：

- $d=8$：末层 global cosine 约 0.194，而 target-selective cosine 约 0.762；
- $d=32$：末层 global cosine 约 0.239，而 target-selective cosine 约 0.818。

所以任务表示形成不是“所有 token 越来越相似”的单点 consensus；它同时形成了有标签结构
的选择性几何。这里的 cosine difference 是 representation statistic，不是 causal routing。
global mean cosine 也不能排除多个彼此分离的 clusters；要判断 multi-cluster clustering，仍需
pairwise-cosine distribution、谱或显式 cluster order parameter。

### 5.9 loss landscape、NTK 和 plateau 个案说明了什么

我们对同一 cell/seed 的 high-LR plateau 与 tuned run 使用完全相同初始化；step 0 的 23 个
probe arrays bitwise identical。固定 8,192 episodes 的 step-400 结果为：

| run | MSE | accuracy | value flip | target-key effect |
|---|---:|---:|---:|---:|
| high LR plateau | 1.00834 | 0.4982 | $-1.23\times10^{-6}$ | $-1.46\times10^{-6}$ |
| tuned | $4.626\times10^{-4}$ | 1.0000 | 0.9986 | 0.9877 |

high-LR 的 QK-group NTK norm 从 0.07517 降到 $5.62\times10^{-11}$；tuned step 400 仍为
0.001049。初始化线性化误差分别为 8.227 和 1.944。二维 filter-normalized landscapes、
Lanczos Ritz values 和 Hutchinson trace 共同显示两条优化轨迹进入了完全不同的局部动力学
区域。

这个个案支持“高 LR 使 feature-learning channels 近乎失活”的机制解释，但它只有一个
common-initialization seed，因此不是 LR 的群体因果效应。群体层结论来自完整 tuned 网格：
调优后 160/160 base gate 通过。

---

## 6. 与 *A Mathematical Perspective on Transformers* 的 clustering 是什么关系

两项研究的“时间”不同。

Perspective 在固定参数 $Q,K,V$ 后研究 token 随层深/连续深度 $t$ 的粒子动力学，例如

\[
\dot x_i=P_{x_i}\sum_j a_{ij}(x)x_j. \tag{24}
\]

本项目固定 architecture，研究参数随训练时间 $s$ 的变化

\[
\dot\theta_s=-\nabla\mathcal R(\theta_s), \tag{25}
\]

以及由此诱导的 $x_{s}^{\ell}$、$B_{\ell h,s}$、$C_{\ell h,s}$ 和 $G_{E,s}$。

我们按 Perspective 官方 `sphere.py` 的核心更新复现了 $A=V=I$ baseline：64 个球面粒子从
mean off-diagonal cosine 0.01447 演化到几乎 1，Gram participation rank 从 2.7948 降到
约 1。这是非常强的 global clustering。

但终点 attention entropy 等于 1（按 $\log n$ 归一化），且 $a_{ij}=1/n$：attention 是
均匀的，没有选择 queried token。因此

\[
\boxed{\text{global clustering}\not\Rightarrow
\text{selective routing}\not\Rightarrow
\text{task-level causal routing}.} \tag{26}
\]

两者真正的桥梁不是把 clustering 当作 routing，而是研究训练如何改变 interaction kernel：

\[
Q,K,V\ \text{fixed}
\quad\longrightarrow\quad
Q_s,K_s,V_s,E_s\ \text{learned}. \tag{27}
\]

本项目的 representation geometry 进一步排除了“成功只需无条件单点 consensus”这一简单
解释：成功训练往往降低 global mean cosine，却提高 label-conditioned target selectivity。
这说明任务需要有结构的分离/聚合；它不排除 Perspective 意义下更一般的 multi-cluster
dynamics，也不把 cosine statistic 称为 causal routing。

---

## 7. 文献边界：哪些已解决，哪些仍开放

### 7.1 三份原始材料的明确位置

- Hanin `nn.notes`：Open Problem 5（width 与样本共同增长的 DMFT）、Open Problem 6
  （general DAG architectures），以及 superposition 部分对 sparse features、capacity 与
  learned computation 的问题。
- *A Mathematical Perspective on Transformers*：Problem 2.15（随机 QKV）、Problems
  3.1/3.2/3.5/3.7/3.8，以及 §3.4 明确指出的 training/learned-parameter gap。
- *A Mathematical Explanation of Transformers for LLMs/GPTs*：没有编号为
  “Open Problem”的条目。它把 Transformer 写成受控积分—微分/约束优化系统，但没有给出
  optimizer 所产生的控制轨迹理论；因此它提供数学骨架，而不是一张可直接引用的 open-
  problem 清单。

### 7.2 截至 2026-08-15 最接近的理论

- [Im et al. 2026](https://arxiv.org/abs/2601.19208)：固定 one-hot embedding、
  attention-only LM 的早期 GD leading terms 与误差界。
- [Yang et al. 2024](https://proceedings.neurips.cc/paper_files/paper/2024/hash/520416e27d3b0cef3cd70a083e2991c7-Abstract-Conference.html)：
  固定正交 embedding、factorized Q/K/V 与线性 MLP 的二阶段动力学。
- [He et al. 2025](https://proceedings.mlr.press/v267/he25q.html)：固定 Gaussian
  features、causal softmax attention 的 reduced QK/OV gradient-flow ODE。
- [Chen et al. 2024](https://arxiv.org/abs/2409.10559)：阶段式 population GF 下 induction
  head 与 FFN/QK 协同形成。
- [Vural et al. 2026](https://arxiv.org/abs/2603.15923)：固定非正交随机 embeddings 下的
  QK/value/MLP 早期训练理论。
- [Nichani et al. 2024](https://arxiv.org/abs/2412.06538)：OV 与 FFN associative memory 的
  容量和存在性 trade-off。
- [Ravfogel et al. 2026](https://arxiv.org/abs/2605.12426)：learned subject embeddings 以
  attribute superposition 编码、下游 relation-conditioned selector 解码的构造/容量定理和
  causal interventions；GD 发现结构主要是实验结论。
- [Adler 2025](https://arxiv.org/abs/2509.22840)：压缩表示下 multi-head QK capacity 的上下界。
- [Persian Rug 2024](https://arxiv.org/abs/2410.12101) 与
  [Neural Computation in Superposition 2024](https://arxiv.org/abs/2409.15318)：非 Transformer
  toy models 中的 cross-talk、非线性 denoising 和容量构造。

在本报告检索的一手论文、其参考链和截至 2026-08-15 的关键词查重范围内，没有找到一篇论文同时证明

\[
\text{learned compressed }E
+\text{ causal multi-head softmax}
+\text{ trainable QK/OV/FFN}
+\text{ joint population GF}
+\text{ finite downstream compensation}. \tag{28}
\]

这不是说各部分“没人研究”；恰恰相反，每一部分都有非常接近的结果。开放的是它们在同一
个受控问题中的交集。

### 7.3 四类研究地图

| 分类 | 当前结论 |
|---|---|
| 已解决 | fresh random-value retrieval 的函数层 Walsh/Parseval forcing；固定 $Q=K=V=I$ 的 global clustering baseline；代码级 finite/JVP identities |
| 已有近似理论 | 固定 one-hot/orthogonal/Gaussian/nonorthogonal embeddings；简化或分阶段 QK/OV/MLP training；superposition 的容量/构造理论 |
| 实验已知、理论缺失 | learned embedding rank/coherence 的 exploratory architecture patterns；OV directional selectivity；高负载多头 residual cross-talk；plateau 中 group-NTK collapse |
| 真正开放 | 联合 learned $E+$factorized QK/OV/FFN/readout 的 population training-selection closure；episodic on-manifold cross-talk 的有限模块补偿定理/反例 |

---

## 8. 两个最值得推进的主问题

### A. 复合 routing kernel 的训练选择理论

### 已经解决的部分

公式 (12)–(14) 证明：低 risk 强迫函数层的正确 causal coefficient。它是 exact theorem，
不是实验猜想。

### 尚未解决的部分

令

\[
G_s=E_sE_s^\top,
\quad B_{\ell h,s}=Q_{\ell h,s}^\top K_{\ell h,s},
\quad C_{\ell h,s}=O_{\ell h,s}V_{\ell h,s}. \tag{29}
\]

若 $G_B=\nabla_B\mathcal R$，factorized GF 精确满足

\[
\dot B=-G_BS_K-S_QG_B,
\quad S_Q=Q^\top Q,
\quad S_K=K^\top K, \tag{30}
\]

\[
\dot S_Q=-G_BB^\top-BG_B^\top,
\qquad
\dot S_K=-G_B^\top B-B^\top G_B. \tag{31}
\]

OV 有完全类似的方程。困难不在写出 (30)，而在 $G_B$ 仍依赖完整数据分布、softmax、
RMSNorm、embedding、其他层和 readout，尚未对有限 order parameters 闭合。

### 可做成论文的具体 theorem/anti-theorem 目标

先限制到 $L=1$、无 FFN、value-blind scores、population GF 和 exchangeable random
initialization。定义 target/distractor score moments、value-readout overlap、embedding
Gram moments和 factor imbalance。目标是证明，在一个非平凡时间窗 $s\in[0,S]$，这些
order parameters 满足有限维 ODE 加显式误差：

\[
\sup_{0\le s\le S}
\|M_d(s)-M(s)\|
\le C_S d^{-1/2}\operatorname{polylog}(C,m). \tag{32}
\]

随后证明下列二者之一：

1. **选择定理**：在明确初始化尺度、$C/d$、$m$、$H$ 条件下，ODE 从 exchangeable state
   失稳并选择 target score/value overlap，使 risk 到 0；或
2. **反例定理**：存在同样低风险的稳定 attractors，具有不同 attention routing 或绕行
   paths，从而说明 function-level forcing 不能升级成唯一内部 routing。

这与 Im/Yang/He 的差异必须保留为 learned $E$、episodic random values、factorized
composites 和 causal routing，不应写成泛泛的“研究 Transformer GD”。

### B. learned superposition 的下游补偿理论

### 宽泛版本为什么不能再声称完全开放

Ravfogel 已给出 attribute-superposed embeddings 与 relation-conditioned downstream
selector 的构造/容量理论；Nichani、Vural、Persian Rug、Adler & Shavit 分别给出 OV/FFN
trade-off、非正交 interference、ReLU denoising 和 superposition 中计算的理论。因此“下游
能否读取/修复 superposed representation”已有多种肯定答案。

### 本项目保留的窄问题

对 (1) 的 episodic random-value causal retrieval，让 $E,B,C,F,w$ 从随机初始化联合训练。
定义逐层 on-manifold cross-talk

\[
\chi_\ell
=\mathbb E\|x_T^\ell(X')-x_T^\ell(X)\|^2,
\qquad
\chi_{\rm out}=\mathbb E(f(X')-f(X))^2. \tag{33}
\]

同时定义 finite QK content/route、OV direction-conditioned gains、FFN signed residual
contributions。目标不是观察 $\chi_{\rm out}$ 变小，而是证明某个模块的结构方程使它变小。

可证命题应具有形式：在明确的 compression/load/sparsity 条件下，若输入 cross-talk 超过
$a>0$，训练后的某个 suffix map $T_{\ell:L}$ 满足

\[
\mathbb E\|T_{\ell:L}(x+\Delta_{\rm dist})-T_{\ell:L}(x)\|^2
\le\rho\,\mathbb E\|\Delta_{\rm dist}\|^2,
\quad \rho<1, \tag{34}
\]

但对 target-value direction 保持

\[
\mathbb E\|T_{\ell:L}(x+\Delta_{\rm target})-T_{\ell:L}(x)\|^2
\ge\gamma\,\mathbb E\|\Delta_{\rm target}\|^2,
\quad \gamma>\rho. \tag{35}
\]

并且 (34) 的下降能由 QK、OV、FFN 或 readout nulling 中至少一个 finite intervention 唯一
归因；否则要给出反例，证明补偿天然是分布式、不能模块唯一定位。

当前探索性 midpoint 诊断反对“QK route 普遍抵消”这个朴素候选，但预注册 QK estimand
尚未计算；OV selectivity 是下一候选，但仍未证明 (34)–(35)。

---

## 9. 下一轮最小但有决定力的实验

不应立刻扩大到 GPT-scale。最有信息量的是围绕 $d=8,C=32,m=4,H=4$ 残差做下列受控矩阵：

1. **训练极限**：同 seed 继续到 3200/6400 steps，保留 LR=.003，并加入 cosine decay；
   判定 swap/Walsh leakage 是否与 base risk 同速趋零。
2. **factorization conditioning**：直接训练 composite $B,C$，对照 factorized $Q/K,O/V$；
   若 direct composite 修复，问题更接近优化几何而非函数容量。
3. **表示来源**：固定正交/随机 $E$、训练 $E$、以及受控低 coherence $E$；分离 compressed
   dictionary cross-talk 与 attention head bottleneck。
4. **head capacity**：保持总 $d$、保持每头 $d_h$、保持参数量三种不同对照，避免把“heads
   多”与“每头维数小”混为一谈。
5. **模块 causal localization**：对每个 seed 使用同一 on-support swap，逐层测 finite
   QK chord、OV selective gain、FFN signed energy 和 suffix output；不得用相邻 coherent
   patch 的理论等价制造假 attenuation。
6. **population-GF bridge**：在小 $C,m$ 上枚举 concepts、target 和 $2^m$ values，运行真正
   full-batch GF-like 更新，与 stochastic AdamW/SGD 比较 order-parameter ODE。

只有当更长训练、scheduler、direct composites、fixed-$E$ controls 和 head-capacity controls
都不能解释稳定残差，并且它在 seeds/optimizers 上重复，才有资格把它升级成新的容量或
动力学 open problem。

---

## 10. 可复现材料与阅读顺序

项目入口：`README.md`。

建议顺序：

1. 本报告：总问题、实验和结论；
2. `reports/THEORY_PROBLEMS.md`：完整符号、动力学和 theorem targets；
3. `reports/LITERATURE_MAP.md`：逐个原始 open problem 和 2026-08-15 查重；
4. `reports/ANALYSIS_PROTOCOL.md`：bootstrap、max-T、TOST、claim ladder；
5. `reports/MECHANISM_RESULTS_DRAFT.md`：QK/OV/FFN 的完整负结果与候选；
6. `reports/CLUSTERING_BASELINE.md`：Perspective clustering 复现；
7. `reports/DYNAMICS_RESULTS.md`：landscape/NTK/Hessian 个案；
8. `results/scaling-analysis-v1/`：factorial、表示几何和 remedy 派生表/图。

核心复现命令：

```bash
PYTHONPATH=src python -m unittest discover -s tests -v

PYTHONPATH=src python -m routing_lab.run \
  --config configs/scaling_tuned_adamw.json \
  --output results/scaling-tuned-reproduction \
  --device cuda

PYTHONPATH=src python -m routing_lab.study_evaluate \
  --run-directory results/scaling-tuned-reproduction \
  --output-directory results/scaling-tuned-mechanisms \
  --selected-steps 0 800 \
  --evaluation-batch-size 256 \
  --evaluation-seed-offset 900000 \
  --device cuda
```

代码采用显式 dataclass 配置、局部 generator、原子 records、可恢复 runner 和不可变 manifest。
每个关键数学 identity 都有手算或解析测试；昂贵实验与派生分析分离，分析不会修改 checkpoint。
GitHub 版本保存 source、tests、configs、reports、聚合 CSV/JSON 和 PNG/SVG，并只额外保存
四个已报告 dynamics cases 所需的 15 个最小 source snapshots。其余 `*.pt` checkpoint
留在本地，但可由 config 和 seed 重建。

---

## 11. 限制与不能主张的内容

- 实验是小到中等规模 synthetic retrieval，不等于真实 LLM 已具有同一动力学。
- fresh random values 是优点也是边界：它隔离 routing，但不覆盖长期 factual memory。
- AdamW/SGD online training 近似 population GF，不是连续时间 theorem 的实证替代。
- loss landscape 是二维 filter-normalized slice；Hessian 用 Lanczos/Hutchinson 估计，不能
  描述整个高维地形。
- 机制个案中的 common initialization 很有诊断价值，但单 seed 不能支持群体因果结论。
- dictionary compression、activation superposition、polysemantic neurons 是不同概念；本报告
  不将它们互换。
- attention weights、cosine、NTK 和 gradient 是描述性量；只有明确 `do`/patch 并重算后代的
  finite effect 被称为 causal。
- 当前没有找到 confirmed downstream compensator；这不是“网络没有补偿”的证明，只是
  现有证据不足以定位和确认。

---

## 12. 最终研究判断

现在最值得写成理论论文的不是“Transformer 会不会 clustering”，也不是“accuracy 很高所以
attention 学会了 routing”。最清楚的论文主线是：

\[
\boxed{
\text{函数层 causal forcing 已知}
\quad\Longrightarrow\quad
\text{联合参数训练选择未知}
\quad\Longrightarrow\quad
\text{compressed geometry 下的有限补偿未知}
}. \tag{36}
\]

问题 A 应先在单层、value-blind score、population GF 中建立可闭合的 composite order-
parameter theory，再逐项释放 RMSNorm、多层和 FFN。问题 B 应以本轮找到的高负载多头 residual
cross-talk 为实验靶点，但继续完成优化/容量 controls；只有通过 finite on-manifold 模块归因，
才能声称发现了 learned downstream compensation。

这两个方向既直接连接原始 notes/Perspective 的明确缺口，又被现有 2024–2026 理论严格限定；
它们不再是模糊口号，而是已有数据、反例候选、order parameters、可证命题和复现代码支撑的
具体研究问题。
