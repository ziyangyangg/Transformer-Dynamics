# Phase II 预注册：从有限 routing 残差到预训练语言模型

**冻结日期：** 2026-08-20  
**状态：** 在任何 Phase-II production seed 或预训练模型测试 episode 运行前冻结。  
**目标：** 先把 $d=8,C=32,m=4,H=4$ 的残余 cross-talk 分解成收敛、参数化、
字典几何、per-head bottleneck、容量分配或分布式补偿；全部已知解释失败后，才允许把它
升级为经验开放问题。

本协议不改变 Phase-I 的任何结果或 hash contract。Phase-II 使用独立 schema/version。

---

## 1. 数据、网络与风险

一个 episode 为

\[
U=(c_{1:m},v_{1:m},J),\qquad
c_i\ne c_j,\quad v_i\in\{-1,+1\},
\]

\[
q=c_J,\qquad Y=v_J .
\tag{P1}
\]

主 hard cell 固定

\[
(C,d,m,L,H)=(32,8,4,2,4),\qquad d_h=d/H=2,
\tag{P2}
\]

并分别运行 attention-only 与 FFN-width-16。网络、RMSNorm、残差缩放和 MSE 与
Phase-I 相同。population risk 始终定义为

\[
R(\theta)=\frac12\mathbb E_U
\left[f_\theta(U)-Y\right]^2 .
\tag{P3}
\]

若代码保存的是 mean squared error `mse`，则必须验证

\[
R=\tfrac12\operatorname{mse}.
\tag{P4}
\]

不允许在同一图表或 gate 中混用 $R$ 与 MSE。

---

## 2. 函数级 routing 与 leakage

固定 $(c_{1:m},J)$，枚举 $v\in\{-1,+1\}^m$。Walsh coefficient 为

\[
\widehat f_S(c,J)
=2^{-m}\sum_v f(c,v,J)\prod_{i\in S}v_i .
\tag{P5}
\]

记

\[
\kappa_i=\widehat f_{\{i\}},\quad
E_T=\mathbb E(\kappa_J-1)^2,
\]

\[
L_D=\mathbb E\sum_{i\ne J}\kappa_i^2,\quad
L_H=\mathbb E\sum_{|S|\ge2}\widehat f_S^2,\quad
L_0=\mathbb E\widehat f_\varnothing^2 .
\tag{P6}
\]

本轮 leakage 是

\[
L_W=L_D+L_H+L_0,
\qquad
2R=E_T+L_W .
\tag{P7}
\]

令 $v^{\oplus J}$ 只翻转 queried value $v_J$。注册的 value-routing functional
effect 不是 accuracy 的别名，而是独立 intervention：

\[
\Xi_{value}
=\frac12\mathbb E\left[v_J\{f(c,v,J)-f(c,v^{\oplus J},J)\}\right]
=\mathbb E\,\kappa_J
=K_{target}.
\tag{P7a}
\]

实现必须分别从 flip intervention 与 Walsh singleton 计算式 (P7a) 的左右两端，并要求
absolute gap $<10^{-6}$。由 Cauchy--Schwarz，

\[
|1-\Xi_{value}|\le\sqrt{E_T}\le\sqrt{2R}.
\tag{P7b}
\]

不能把全部 Parseval error energy 与 $R$ 比衰减率，因为式 (P7) 会把该比较变成近似
恒等式。on-support distractor swap $U\mapsto U'$ 保持 $J,v,Y$ 不变，只把一个
$i\ne J$ 的 concept 替换为 support 外 concept：

\[
I_{\rm swap}
=\mathbb E\left[f(U')-f(U)\right]^2 .
\tag{P8}
\]

每个 checkpoint 还报告

\[
F_W=\frac{L_W}{2R+10^{-12}},
\qquad
F_{\rm swap}=\frac{I_{\rm swap}}{2R+10^{-12}} .
\tag{P9}
\]

---

## 3. 真正的直接 key-path selectivity

对每个 episode、每个 memory slot $i$，把所有层/head 的 final-query-to-slot-$i$
score 设为 $-\infty$，重算 softmax 和全部后代：

\[
\delta_i(U)
=Y\left[
f(U)-
f\!\left(\operatorname{do}
(s^{\ell h}_{Ti}=-\infty,\ \forall\ell,h)\right)
\right].
\tag{P10}
\]

确认性 estimand 是

\[
S_{\rm key}
=\mathbb E_U\left[
\delta_J-
\frac1{m-1}\sum_{i\ne J}\delta_i
\right].
\tag{P11}
\]

必须分别保存 $\mathbb E\delta_J$、平均 distractor effect 和二者之差。attention mass
只作描述量；target-edge effect 也不能替代式 (P11)。

---

## 4. 随机单位、CRN 与 evaluation

训练 seed $r$ 是独立推断单位。episode、value cube、swap pair、layer、head 和 token
都先在 seed 内聚合，不能充当独立样本量。

三个不重叠 cohort：

| cohort | seeds | 数量 | 用途 |
|---|---:|---:|---|
| discovery/remedy | 100–111 | 12 | debug、LR/scheduler、closure 候选 |
| untouched confirmation | 1000–1023 | 24 | headline confirmation |
| optimizer replication | 2000–2015 | 16 | plain/momentum SGD 复制 |

每个 master seed 分解成 counter-based streams：

\[
r\mapsto
(r_{init},r_{train},r_{eval},r_{Walsh},
r_{swap},r_{patch},r_{diag}).
\tag{P12}
\]

训练 episode 由 `(seed, step, episode_index)` 唯一决定。所有允许配对的 arms 使用相同
abstract episode。每 checkpoint：

- base episodes 至少 8192；
- 至少 512 个 concept-target skeleton，每个完整枚举 16 个 value assignments；
- finite localization 先用 2048 个固定 swap pairs；若 Monte Carlo gate 失败，只把
  evaluation 增至 4096、8192，不增加/筛选 seed；
- analysis 使用 float64；
- Parseval relative gap $<10^{-6}$；
- finite chord relative reconstruction gap $<10^{-5}$。

---

## 5. Matrix A：训练极限与 scheduler

所有新 seeds 先用 AdamW，$\eta=.003$，weight decay $=0$ 训练到 step 800，再从同一
完整 state（模型、optimizer、data stream）分叉：

| arm | schedule |
|---|---|
| constant-6400 | $\eta_s=.003$ |
| cosine-3200 | $.003[1+\cos(\pi(s-800)/2400)]/2$ |
| cosine-6400 | $.003[1+\cos(\pi(s-800)/5600)]/2$ |

checkpoint：

\[
s\in\{0,25,50,100,200,400,800,1200,1600,
2400,3200,4800,6400\}.
\tag{P13}
\]

对 $Z\in\{R,L_W,I_{\rm swap}\}$，每 seed 在
$s\in\{800,1600,3200,6400\}$ 拟合

\[
\log_2\max\{Z_r(s),10^{-8}\}
=a_{r,Z}-p_{r,Z}\log_2(s/800)+e .
\tag{P14}
\]

主要 rate differences：

\[
d_r^W=p_{r,L_W}-p_{r,R},
\qquad
d_r^{swap}=p_{r,I_{swap}}-p_{r,R}.
\tag{P15}
\]

“同速”需要 simultaneous 90% TOST CI 完全落在 $[-.25,.25]$；这表示训练步数每翻倍，
相对比例变化不超过 $2^{.25}=1.189$。若超过 20% seeds 碰到 $10^{-8}$ floor，rate
记为不可识别。

稳定残差还必须同时满足：

\[
q_{r,Z}=\log_2\frac{Z_r(6400)}{Z_r(3200)}
\]

的 simultaneous 90% CI 位于
$[-\log_2(1.25),\log_2(1.25)]$，且 final simultaneous 95% CI 下界高于

\[
\tau_{swap}=\tau_W=2.5\times10^{-3}.
\tag{P16}
\]

---

## 6. Matrix B：factorization conditioning 与容量

每个 head：

\[
B_h=Q_h^\top K_h,\qquad C_h=O_hV_h,
\qquad \operatorname{rank}(B_h),\operatorname{rank}(C_h)\le d_h .
\tag{P17}
\]

运行三个版本：

1. raw factorized $Q/K,O/V$；
2. dense direct $B_h,C_h\in\mathbb R^{d\times d}$：rank 和参数增加，只是
   capacity+conditioning upper bound；
3. rank-matched direct：Euclidean composite update 后以 truncated SVD retract 到
   rank $\le d_h$，作为相同函数类的 conditioning control。

另运行 $H=1,d_h=d$ 的 direct/factorized 容量等价校准。所有 arms 从同一初始函数开始：

\[
B_h(0)=Q_h(0)^\top K_h(0),\qquad
C_h(0)=O_h(0)V_h(0),
\tag{P18}
\]

step-0 predictions 的 max absolute gap 必须 $<10^{-6}$。

主要 endpoint：

\[
\Delta_Z^{rank}
=\mathbb E_r\log_2
\frac{Z_r^{rank-direct}(6400)}
{Z_r^{factorized}(6400)},
\quad Z\in\{L_W,I_{swap}\}.
\tag{P19}
\]

称为 conditioning remedy 必须：

- risk noninferiority upper 90% CI $<.01$；
- accuracy / value routing noninferiority margins 为 $.02/.05$；
- 至少一个 residual 降低两倍：式 (P19) simultaneous 95% CI 上界 $<0$ 且 point
  estimate $<-1$；
- 至少 80% seeds 通过 function gate。

只有 dense direct 修复而 rank-matched 不修复时，结论是 rank/function capacity，不是
纯优化几何。

---

## 7. Matrix C：表示来源

$C=32,d=8$ 的 Welch bound：

\[
\mu_W=
\sqrt{\frac{C-d}{d(C-1)}}\approx0.3111 .
\tag{P20}
\]

hard cell 运行：

- random Gaussian unit-row $E_0$，learned；
- 完全相同 $E_0$，fixed；
- fixed low-coherence frame；
- 完全相同 low-coherence frame，learned。

另以 $C=d=8$ 运行真正 orthogonal fixed-$E$ negative calibration；它改变 concept load，
不能与 $C=32$ 作单因素比较。

low-coherence frame 在解盲前生成并冻结，四个 frame replicas。验收：

\[
\mu(E)\le1.25\mu_W\approx0.389,
\]

\[
\frac{\|E^\top E-(C/d)I\|_F}
{\|(C/d)I\|_F}\le.02.
\tag{P21}
\]

所有 $E$ 匹配 row norm 与 Frobenius scale。主要 paired contrasts：

\[
\Delta_Z^{coh}
=\mathbb E_r\log_2
\frac{Z_r(fixed\ low\ coherence)}
{Z_r(fixed\ random)},
\]

\[
\Delta_Z^{learn}
=\mathbb E_r\log_2
\frac{Z_r(learned\ random)}
{Z_r(fixed\ random)} .
\tag{P22}
\]

低相干修复同样要求功能非劣和至少 2× residual reduction。

---

## 8. Matrix D：head capacity

令 residual width 为 $d$，attention inner width 为 $p=Hd_h$。每层 bias-free
Q/K/V/O 参数数

\[
P_{att}=4dp .
\tag{P23}
\]

标准 MHA 的 $p=d$，所以“固定 $d$”已经固定 attention 参数量；二者不是两个对照。
运行：

| family | $H$ | $p$ | $d_h$ | FFN | 解释 |
|---|---:|---:|---:|---:|---|
| A fixed residual/standard | 1,2,4,8 | 8 | 8,4,2,1 | fixed | heads 增多且每头变窄 |
| B fixed per-head | 1,2,4,8 | $2H$ | 2 | fixed | 总 attention channels 增加 |
| C fixed budget | 1,2,4,8 | $2H$ | 2 | $r=36,32,24,8$ | attention/FFN allocation |

Family C 使用 bias-free FFN，并令

\[
4dp+2dr=2d(2p+r),\qquad 2p+r=40,
\tag{P24}
\]

使每层 attention+FFN weight count 相同。它只能回答固定预算的容量分配，不能称为纯 head
数效应。

对 family $g$、seed $r$ 拟合

\[
\log_2 Z_{r,g,H}=a_{r,g}+\beta_{r,g}\log_2H+\epsilon ,
\tag{P25}
\]

主要 bottleneck interaction：

\[
\Gamma_{bottleneck}=\mathbb E(\beta_A-\beta_B).
\tag{P26}
\]

practical threshold 为 $.25$，directional simultaneous 95% CI 下界须 $>0$。

---

## 9. Matrix E：finite module localization

所有 site 使用相同 2048 on-support swap pairs，并保存 episode-level sidecar：

`config_hash, seed, step, episode_id, layer, head, site, direction, raw_suffix_delta,
input_energy, target_label, swap_slot, donor_concept`。

### 9.1 非对称 QK chord

\[
\delta m_C=\sum_i a_i(z'_i-z_i),\quad
\delta m_R=\sum_i(a'_i-a_i)z_i,
\]

\[
\delta m_I=\sum_i(a'_i-a_i)(z'_i-z_i),
\quad
m'-m=\delta m_C+\delta m_R+\delta m_I .
\tag{P27}
\]

令 $u_{h,p}=L^{-1/2}C_h\delta m_{h,p}$。从该 attention residual 的 base state $z$
开始，用真实 nonlinear suffix $G_{\ell,e}$：

\[
p_{C+I}=G(z+\sum_h(u_{h,C}+u_{h,I}))-G(z),
\]

\[
p_{C+R+I}=G(z+\sum_h(u_{h,C}+u_{h,R}+u_{h,I}))-G(z).
\tag{P28}
\]

finite QK contrast：

\[
C^{finite}_{QK,\ell,r}
=\mathbb E_e\log
\frac{p_{C+I,e}^2+10^{-12}}
{p_{C+R+I,e}^2+10^{-12}}.
\tag{P29}
\]

同时保存 tangent $t_{h,p}=r^\top u_{h,p}$ 和 opposition rate。禁止把 midpoint split
重命名成式 (P29)。

### 9.2 OV 方向选择

\[
g_{swap}=\frac{\|C_h\delta m_h\|^2}
{\|\delta m_h\|^2+10^{-12}},\qquad
g_{iso}=\frac{\|C_h\|_F^2}{d},
\]

\[
A_{OV}=\mathbb E\log\frac{g_{iso}+10^{-12}}{g_{swap}+10^{-12}} .
\tag{P30}
\]

训练诱导结论使用 $\Delta A_{OV}=A_{OV}^{final}-A_{OV}^{init}$。pre/post-OV coherent
patch 差只作等价性测试。

### 9.3 FFN signed energy 与 finite suffix

\[
\delta x_{skip}=x'-x,
\qquad
\delta x_{ffn}=L^{-1/2}[F(N(x'))-F(N(x))].
\tag{P31}
\]

\[
t_{skip}=r^\top\delta x_{skip},\quad
t_{ffn}=r^\top\delta x_{ffn},
\]

\[
C_{FFN}=\mathbb E\log
\frac{t_{skip}^2+10^{-12}}
{(t_{skip}+t_{ffn})^2+10^{-12}}.
\tag{P32}
\]

真实 suffix 还保存 $p_{skip},p_{ffn},p_{joint}$ 与

\[
p_{nonlin}=p_{joint}-p_{skip}-p_{ffn}.
\tag{P33}
\]

任一 module/layer 只有同时满足以下条件才称 compensator：

1. upstream finite energy $\ge10^{-4}$；
2. tangent 与 finite 同方向；
3. 至少 60% pairs 同方向；
4. module×layer simultaneous 95% CI 在抑制方向；
5. energy attenuation 至少 20%，即 contrast $\ge\log(1.25)$；
6. functional gate 通过；
7. 第二 optimizer 或 architecture 复制。

---

## 10. Matrix F：exact population GF-like bridge

完整 population 大小

\[
|\Omega|=\frac{C!}{(C-m)!}\,m\,2^m .
\tag{P34}
\]

注册 $(C,m)=(4,2)$ 的 96 states 和 $(6,3)$ 的 2880 states。reference dynamics：

\[
\theta_{k+1}=\theta_k-\eta\nabla R(\theta_k).
\tag{P35}
\]

初始步长按规则

\[
\eta_0=\min\left\{.003,
\frac{.25}{\lambda_{max}(H_{\theta_0})+10^{-12}}\right\},
\tag{P36}
\]

并运行 $\eta_0,\eta_0/2,\eta_0/4$，按 $s=k\eta$ 对齐。order parameters：

\[
z=(R,K_{target},L_D,L_H,\Xi_{value},S_{key},
r_{eff}(E),\|B\|_F,\|C\|_F,
S_Q-S_K,S_O-S_V).
\tag{P37}
\]

step-halving discrepancy：

\[
D_z=
\frac{\{\sum_s[z_\eta(s)-z_{\eta/2}(s)]^2\}^{1/2}}
{\{\sum_s[z_{\eta/2}(s)-z_0]^2\}^{1/2}+10^{-12}} .
\tag{P38}
\]

所有注册 $z$ 均满足 $D_z\le.10$ 才称 GF-like。AdamW 是不同的 adaptive
preconditioned dynamics，偏离 Euclidean GF 不是 closure 反例。

**P39-A 坐标修正（2026-08-20，任何 closure 拟合或 untouched closure 输出读取前）：**
预生产审阅指出原式没有指定量纲不同的 order parameters 如何加权。只在 discovery
seeds 上定义

\[
\mu_j=\mathbb E_D z_j,\qquad
\sigma_j=
\left\{\mathbb E_D(z_j-\mu_j)^2\right\}^{1/2},qquad
\widetilde z_j=\frac{z_j-\mu_j}{\sigma_j},
\tag{P39a}
\]

其中经验 $\sigma_j\le10^{-12}$ 的 discovery-constant coordinate 固定取
$\sigma_j=1$。经验 vector field $F_\phi(\widetilde z)$、ridge 和所有 scaler
只在 discovery seeds 拟合；untouched seeds 不参与任何选择。注册 gate 为

\[
E_{closure}^{std}=
\frac{\sum_U\|
\dot{\widetilde z}-F_\phi(\widetilde z)\|^2}
{\sum_U\|
\dot{\widetilde z}-\overline{\dot{\widetilde z}}_D\|^2}
\le.10 ,
\tag{P39}
\]

其中 $U$ 表示 untouched points，分母 baseline 是 discovery seeds 的平均标准化速度。
同时必须用同一个 fitted field 报告、不设通过阈值的 raw-coordinate sensitivity：

\[
E_{closure}^{raw}=
\frac{\sum_U\|
\dot z-D_\sigma F_\phi(\widetilde z)\|^2}
{\sum_U\|
\dot z-D_\sigma\overline{\dot{\widetilde z}}_D\|^2},
\qquad D_\sigma=\operatorname{diag}(\sigma_1,\ldots,\sigma_p).
\tag{P39b}
\]

这个修正改变坐标权重，不改变 source trajectories、P38 gate 或 untouched seed。若标准化
与 raw sensitivity 给出冲突，必须公开冲突，不能事后选择较好者。

否则只能叫低维描述；应寻找相同 $z$、不同 $\dot z$ 的 closure counterexample。

---

## 11. 预训练模型 bridge

首个真实模型族为 EleutherAI Pythia。官方模型卡给出每个 run 154 个训练 checkpoint，
许可证为 Apache-2.0。这里预先把三种不同统计对象拆开，禁止把它们混成“很多
checkpoints/seeds”：

1. **Suite A（单轨迹机制描述）：** `EleutherAI/pythia-70m-deduped`，固定 revisions
   $\{\text{step0},64,512,1000,4000,16000,64000,143000\}$。这是一条训练轨迹；
   checkpoint 不是独立样本，所有区间只描述注册 prompt population 的有限枚举误差，
   不对“预训练随机性”作推断。
2. **Suite B（多预训练 seed 复制）：** 标准、非 deduplicated Pythia-70M：官方原 run
   `EleutherAI/pythia-70m`（seed 1234）加 `pythia-70m-seed1` 至 `seed9`，共
   $N=10$ 条独立 pretraining runs。冻结 revisions
   $\{\text{step0},1000,10000,50000,143000\}$。seed 是推断单位；同一 seed 内所有
   checkpoint、template、episode 用 common random prompts 配对，studentized
   seed-block max-$T$ 同时校正 checkpoint$\times$template。不得把 Suite A 的 deduplicated
   run 与 Suite B 合并。
3. **Suite C（随机性来源拆分，探索性）：** Pythia-160M 的 `data-seed1..3` 与
   `weight-seed1..3`。前者只改变 data order，后者只改变 initialization；每组只有
   $N=3$，因此只报告逐 seed 轨迹、效应量和区间，不作确认性总体结论。

上述命名、seed 结构与 checkpoint 数由 PolyPythias 官方模型卡和仓库冻结：
https://huggingface.co/EleutherAI/pythia-70m-seed3 ，
https://github.com/EleutherAI/pythia 。

对两个 answer strings $+$ 与 $-$ 的条件 log likelihood $\ell_+,\ell_-$，定义有界输出

\[
f_\theta(X)=
\frac{e^{\ell_+}-e^{\ell_-}}
{e^{\ell_+}+e^{\ell_-}}
=\tanh\frac{\ell_+-\ell_-}{2}\in[-1,1].
\tag{P40}
\]

这样式 (P3)–(P11) 可原样迁移。每个 prompt：

- 四个 distinct concept-value pairs 与一个 target query；
- values 是 episode 内新随机二元 labels；
- distractor swap 替换为未出现、token-length-matched concept；
- base/donor 的所有其他 tokens 和 positions 完全相同；
- 每个 skeleton 枚举全部 16 个 values；
- 四个 prompt templates 在 test data 前冻结并逐 template 报告。

代码与 tokenizer 的首轮校准固定为 16 skeletons/template，只能验证运行时间、token
alignment、Parseval/replay/causal-mask identity，**不能进入论文统计表或用于选择 template**。
生产评估至少 512 skeletons/template；四个 templates 和 concept pool 在读取生产输出前冻结。
对 Suite A，checkpoint/size 仍不是训练 seed；对 Suite B，唯一推断单位是预训练 seed。
若后续 fine-tune：

- $\le500$M 至少 12 independent fine-tuning seeds；
- 1–3B 至少 10 seeds；
- 少于 10 seeds 只作 descriptive bridge。

开源模型的 $S_{key}$ 仍逐 memory span 屏蔽 final decision position 到该 span 的直接
attention edges；它不覆盖经过中间 token 的间接路径。QK chord 只在 base/donor token
positions 一一对齐时计算。冻结模型只能报告绝对 OV/FFN 作用，不能说“训练诱导”；训练
诱导结论需要 checkpoint trajectory 或独立 fine-tuning trajectories。

---

## 12. Multiplicity、功能 gate 与失败 ladder

三个 headline families：

| family | endpoints | correction |
|---|---|---|
| residual etiology | Matrix A–D 的 rate/endpoint/remedy | studentized seed-block max-$T$ |
| localization | architecture×optimizer×module×layer | 100,000 Rademacher max-$T$ |
| GF bridge | 全部注册 order parameters | equivalence IUT / deviation max-$T$ |

若论文从三 families 中选择任一显著 headline，再对三 family-level $p$ 做 Holm。bootstrap
使用 20,000 paired whole-seed blocks；trajectory band 对 checkpoint 取 $\max_t|T_t^*|$。
未注册 spectra、head specialization、NTK、Hessian、landscape 使用 BH $q=.10$，并标为
exploratory。

功能 gate：

\[
A\ge.95,\quad R\le.01,\quad \Xi_{value}\ge.90.
\tag{P41}
\]

noninferiority margins：accuracy $.02$、$\Xi_{value}$ $.05$、risk $.01$。

固定 remedy 顺序：

1. support/replay/Walsh/chord/JVP 失败：停止 grid，修 instrumentation 并升 version；
2. MC 不足：evaluation 2048→4096→8192；
3. function gate 失败：800→1600→3200→6400；
4. constant 与预注册 cosine；
5. discovery seeds 上 LR $\{.00075,.0015,.003,.006\}$，只能依据 risk/NaN/gate 选择；
6. factorization controls；
7. fixed/low-coherence/learned $E$；
8. fixed-$d$/fixed-$d_h$/fixed-budget heads；
9. untouched confirmation seeds；
10. second optimizer/architecture；
11. 再次联网检索已知 optimizer、scheduler、normalization、precision 技术。

---

## 13. 升级成 empirical open problem 的必要条件

以下十二条全部满足才允许升级：

1. 至少 80% untouched confirmation seeds 通过式 (P41)；
2. step 6400 的 $L_W$ 或 $I_{swap}$ simultaneous 95% CI 下界 $>.0025$；
3. 3200→6400 通过 1.25× plateau equivalence；
4. constant/cosine 均未消除；
5. rank-matched direct 未消除；
6. fixed low-coherence $E$ 未消除；
7. fixed $d_h$ 与 fixed budget 未消除；
8. 两个 optimizer 且 no-FFN/FFN 同方向复制；
9. 式 (P11) 已真正逐 slot 测量；
10. QK 用非对称 chord，OV/FFN 有 finite suffix；
11. failure、MC、precision、replay 全部通过审计；
12. 现象发生后再次查重，现有方法仍不能解释。

对应分类规则：

- 更长训练/cosine 修复 → 未收敛；
- rank-matched direct 修复 → factorization optimization geometry；
- 只有 dense direct 修复 → rank/function capacity；
- low-coherence $E$ 修复 → dictionary collision；
- fixed $d_h$ 修复 → per-head bottleneck；
- fixed budget 改变 → capacity allocation；
- upstream cross-talk 本就接近零 → 下游补偿假说的反例；
- upstream 明显但无单模块定位 → distributed/non-identifiable compensation。

这些分类先于任何新 theorem 命名。
