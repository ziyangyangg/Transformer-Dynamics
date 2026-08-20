# Phase II 结果报告模板：从 residual 诊断到可证明问题

> 这个模板只用于已经通过 `phase2_results.py` 完整性审计的结果目录。任何数字都必须能回指
> `analysis_summary.json` 或 seed-level CSV；不能从图上手工抄录，也不能把 checkpoint、head、
> slot、episode 当作独立样本。

## 0. 结果身份

- cohort：`<discovery-remedy / untouched-confirmation / optimizer-replication>`
- study config hashes：`<hashes>`
- 独立训练 seeds：`<seed list>`
- checkpoint 主键：
  `(study_config_hash, cell_hash, seed, step)`
- bootstrap：20,000 次 whole-seed blocks；RNG seed `<seed>`
- 完整性审计：root config hash、root/seed manifest、checkpoint schedule、seed table union
  `<passed / failed>`

这里首先写“哪些数据真正完成”，而不是先写最显著的现象。若 `_SUCCESS`、某个 seed
manifest、某个 checkpoint state 或 root/seed row identity 不一致，整项研究标为未完成。

## 1. 训练极限：residual 是否只是尚未收敛

对 constant-6400 与 cosine-6400 分开报告

\[
\log_2\max\{Z_r(s),10^{-8}\}
=a_{r,Z}-p_{r,Z}\log_2(s/800)+e,
\quad Z\in\{R,L_W,I_{swap}\}.
\]

每个 arm 填写：

| endpoint | mean slope | simultaneous 90% CI | floor-hit seeds |
|---|---:|---:|---:|
| \(p_R\) | | | |
| \(p_{L_W}\) | | | |
| \(p_{I_{swap}}\) | | | |

同速检验填写：

| contrast | estimate | simultaneous 90% TOST CI | 完全位于 \([-0.25,0.25]\)? |
|---|---:|---:|---:|
| \(p_{L_W}-p_R\) | | | |
| \(p_{I_{swap}}-p_R\) | | | |

再分别报告：

- \(q_Z=\log_2[Z(6400)/Z(3200)]\) 的 plateau-equivalence；
- final \(L_W,I_{swap}\) 的 simultaneous 95% CI 是否高于 \(2.5\times10^{-3}\)；
- \(F_W=L_W/(2R+10^{-12})\) 与
  \(F_{swap}=I_{swap}/(2R+10^{-12})\) 的 seed 分布；
- constant 与 cosine 的同 seed slope 差。

**允许的最小结论：** `<继续下降 / 与 risk 同速 / 形成 practical plateau / rate 不可识别>`。
未同时通过同速、plateau、practical-floor 三道 gate 时，不写“稳定 residual”。

## 2. factorization conditioning 与 function capacity

三个对照必须按不同角色书写：

1. factorized \(Q/K,O/V\)：rank-limited baseline；
2. rank-matched direct \(B,C\)：相同 rank/function class 的 conditioning control；
3. dense direct \(B,C\)：capacity + conditioning upper bound，不是纯优化对照。

报告 step 6400 的 paired seed estimand

\[
\Delta_Z=\mathbb E_r\log_2
\frac{Z_r^{treatment}}{Z_r^{factorized}},
\quad Z\in\{R,L_W,I_{swap}\},
\]

以及 simultaneous 95% max-\(T\) CI。另列 H=1 dense/factorized 的 full-rank
capacity-equal calibration。

| comparison | endpoint | estimate | simultaneous 95% CI | 解释 |
|---|---|---:|---:|---|
| rank-matched / factorized | | | | conditioning candidate |
| dense / factorized | | | | capacity upper bound |
| H=1 dense / factorized | | | | parameterization calibration |

**强制 guardrail：** 只有 dense 修复时，写“rank/function capacity candidate”，不能写
“factorization optimization geometry”。checkpoint wide schema 同时保存 intervention
计算的 \(\Xi_{value}\)、Walsh singleton \(K_{target}\) 及二者恒等式误差。完整功能 gate
\(A\ge .95,R\le .01,\Xi_{value}\ge .90\) 在每个训练 seed 内判定；通过 gate 也不会
改变 rank-matched 与 dense-direct 两种对照不同的证据角色。

## 3. 表示来源：2×2 exploratory factorial

主 C=32 因子为：

- geometry：random / low-coherence；
- codebook：fixed / learned。

对 \(R,L_W,I_{swap}\) 使用 log2 scale；accuracy、\(S_{key}\)、coherence、effective rank
使用 raw scale。报告 geometry main、learning main、geometry×learning。所有 endpoints ×
contrasts 组成一个明确 family；paired-seed sign-flip p-values 以 BH \(q=0.10\) 调整，CI
仍明确标为 unadjusted pointwise 95% seed bootstrap。

C=8 orthogonal fixed-E 只作为 negative calibration，单列，绝不并入 C=32 factorial。

## 4. head capacity：三个不同问题

- Family A：固定 total attention width \(p=d\)，H 增加时每头变窄；
- Family B：固定 \(d_h=2\)，H 增加时总 attention width 增加；
- Family C：固定 attention+FFN weight budget，只回答容量分配。

每 seed、每 family 拟合

\[
\log_2 Z_{r,g,H}=a_{r,g}+\beta_{r,g}\log_2H+\epsilon,
\]

并报告 \(\Gamma=\mathbb E(\beta_A-\beta_B)\)。本轮所有 head factorial 数字标为
exploratory，使用输出中声明的 BH family，不把 Family C 命名成“纯 head 数效应”。

## 5. 图与统计的逐项核对

- `01_training_limit_same_rate`：薄线是 seed trajectory；实线/阴影是 seed mean 与 95% CI。
- `02_schedule_paired_slopes`：每条连接线是一对同 seed constant/cosine slopes；Δ 为
  simultaneous 95% max-\(T\) CI。
- `03_factorization_controls`：rank-matched、dense、H=1 calibration 分区；点为 seeds。
- `04_representation_geometry`：C=8 orthogonal 明确标作 negative calibration。
- `05_head_capacity_geometry`：A/B/C family 不合并；薄线为 seeds。

PNG 用于快速阅读；SVG 用于论文排版和逐元素审阅。两种格式应当出现在
`artifact_manifest.json` 并带内容 hash。

## 6. 结论阶梯

按下面顺序写，不能跳级：

1. **测量事实：** 哪个 seed-level estimand 如何变化，CI 是什么；
2. **最小机制解释：** convergence、conditioning、capacity、dictionary collision 或
   per-head bottleneck 中哪个仍与数据相容；
3. **已排除解释：** 只列真正运行并通过完整 gate 的 controls；
4. **仍缺的证据：** \(\Xi_{value}\)、finite module localization、第二 optimizer、
   untouched confirmation 等；
5. **论文问题资格：** 只有协议的 open-problem ladder 全部完成后才写“真正开放”。

最终摘要必须同时回答：实验做了什么、统计单位是什么、观察到什么、能说明什么、不能说明
什么、下一条最有信息量的实验或定理目标是什么。
