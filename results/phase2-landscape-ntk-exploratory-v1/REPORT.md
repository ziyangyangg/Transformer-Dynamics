# Phase-II loss landscape、NTK 与表示叠加诊断

> **结论状态：探索性诊断，不是预注册验证。** 本报告只描述冻结 checkpoint 在固定 probe 上的几何与函数现象；它不能单独证明因果机制、定理或新的 open problem。

## 一句话说明

我们把参数分解本身造成的假平坦与真正的函数变化分开：loss plane 在 gauge-invariant 的复合映射 B=QᵀK、C=OV 中计算，raw factor gauge orbit 只作为应当完全平坦的负对照；同时在同一固定样本上跟踪 raw-coordinate empirical NTK、concept codebook 几何、query–target/distractor 几何和精确 Walsh leakage。

## 统计设计与样本边界

- 独立重复数 **N=3 个训练 seed**：100, 101, 102。
- 冻结训练时刻：0, 800, 3200, 6400；checkpoint 和平面网格点都不是独立重复。
- empirical NTK probe：固定 24 个 episode，seed=820202601。
- 表示与 loss-plane probe：固定 256 个 episode，seed=820202602。
- 图中的细线是 seed，粗线是 seed 均值，阴影是 observed min–max；没有把它画成置信区间。

比较的训练臂：

- H4 fact. constant：hard-factorized-constant-6400
- H4 fact. cosine：hard-factorized-cosine-6400
- H4 rank-direct：hard-rank-matched-constant-6400
- H4 dense-direct：hard-dense-direct-constant-6400
- H1 factorized：h1-factorized-constant-6400

## 数学对象

### 1. 复合函数坐标中的 loss plane

每层每头只通过下列复合映射决定 attention score 与 value transport：

$$B_{\ell h}=Q_{\ell h}^{\top}K_{\ell h},\qquad C_{\ell h}=O_{\ell h}V_{\ell h}.$$

在 checkpoint t，令 D_t 是相邻真实 checkpoint 的复合位移。step 0 使用指向 step 800 的 outgoing 位移；其余时刻使用来自前一 checkpoint 的 incoming 位移。对每个 (层, 头, B/C) 矩阵独立生成 U_t，并执行

$$\langle D_{t,\ell hm},U_{t,\ell hm}\rangle_F=0,\qquad \|U_{t,\ell hm}\|_F=\|D_{t,\ell hm}\|_F.$$

随后在同一个固定 probe 上计算

$$\mathcal R_t(\alpha,\beta)=\frac{1}{2n}\sum_{i=1}^{n}\left[f_{M_t+\alpha D_t+\beta U_t}(x_i)-y_i\right]^2,$$

其中 M_t 收集全部 B 和 C。factorized 与 rank-matched 臂在中心满足原函数精确等价，但离开中心后这个 ambient plane 可能越出 rank≤d_h 的可实现集合；因此它不是 constrained basin volume。

### 2. factor gauge-flat 负对照

对 factorized attention 使用可逆 G(t)=exp(tS)：

$$Q\mapsto GQ,\quad K\mapsto G^{-\top}K,\quad O\mapsto OG^{-1},\quad V\mapsto GV.$$

于是 B 和 C 理论上完全不变。raw 参数可以移动很远，但预测与风险必须只在浮点误差范围内变化；这说明 raw-factor plane 的平坦方向不能直接解释为宽 basin。

### 3. empirical NTK

对参数组 g∈{full,E,QK,OV,readout}，固定 probe 输出向量的 Jacobian 为 J_g，定义

$$K_g(t)=\frac{1}{P_g}J_g(t)J_g(t)^{\top}.$$

相对漂移、alignment 与 participation effective rank 分别为

$$\Delta_g(t)=\frac{\|K_g(t)-K_g(0)\|_F}{\|K_g(0)\|_F},\qquad A_g(t)=\frac{\langle K_g(t),K_g(0)\rangle_F}{\|K_g(t)\|_F\|K_g(0)\|_F},$$

$$r_{\mathrm{eff}}(K_g)=\frac{(\operatorname{tr}K_g)^2}{\operatorname{tr}(K_g^2)}.$$

这些量位于 raw trainable coordinates。臂内随时间的漂移可解释；不同 parameterization 之间的绝对 kernel 尺度不是 gauge-invariant 结论。

### 4. learned superposition 与功能 leakage

concept dictionary E 的行归一化 Gram 矩阵 G_E 给出 coherence μ=max_{c≠c'}|G_{E,cc'}|。另令 σ_j(E) 是原始 E（不是 G_E）的奇异值，则 participation rank 为 r_E=(Σ_jσ_j(E)²)²/(Σ_jσ_j(E)⁴)。残差流中另测 query–target cosine 与平均 query–distractor cosine 的差。精确 Walsh leakage L_W 沿用 float64 precision supplement 的完整枚举值，而不是固定 probe 近似。

## step 6400 的 seed-level 结果

表中均为 mean [observed min, observed max]，N 只等于训练 seed 数。

| arm | population risk | Walsh L_W | I_swap | codebook μ | codebook rank | target−distractor cosine | full NTK drift | full NTK alignment |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| H4 fact. constant | 0.001699 [0.0006917, 0.002989] | 0.002996 [0.001271, 0.005281] | 0.004613 [0.0001606, 0.0115] | 0.9288 [0.898, 0.9442] | 4.367 [3.835, 5.002] | 0.8973 [0.6607, 1.02] | 0.9796 [0.9766, 0.9852] | 0.2233 [0.2136, 0.2406] |
| H4 fact. cosine | 0.0004024 [7.743e-05, 0.0007788] | 0.0007208 [0.0001311, 0.001402] | 0.0006041 [0.0003534, 0.0008208] | 0.9224 [0.8978, 0.9482] | 4.411 [4.05, 4.93] | 0.9222 [0.7298, 1.02] | 0.9784 [0.9741, 0.9849] | 0.2258 [0.2178, 0.2333] |
| H4 rank-direct | 0.003996 [0.001767, 0.005896] | 0.006765 [0.003028, 0.01022] | 0.007417 [0.003374, 0.01191] | 0.9587 [0.9271, 0.9749] | 3.195 [2.534, 3.756] | 0.8476 [0.6986, 0.9325] | 1.145 [0.9781, 1.467] | 0.213 [0.1558, 0.2695] |
| H4 dense-direct | 5.076e-06 [2.928e-06, 6.595e-06] | 6.18e-06 [3.466e-06, 8.764e-06] | 5.324e-06 [2.83e-06, 6.796e-06] | 0.9233 [0.8628, 0.9652] | 4.932 [4.641, 5.128] | 1.029 [1.027, 1.033] | 0.9797 [0.9744, 0.9881] | 0.2182 [0.189, 0.2365] |
| H1 factorized | 3.899e-06 [2.02e-06, 6.378e-06] | 5.626e-06 [2.765e-06, 9.269e-06] | 2.96e-06 [2.513e-06, 3.247e-06] | 0.94 [0.9071, 0.9713] | 4.846 [4.527, 5.084] | 1.024 [0.9999, 1.036] | 1.003 [0.9818, 1.023] | 0.1782 [0.1638, 0.1898] |

## 与 H4 factorized constant 的配对终点对照

同一 seed 配对后报告 log2(对照臂/基线)。正值表示对照臂更大；这是 N 个配对 seed 的描述统计，不做小样本显著性声明。
P19 的 -1 阈值只适用于 L_W 与 I_swap；population risk 只承担 noninferiority guardrail，不能套用两倍改善线。

| arm | log2 risk ratio | log2 Walsh ratio | log2 swap ratio |
|---|---:|---:|---:|
| H4 fact. cosine | -2.371 [-3.159, -1.94] | -2.382 [-3.278, -1.913] | -1.481 [-4.171, 2.354] |
| H4 rank-direct | 1.315 [0.3194, 2.644] | 1.245 [0.3134, 2.469] | 2.04 [0.05101, 4.393] |
| H4 dense-direct | -8.221 [-8.918, -6.922] | -8.782 [-9.709, -7.181] | -8.325 [-10.82, -4.563] |
| H1 factorized | -8.679 [-10.53, -6.761] | -8.991 [-10.9, -7.1] | -9.079 [-12.16, -5.686] |

## 轨迹相关性（仅描述）

每个 arm×seed 先在 checkpoint 轨迹内计算 Spearman ρ；随后先在同一 master seed 内对可用臂取均值，再跨 seed 汇总，因此总计数仍是 N，而不是 arms×N。每臂各 N 个 seed 的描述统计保存在 summary.json。没有把 checkpoint 当成独立样本，也不报告 checkpoint-level p-value。

| trajectory variable vs Walsh L_W | master-seed arm-mean [min, max] ρ |
|---|---:|
| codebook_coherence | -0.5067 [-0.84, -0.2] |
| codebook_effective_rank | 0.7867 [0.56, 1] |
| final_query_target_minus_distractor_cosine | -0.8933 [-1, -0.72] |
| ntk_full_relative_drift | -0.84 [-0.96, -0.6] |

## 数值正确性审计

- dense composite proxy 中心预测最大误差：0.000e+00。
- 每个 B/C map 的训练轴–随机轴最大绝对 cosine：9.496e-17。
- 每个 B/C map 的轴范数最大相对差：2.567e-16。
- gauge orbit 最大 composite / prediction / risk gap：8.882e-16 / 2.698e-14 / 4.441e-16。

## 这些图能回答什么

1. Figure 1 判断真实训练复合位移附近是否比同尺度正交方向更陡、是否仍存在下降方向；它直接对应 routing composite 的函数几何，而不是 factor gauge。
2. Figure 2 判断训练是否处在接近初始化 kernel 的 lazy regime，并把漂移定位到 E、QK、OV 或 readout；但 raw-coordinate NTK 不能用来宣称跨 parameterization 的绝对优劣。
3. Figure 3 把 learned codebook superposition、残差 query routing 几何和精确 Walsh leakage 放在同一训练轨迹中；相关轨迹只产生 theorem candidate，不等于干预式因果定位。
4. Figure 4 是关键负对照：若 raw 参数明显移动而 B、C、预测和 loss 不变，任何 raw-factor flatness 都必须先扣除 gauge 冗余。

## 不能据此声称什么

- 不能把二维 plane 当成整个高维 basin 的体积、Hessian 谱或优化可达概率。
- 不能把 ambient composite plane 离开中心的点视为 rank-limited 臂可实现的模型。
- 不能从 observational checkpoint correlation 推断 E、QK 或 OV 对 leakage 的因果效应。
- N=3 只支持探索性重复；稳定机制仍需新 seed、预注册干预和 population-GF bridge。
- 数值异常必须先按现有优化与测量技术排查，不能直接升级成 open problem。

## 可复现材料

- checkpoint_diagnostics.csv：每个 arm×seed×step 的合并诊断。
- ntk_metrics.csv 与 numeric/**/ntk_kernels.npz：五组 kernel 指标与原矩阵。
- landscape_index.csv、landscape_points.csv 与 numeric/**/composite_loss_plane.npz：轴定义、审计与完整平面。
- representation_geometry.csv：codebook 和每个残差位置的表示几何。
- gauge_orbit.csv 与 numeric/**/factor_gauge_orbit.npz：factor gauge 负对照。
- summary.json、manifest.json 和 _SUCCESS：机器可读结论、源码/checkpoint hash 与原子完成标记。

从仓库根目录运行：

    PYTHONPATH=src python -m routing_lab.phase2_landscape_ntk_study --config configs/phase2_landscape_ntk_exploratory_v1.json --source-directory results/phase2-residual-factorization-noffn-discovery-remedy-v2 --precision-audit-directory results/phase2-residual-factorization-noffn-precision-audit-v2 --output-directory results/phase2-landscape-ntk-exploratory-v1 --device cuda
