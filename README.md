# Transformer Routing & Superposition Lab

一个面向 **causal Transformer training dynamics、learned compressed representations、
QK/OV/FFN 机制定位** 的可复现实验室。项目不把 accuracy、attention 热图或低秩单独
当作解释；每一层结论都有独立的数学对象、干预和统计门槛。

## 我们研究什么

每个 episode 包含 `m` 张记忆卡和一个 query：

```text
(concept_1, value_1), ..., (concept_m, value_m), query
```

concept 互不相同，`value_i` 每题重新独立采样为 `-1/+1`，query 等于其中一个
concept，标签是对应的随机 value。因为 value 每题重采样，网络不能把答案存进 concept
embedding；它必须在当前序列中找到正确卡片并读取 value。

理论对象是有限 causal Transformer 的 population gradient flow：

```text
R(theta) = 1/2 E[(f_theta(X)-Y)^2]
d theta_s / ds = -grad R(theta_s)
```

实验用 fresh online batches 近似 population expectation，并分别训练 learned embedding
`E`、factorized `Q/K`、`O/V`、可选 FFN 和 readout。

本项目严格区分：

1. **函数级 causal routing**：输出是否真正依赖 queried value；
2. **参数级训练选择**：联合训练选择了什么 `E + QK + OV + FFN + readout` 分解；
3. **compressed dictionary geometry**：`C>d` 时 concept vectors 必然非正交；
4. **activation superposition**：同一 hidden state 是否同时承载多个可独立解码特征；
5. **downstream compensation**：上游 on-support cross-talk 是否被某个具体模块有限地消除。

第 3 项不自动推出第 4 项，最终 accuracy 也不自动推出第 5 项。

## 最重要的数学接口

### 端到端 value causal kernel

```text
kappa_i = 1/2 [f(do(v_i=+1)) - f(do(v_i=-1))]
```

完整 Boolean-cube Walsh 分解给出精确 Parseval 恒等式：低 population risk 强迫 target
slot 的一阶系数趋近 1，并压低 distractor 与高阶交互能量。这证明的是**函数层必要条件**，
不是特定 attention head 的唯一性。

### 复合矩阵

```text
B_lh = Q_lh^T K_lh       # routing / score geometry
C_lh = O_lh V_lh         # value-to-residual map
```

代码同时保存 raw factors 与 gauge-invariant composites。factorized gradient flow 的闭合
方程和可证命题/反例目标见 [THEORY_PROBLEMS.md](reports/THEORY_PROBLEMS.md)。

### on-support distractor swap

只把一个非 target concept 换成当前 memory 中不存在的新 concept；values、query、target
index 与 label 均不变。两个端点都来自原始数据分布，因此它比任意 off-manifold 向量扰动
更适合检验真实 cross-talk。

### 模块局部化

项目记录并干预：query QK scores、attention probabilities、pre/post-OV、post-attention
residual、FFN branch、post-FFN residual 与 prediction。有限 attention chord 被精确分成
content 与 route 两项；tangent 版本同时对照手算、autograd JVP 和中心差分。

## 研究边界与当前结论

- 固定权重的 sphere clustering 已按 Perspective 官方实现复现；它产生 global token
  collapse，但终点 attention 是均匀的，因此 **global clustering 不等于 selective causal
  routing**。
- 在成功模型中，Walsh target coefficient 和 value-flip effect 接近 1，支持函数级
  composite routing。
- “QK route 抑制 content cross-talk”这一具体假设在两个优化器上方向相反：route 总体
  放大而非抑制该有限 chord，因此被保留为负结果。
- OV 学到稳定的 target-value 相对 distractor-concept 方向选择性；这是候选机制，不是
  已证明的 causal compensation。
- FFN 局部 signed cancellation 尚未通过全部 energy-floor、finite on-support 与跨优化器
  门槛；confirmed compensator 数量仍为 0。
- 高学习率下的若干 plateau 可用更小学习率和更长训练修复，因此不被包装成新的容量
  open problem。

完整、带限定词的结论请从 [最终研究报告](reports/FINAL_REPORT.md) 开始阅读；截至
2026-08-15 的逐篇文献查重在 [LITERATURE_MAP.md](reports/LITERATURE_MAP.md)。

## 目录

```text
src/routing_lab/
  data.py                   数据分布与 on-support swap
  model.py                  可干预 causal Transformer
  metrics.py                Walsh、causal 与表示容量指标
  tangent.py                attention JVP 的 content/route 分解
  diagnostics.py            finite QK/OV/FFN 模块诊断
  interventions.py          value/key/activation interventions
  training.py               可复现 online training 与 checkpoint
  run.py                    crash-safe grid runner
  evaluate.py               单 snapshot 机制评估
  study_evaluate.py         整个训练轨迹的只读机制重放
  dynamics.py               NTK、线性化、Hessian 与 loss surface
  dynamics_study.py         snapshot dynamics runner

configs/                    不可变实验网格
tests/                      手算、解析、有限差分与端到端契约
results/                    原始轨迹、派生统计与 PNG/SVG 图
reports/                    理论、文献、协议和结果解释
autoresearch/               keep/discard 迭代审计账本
third_party/                固定 commit 的官方 clustering 代码
```

## 环境

- Python 3.11+
- PyTorch
- NumPy / SciPy / pandas
- Matplotlib

安装为 editable package：

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -e .
```

如已有兼容环境，也可以只设置：

```bash
export PYTHONPATH=src
```

所有随机过程都使用显式 seed/generator；实验 manifest 保存代码可见的全部模型、优化、
checkpoint 和 evaluation 选择。

## 从零验证

### 1. 全部测试

```bash
PYTHONPATH=src python -m unittest discover -s tests -v
```

测试覆盖：

- 数据支持与 label-preserving swap；
- causal mask 与 patch 后代重算；
- `Q^T K`、`OV` 矩阵方向；
- Walsh/Parseval 手算；
- finite chord = content + route；
- JVP = 中心差分；
- checkpoint byte-stable resume；
- seed-level bootstrap、max-T 与 TOST；
- NTK、HVP、Lanczos 和 filter-normalized landscape 的解析小例子。

### 2. 快速训练 smoke

```bash
PYTHONPATH=src python -m routing_lab.run \
  --config configs/smoke.json \
  --output results/smoke-reproduction \
  --device cpu
```

### 3. 主实验

```bash
# AdamW: C x H x FFN, 8 cells x 12 seeds
PYTHONPATH=src python -m routing_lab.run \
  --config configs/primary_adamw.json \
  --output results/primary-adamw-reproduction \
  --device cuda

# Momentum-SGD optimizer replication
PYTHONPATH=src python -m routing_lab.run \
  --config configs/replication_sgd.json \
  --output results/replication-sgd-reproduction \
  --device cuda

# Width x load x heads x FFN, tuned 16 cells x 10 seeds
PYTHONPATH=src python -m routing_lab.run \
  --config configs/scaling_tuned_adamw.json \
  --output results/scaling-tuned-reproduction \
  --device cuda
```

Runner 以 `cell -> seed` 顺序执行；每个 seed 用 `_SUCCESS` 原子提交。中断后原命令可安全
重跑，已完成 seed 不会重复写入。

### 4. 固定 held-out episodes 重放机制

```bash
PYTHONPATH=src python -m routing_lab.study_evaluate \
  --run-directory results/primary-adamw-reproduction \
  --output-directory results/primary-mechanisms \
  --selected-steps 0 50 100 400 \
  --evaluation-batch-size 512 \
  --evaluation-seed-offset 900000 \
  --device cuda
```

同一个 training seed 在不同 step 使用完全相同的 held-out batch 与 swap stream，因此轨迹
变化不会混入新的 Monte Carlo 样本。

### 5. NTK / Hessian / loss landscape

```bash
PYTHONPATH=src python -m routing_lab.dynamics_study --help
```

该 runner 对指定 cell/seed/steps 使用同一个 probe，输出 full 与 E/QK/OV/FFN/readout
group NTK、初始化线性化误差、Lanczos/Hutchinson Hessian 诊断，以及二维
filter-normalized loss surface。生产参数和已运行个案见
[DYNAMICS_RESULTS.md](reports/DYNAMICS_RESULTS.md)。

## 统计规则

- 独立统计单位是 training seed，不是 layer、head、checkpoint 或 episode。
- 主要区间使用 20,000 次 whole-seed paired bootstrap。
- optimizer 与 architecture replication 是 robustness gate。
- 失败 seed、NaN、门槛失败和调优尝试全部保存在 ledger。
- attention、cosine、NTK 和 landscape 是描述/局部诊断；只有显式替换结构方程并重算
  后代的量称为 causal intervention。

## 推荐阅读顺序

1. [FINAL_REPORT.md](reports/FINAL_REPORT.md)：做了什么、结果和下一步；
2. [THEORY_PROBLEMS.md](reports/THEORY_PROBLEMS.md)：完整模型、动力学、定理/反例目标；
3. [LITERATURE_MAP.md](reports/LITERATURE_MAP.md)：原始 open problems 与 2026-08-15 边界；
4. [ANALYSIS_PROTOCOL.md](reports/ANALYSIS_PROTOCOL.md)：统计与 claim ladder；
5. [MECHANISM_RESULTS_DRAFT.md](reports/MECHANISM_RESULTS_DRAFT.md)：QK/OV/FFN 负结果与候选；
6. [CLUSTERING_BASELINE.md](reports/CLUSTERING_BASELINE.md)：Perspective baseline；
7. [DYNAMICS_RESULTS.md](reports/DYNAMICS_RESULTS.md)：loss landscape、NTK 与 Hessian 个案。

## 可复现性边界

仓库保存代码、配置、seed-level 聚合轨迹、机制表和派生图。模型 checkpoint 体积较大，
本地研究目录保留完整副本；GitHub 版本默认不提交 `*.pt`。所有 checkpoint 都可由保存的
配置和 seed 重新生成。

