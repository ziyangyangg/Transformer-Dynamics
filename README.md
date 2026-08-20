# Transformer Routing & Superposition Lab

当前阶段的精简结论见 [STAGE_DECISION.md](reports/STAGE_DECISION.md)：主理论对象、已解决部分、反例边界与唯一下一实验均集中在该文件。Pythia-70M 的 8-checkpoint float64 校准见 [短报告](results/pretrained-pythia70m-suite-a-calibration-float64-v4-analysis-v1/REPORT.md)。

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
- 当前 target-key mask 只阻断 target edge，没有逐个阻断 distractor edges；因此注册的
  causal key selectivity $S_{key}$ 尚未评估，现有 target-edge + attention 量只是探索性
  screen。
- 对称 midpoint QK 诊断在两个优化器上不支持“route 抑制 content cross-talk”的简单
  故事；它与预注册的非对称 content/route/interaction estimand 不同，因此只是探索性
  protocol deviation，不能称为预注册反证。
- OV 学到稳定的 target-value 相对 distractor-concept 方向选择性；这是候选机制，不是
  已证明的 causal compensation。
- FFN 局部 signed cancellation 尚未通过全部 energy-floor、finite on-support 与跨优化器
  门槛；confirmed compensator 数量仍为 0。
- normalized embedding-rank 的 7 个 factorial contrasts 属于选择后的 secondary family；
  当前只有未做 BH/family correction 的 pointwise intervals，因此只作为 exploratory
  architecture patterns，不作为确认性架构效应。
- 高学习率造成的 base-retrieval plateau 可由 tuned schedule 修复，因此不被包装成容量
  open problem。另一个更严格的问题是高负载多头的 on-manifold cross-talk：保持
  LR=.003 延长到 1600 steps 后，未校正 pointwise 区间支持最困难 cells 3/7 明显改善，
  但仍未全部通过；降到 LR=.001 不是统一 remedy。这里复用了筛选阶段的 cells 与 seeds，
  区间是未做 family correction 的 targeted exploratory evidence，不是新种子确认性推断。
  它目前是未完全解决的优化路径现象，不是已证实的容量障碍。

完整、带限定词的结论请从 [交互式可移植报告](reports/report.html) 或
[最终数学研究报告](reports/FINAL_REPORT.md) 开始阅读；截至 2026-08-15 的逐篇文献查重
在 [LITERATURE_MAP.md](reports/LITERATURE_MAP.md)。

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
  scaling_analysis.py       seed-block factorial、gate 与表示统计
  scaling_io.py             严格只读的实验/机制表解析
  scaling_figures.py        可复现 PNG/SVG 图
  scaling_remedy.py         b=2,048 同 seed 配对 estimands
  scaling_remedy_study.py   补救实验汇总与 provenance audit
  report_artifact.py        最终交互报告的 canonical 数据合同

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

若要运行 Pythia-70M 实验或完整单测，请显式安装预训练模型依赖：

```bash
python -m pip install -e ".[pretrained]"
```

若要复现本次发布所用的 Python 3.11 / CUDA 12.8 软件栈，先安装精确依赖闭包，再以
`--no-deps` 安装本项目：

```bash
python -m pip install --extra-index-url https://download.pytorch.org/whl/cu128 \
  --requirement requirements-lock.txt
python -m pip install --no-deps -e .
```

`requirements-lock.txt` 固定 Python 包版本，但不固定操作系统、GPU driver 或硬件；CPU 或
其他 CUDA backend 应使用 `pyproject.toml` 的有界版本范围，并在结果 manifest 中记录差异。

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

### 6. 重建 scaling 与 b=2,048 配对补救分析

```bash
# 2^4 tuned scaling 的 seed-block factorial/representation 分析
MPLCONFIGDIR=/tmp/transformer-dynamics-mpl PYTHONPATH=src \
  python -m routing_lab.scaling_study

# 同 cell、同 seed、同 evaluation stream 的 20,000-bootstrap remedy 分析
MPLCONFIGDIR=/tmp/transformer-dynamics-mpl PYTHONPATH=src \
  python -m routing_lab.scaling_remedy_study
```

主结果入口分别为 [scaling-analysis-v1](results/scaling-analysis-v1/README.md) 和
[scaling-remedy-analysis-b2048-v1](results/scaling-remedy-analysis-b2048-v1/README.md)。

### 7. 重建最终报告数据合同

```bash
PYTHONPATH=src python -m routing_lab.report_artifact \
  --project-root . \
  --output reports/artifact.json
```

测试会逐字节比较生成结果与仓库中的 `reports/artifact.json`。`reports/report.html` 是由
Codex Data Analytics 的可移植报告打包器从该 JSON 生成的便捷阅读快照；它不是纯 Python
环境的 canonical build target。本次打包通过 schema/package 验证，但由于环境没有兼容的
Chromium headless-shell，浏览器交互验证仅为 structural-only；这一点记录在
`reports/report_delivery_receipt.json`。

## 统计规则

- 独立统计单位是 training seed，不是 layer、head、checkpoint 或 episode。
- 主要区间使用 20,000 次 whole-seed paired bootstrap。
- optimizer 与 architecture replication 是 robustness gate。
- 失败 seed、NaN、门槛失败和调优尝试全部保存在 ledger。
- attention、cosine、NTK 和 landscape 是描述/局部诊断；只有显式替换结构方程并重算
  后代的量称为 causal intervention。

## 推荐阅读顺序

1. [report.html](reports/report.html)：answer-first 图表、精确表格与可展开来源；
2. [FINAL_REPORT.md](reports/FINAL_REPORT.md)：完整数学设定、实验、结果和下一步；
3. [THEORY_PROBLEMS.md](reports/THEORY_PROBLEMS.md)：完整模型、动力学、定理/反例目标；
4. [LITERATURE_MAP.md](reports/LITERATURE_MAP.md)：原始 open problems 与 2026-08-15 边界；
5. [ANALYSIS_PROTOCOL.md](reports/ANALYSIS_PROTOCOL.md)：统计与 claim ladder；
6. [MECHANISM_RESULTS_DRAFT.md](reports/MECHANISM_RESULTS_DRAFT.md)：QK/OV/FFN 负结果与候选；
7. [CLUSTERING_BASELINE.md](reports/CLUSTERING_BASELINE.md)：Perspective baseline；
8. [DYNAMICS_RESULTS.md](reports/DYNAMICS_RESULTS.md)：loss landscape、NTK 与 Hessian 个案。
9. [VALIDATION_REPORT.md](reports/VALIDATION_REPORT.md)：发布测试、provenance 与 claim audit。
10. [NEXT_STEPS.md](reports/NEXT_STEPS.md)：下一轮按 estimand、实验与 theorem 拆分的任务表。

## 可复现性边界

仓库保存代码、配置、seed-level 聚合轨迹、机制表和派生图。模型 checkpoint 体积较大，
本地研究目录保留完整副本；GitHub 只提交四个已报告 dynamics cases 注册的 15 个最小
source snapshots，使默认 loss-landscape/NTK/Hessian 汇总能 fail-closed 重建。其余 `*.pt`
不发布，但都可由保存的配置和 seed 重新生成。

## 许可状态

当前仓库没有授予第三方开源复用许可；项目所有者可用该镜像继续运行和复现，其他读者的
权限仅限适用的平台条款与法律默认范围，公开可见不自动授予复制、修改、运行或再分发权。
若项目所有者希望第三方也能实际复现实验，应在发布后明确选择 MIT、Apache-2.0 或其他
合适许可证。这里刻意不替项目所有者作法律选择。
