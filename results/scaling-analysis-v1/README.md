# Tuned scaling analysis v1

这是一份只读派生分析；原始训练与 mechanism 结果没有被修改。统计单位始终是 training seed，所有主效应和交互先在同一 seed 的完整 16-cell 网格内形成 contrast，再进行 20,000 次 whole-seed bootstrap。normalized rank 与未注册 interactions 属于 secondary family；下列 7 个选择后 contrasts 只报告未做 BH/family correction 的 pointwise percentile intervals，因此是 exploratory pattern discovery，不是 confirmatory factorial inference。

## 精确 estimand

令四个因子编码为 $x_A\in\{-1,+1\}$，endpoint 为 $y_s(x)=r_{eff}/d$。主效应和二阶交互分别是

$$\Delta_A(s)=2\,16^{-1}\sum_x x_Ay_s(x),\qquad
\Delta_{AB}(s)=4\,16^{-1}\sum_x x_Ax_By_s(x).$$

## 结果

- tuned base-routing gate：160/160 seed-runs。
- 含 donor 与 on-manifold swap 的 full causal-robustness gate：136/160 seed-runs，12/16 architecture cells。
- high-LR stress → tuned 的 gate transitions：fail→pass=3，pass→pass=157，pass→fail=0。

Exploratory normalized-rank contrasts（unadjusted pointwise intervals）：

- `width`: -0.0298，95% CI [-0.0359, -0.0231]
- `load`: +0.1404，95% CI [+0.1152, +0.1636]
- `heads`: -0.0491，95% CI [-0.0564, -0.0425]
- `ffn`: +0.0280，95% CI [+0.0172, +0.0390]
- `heads:load`: -0.0586，95% CI [-0.0849, -0.0343]
- `heads:width`: -0.0197，95% CI [-0.0343, -0.0041]
- `ffn:load`: +0.0234，95% CI [-0.0018, +0.0477]

这里不能把 rank contrast 直接称为严格 functional-equivalence 下的 capacity law：base retrieval 全通过；但在固定的 b=256 evaluation stream 上，4 个 cell 没有达到注册的逐 seed 10/10 swap/crosstalk gate。这个 pass count 是阈值筛查，不是 cell mean 的显著性检验。其状态是 **baseline residual requiring a separately paired targeted-remedy analysis; not an open-problem claim**。Cells 3/7 在 step 400→800 仍明显下降，优先解释为 slow convergence / residual crosstalk，并由独立 paired remedy 分析检验；不能提前称为新的 open problem。

## 这些 cosine 能说明什么

- `global_cosine` 是所有非对角 token pair 的平均余弦，只描述全局平均对齐；它既不是多簇 order parameter，也不能排除多个彼此分离的 cluster。
- `target_selectivity = cos(q,k_target) - mean cos(q,k_distractor)` 使用任务标签，只描述 label-conditioned representational selectivity。输入层已因 query 与 target 共享 concept embedding 而具有正值。
- 因此，两者都不能单独证明 attention routing，更不是 causal routing 估计量；真正的 routing 证据来自 attention/path intervention 与 on-manifold function tests。

## 复现

```bash
MPLCONFIGDIR=/tmp/transformer-dynamics-mpl PYTHONPATH=src python -m routing_lab.scaling_study
```

精确数值见同目录 CSV/JSON；`figures/` 同时提供 PNG 与 searchable SVG。
