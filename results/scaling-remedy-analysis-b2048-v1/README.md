# Scaling remedy analysis (b=2048)

这份 follow-up 不把不同 evaluation stream 混进同一个训练效果估计。主分析的 baseline、low-LR remedy 与 same-LR extension 全部使用 `evaluation_batch_size=2048`、`evaluation_seed_offset=910000`；同一 cell 与 seed 因此共享 evaluation RNG contract。

## Estimand

对 cell `c`、training seed `s` 和 endpoint `e`，先形成

$$\Delta_{c,s,e}=Y^{followup}_{c,s,e}-Y^{baseline}_{c,s,e},\qquad
\widehat\Delta_{c,e}=\frac1{10}\sum_{s=0}^9\Delta_{c,s,e}.$$

95% CI 对十维 seed-difference vector 做 20,000 次 percentile bootstrap。head、evaluation episode、不同 cell 都不是独立样本。

严格 full gate 要求同一 seed 同时满足：base accuracy、population risk $\tfrac12\mathrm{MSE}$、value-flip effect、donor accuracy、natural-swap MSE。cell 的 `10/10` pass count 是逐 seed 阈值筛查，不是 cell mean 的置信区间。

## 主要结果

- b=2048 baseline：132/160 seed-runs，通过严格 10/10 gate 的 cell 为 9/16。
- strict-fail cells：[1, 3, 5, 6, 7, 10, 11]；但 natural-swap mean 的 95% CI 完全高于 0.0025 的只有 [3, 7]。这把 material residual 与单-seed threshold tail 分开了。

| comparison | cell | gate baseline | gate follow-up | swap MSE baseline | swap MSE follow-up | paired delta [95% CI] |
|---|---:|---:|---:|---:|---:|---:|
| same_lr_extension_1600 | 3 | 0/10 | 0/10 | 0.021754 | 0.008685 | -0.013069 [-0.017047, -0.009953] |
| same_lr_extension_1600 | 7 | 0/10 | 4/10 | 0.021428 | 0.004853 | -0.016575 [-0.025496, -0.009016] |
| same_lr_extension_1600 | 11 | 7/10 | 8/10 | 0.000955 | 0.001343 | +0.000388 [-0.000966, +0.001783] |
| low_lr_1600 | 3 | 0/10 | 0/10 | 0.021754 | 0.033209 | +0.011455 [+0.005938, +0.017867] |
| low_lr_1600 | 6 | 9/10 | 9/10 | 0.000844 | 0.000883 | +0.000040 [-0.000212, +0.000328] |
| low_lr_1600 | 7 | 0/10 | 0/10 | 0.021428 | 0.028428 | +0.007000 [-0.000065, +0.014697] |
| low_lr_1600 | 11 | 7/10 | 10/10 | 0.000955 | 0.000220 | -0.000735 [-0.001588, +0.000060] |

最重要的区分：same-LR extension 使 cells 3/7 的 swap error 显著下降，但仍未达到 10/10。lower-LR + longer-training schedule 下，两者的 sample mean 都上升；cell 3 的 paired CI 完全高于零，cell 7 的 swap CI 跨零（但其 base、donor 和 Walsh-leakage CI 均显示上升），所以不能把 cell 7 的 swap 变化写成确定恶化。Cell 11 在 low-LR schedule 下达到 10/10，但 same-LR extension 只有 8/10。Cell 6 在 b=2048 下是 9/10→9/10，不是稳健解决。

这些是 schedule-level function effects，不能自动定位为 QK、OV 或 FFN 补偿。要做机制结论，下一步必须把 paired seed 的 attention/path、OV selectivity、FFN signed contribution 与 swap/Walsh change 联合起来。

## Evaluation-stream sensitivity

同一 baseline checkpoints 在旧 b=256 stream 上为 136/160 seed gates，在新 b=2048 stream 上为 132/160。严格 cell gate 因单-seed tail 会变动；cells 3/7 的 material mean residual 则在两条 stream 上都存在。旧 b=256 / remedy b=512 结果只作为 sensitivity，不参与上表的主 paired estimand。

## 复现

```bash
MPLCONFIGDIR=/tmp/transformer-dynamics-mpl PYTHONPATH=src \\
  /home/zion/miniforge3/envs/llm4rec/bin/python -m routing_lab.scaling_remedy_study
```

精确 seed rows、endpoint CIs、source hashes 和图表合同均在本目录中。
