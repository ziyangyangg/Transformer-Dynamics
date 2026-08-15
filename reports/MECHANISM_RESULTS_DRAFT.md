# Mechanism results (draft, v1 snapshot diagnostics)

本报告由 `routing_lab.mechanism_analysis` 从两套只读 snapshot 表自动生成。
独立统计单位始终是训练 seed；layer、head、checkpoint 和 512 个 held-out episode
都没有被当成额外样本。95% 区间使用 20,000 次 paired seed bootstrap。

## 结论先行

1. **功能级复合 routing 得到强支持。** 通过功能门槛的模型同时具有接近 1 的
   queried-value flip effect 与 Walsh target coefficient；这说明输出函数选择了
   queried value，而不是仅仅出现好看的 attention 图。
2. **聚合 QK 结果反对“QK route 抑制 content cross-talk”这一具体命题。**
   两个优化器的全部 cell 都得到负的终点 suppression log-ratio 和负的训练增量；
   虽然 opposition rate 略升到 0.5 以上，但 route 总体放大而非缩小 output-relevant
   chord。并且本表没有协议 6.4 的 finite output validation。
3. **OV 结果是 target-vs-distractor 方向选择性，不是协议式 (9) 的
   isotropic-vs-swap attenuation。** 它可以说明训练让 OV 更偏好任务 value
   方向，但不能单独证明 OV 因果消除了 cross-talk。
4. **FFN cancellation 仍不可确认。** v1 表缺少 `E[t_skip^2]` practical-floor
   统计和 finite intervention，同样只能作为局部 adjoint 候选证据。

## 数据审计

| study | rows | cells | seeds/cell | checkpoints | eval batch |
|---|---:|---:|---:|---:|---:|
| primary_adamw | 384 | 8 | 12 | 4 | 512 |
| replication_sgd | 384 | 8 | 12 | 4 | 512 |

注意：此机制重放使用每 seed 512 个 episode；它适合机制定位，但比注册协议中
最终 confirmatory gate 的 8192 episode 更小。下面明确保留这一限制。

### 数值恒等式检查

| study | max Parseval gap | max |Xi-Walsh| | final max |Xi-Walsh| |
|---|---:|---:|---:|
| primary_adamw | 0.00000477 | 0.016352 | 0.003403 |
| replication_sgd | 0.00000477 | 0.013160 | 0.001927 |

Parseval 重构在 float32 下达到微小绝对误差；但 sampled value-flip 与 exhaustive
Walsh target 尚未达到协议要求的逐 seed `1e-5` 一致性。两者可作为相互支持的
功能证据，不能把 v1 表称为已经通过该严格数值门槛。

## 最终功能、供体与因果门槛

| optimizer | cell | function | donor | joint | acc | risk | Xi_value | swap MSE | Walsh target | direct-key gate |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---|
| adamw | C16-d16-L2-H1-m4-ffn32 | 12/12 | 12/12 | 12/12 | 1.000 | 0.0001 | 1.000 | 0.00004 | 1.000 | pass |
| adamw | C16-d16-L2-H1-m4-ffnnone | 12/12 | 12/12 | 12/12 | 1.000 | 0.0001 | 1.000 | 0.00006 | 1.000 | pass |
| adamw | C16-d16-L2-H4-m4-ffn32 | 12/12 | 12/12 | 12/12 | 1.000 | 0.0001 | 0.999 | 0.00009 | 0.999 | pass |
| adamw | C16-d16-L2-H4-m4-ffnnone | 12/12 | 10/12 | 10/12 | 0.999 | 0.0012 | 0.999 | 0.00203 | 0.999 | pass |
| adamw | C64-d16-L2-H1-m4-ffn32 | 12/12 | 12/12 | 12/12 | 1.000 | 0.0001 | 1.000 | 0.00003 | 1.000 | pass |
| adamw | C64-d16-L2-H1-m4-ffnnone | 12/12 | 10/12 | 10/12 | 1.000 | 0.0007 | 0.999 | 0.00086 | 1.000 | pass |
| adamw | C64-d16-L2-H4-m4-ffn32 | 12/12 | 9/12 | 9/12 | 0.999 | 0.0016 | 0.997 | 0.00179 | 0.998 | pass |
| adamw | C64-d16-L2-H4-m4-ffnnone | 12/12 | 6/12 | 6/12 | 0.999 | 0.0024 | 0.994 | 0.00318 | 0.994 | pass |
| sgd | C16-d16-L2-H1-m4-ffn32 | 12/12 | 12/12 | 12/12 | 1.000 | 0.0001 | 1.001 | 0.00009 | 1.001 | pass |
| sgd | C16-d16-L2-H1-m4-ffnnone | 12/12 | 12/12 | 12/12 | 1.000 | 0.0001 | 1.000 | 0.00007 | 1.000 | pass |
| sgd | C16-d16-L2-H4-m4-ffn32 | 12/12 | 12/12 | 12/12 | 1.000 | 0.0002 | 1.000 | 0.00015 | 1.000 | pass |
| sgd | C16-d16-L2-H4-m4-ffnnone | 12/12 | 12/12 | 12/12 | 1.000 | 0.0001 | 0.999 | 0.00015 | 1.000 | pass |
| sgd | C64-d16-L2-H1-m4-ffn32 | 12/12 | 12/12 | 12/12 | 1.000 | 0.0001 | 0.999 | 0.00015 | 0.999 | pass |
| sgd | C64-d16-L2-H1-m4-ffnnone | 11/12 | 11/12 | 11/12 | 0.974 | 0.0312 | 0.937 | 0.00021 | 0.937 | pass |
| sgd | C64-d16-L2-H4-m4-ffn32 | 12/12 | 9/12 | 9/12 | 1.000 | 0.0005 | 0.997 | 0.00161 | 0.997 | pass |
| sgd | C64-d16-L2-H4-m4-ffnnone | 12/12 | 10/12 | 10/12 | 1.000 | 0.0008 | 0.995 | 0.00170 | 0.994 | pass |

## Walsh 复合 routing 与 attention 几何（终点）

Walsh 系数是端到端函数量；attention mass 只是逐层/头描述量。下表 attention
列先在每个 seed 内对 layer/head 等权平均，不能把高 target mass 当成总因果效应。

| optimizer | cell | k_target | distractor energy | interaction energy | attn target | attn distractor | attn self | log target/distractor |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| adamw | C16-d16-L2-H1-m4-ffn32 | 1.000 | 0.00002 | 0.00002 | 0.552 | 0.058 | 0.273 | 3.958 |
| adamw | C16-d16-L2-H1-m4-ffnnone | 1.000 | 0.00004 | 0.00003 | 0.569 | 0.044 | 0.300 | 4.260 |
| adamw | C16-d16-L2-H4-m4-ffn32 | 0.999 | 0.00004 | 0.00004 | 0.354 | 0.114 | 0.306 | 1.642 |
| adamw | C16-d16-L2-H4-m4-ffnnone | 0.999 | 0.00047 | 0.00073 | 0.367 | 0.109 | 0.306 | 1.738 |
| adamw | C64-d16-L2-H1-m4-ffn32 | 1.000 | 0.00002 | 0.00002 | 0.578 | 0.062 | 0.237 | 4.124 |
| adamw | C64-d16-L2-H1-m4-ffnnone | 1.000 | 0.00013 | 0.00011 | 0.615 | 0.044 | 0.253 | 4.579 |
| adamw | C64-d16-L2-H4-m4-ffn32 | 0.998 | 0.00079 | 0.00084 | 0.400 | 0.118 | 0.245 | 1.957 |
| adamw | C64-d16-L2-H4-m4-ffnnone | 0.994 | 0.00127 | 0.00100 | 0.416 | 0.106 | 0.268 | 2.461 |
| sgd | C16-d16-L2-H1-m4-ffn32 | 1.001 | 0.00004 | 0.00003 | 0.496 | 0.070 | 0.294 | 2.582 |
| sgd | C16-d16-L2-H1-m4-ffnnone | 1.000 | 0.00004 | 0.00002 | 0.546 | 0.070 | 0.244 | 2.949 |
| sgd | C16-d16-L2-H4-m4-ffn32 | 1.000 | 0.00007 | 0.00007 | 0.300 | 0.146 | 0.263 | 0.821 |
| sgd | C16-d16-L2-H4-m4-ffnnone | 1.000 | 0.00005 | 0.00007 | 0.318 | 0.155 | 0.216 | 0.813 |
| sgd | C64-d16-L2-H1-m4-ffn32 | 0.999 | 0.00003 | 0.00003 | 0.542 | 0.074 | 0.235 | 2.622 |
| sgd | C64-d16-L2-H1-m4-ffnnone | 1.000 | 0.00005 | 0.00005 | 0.520 | 0.052 | 0.324 | 3.003 |
| sgd | C64-d16-L2-H4-m4-ffn32 | 0.997 | 0.00049 | 0.00057 | 0.319 | 0.156 | 0.212 | 1.002 |
| sgd | C64-d16-L2-H4-m4-ffnnone | 0.994 | 0.00081 | 0.00116 | 0.338 | 0.150 | 0.213 | 1.020 |

## 聚合局部机制：初始化 → 终点

下表的 head 先在 seed 内等权平均，然后才跨 seed 配对。`Δ` 是 final-init。
这些是局部诊断，不是 finite causal compensation 证明。

| optimizer | cell | metric | n | init | final | Δ | 95% CI |
|---|---|---|---:|---:|---:|---:|---|
| adamw | C16-d16-L2-H1-m4-ffn32 | ffn_cancellation_fraction | 12 | 0.268 | 0.387 | 0.119 | [0.078, 0.161] |
| adamw | C16-d16-L2-H1-m4-ffn32 | ffn_opposition_rate | 12 | 0.504 | 0.669 | 0.166 | [0.131, 0.208] |
| adamw | C16-d16-L2-H1-m4-ffn32 | ov_log_target_over_distractor_gain | 12 | -0.056 | 0.854 | 0.910 | [0.738, 1.070] |
| adamw | C16-d16-L2-H1-m4-ffn32 | qk_opposition_rate | 12 | 0.488 | 0.542 | 0.055 | [0.038, 0.072] |
| adamw | C16-d16-L2-H1-m4-ffn32 | qk_suppression_log_ratio | 12 | -0.364 | -1.112 | -0.748 | [-0.900, -0.604] |
| adamw | C16-d16-L2-H1-m4-ffnnone | ov_log_target_over_distractor_gain | 12 | -0.051 | 0.951 | 1.002 | [0.813, 1.177] |
| adamw | C16-d16-L2-H1-m4-ffnnone | qk_opposition_rate | 12 | 0.513 | 0.558 | 0.046 | [0.010, 0.080] |
| adamw | C16-d16-L2-H1-m4-ffnnone | qk_suppression_log_ratio | 12 | -0.193 | -1.153 | -0.960 | [-1.157, -0.749] |
| adamw | C16-d16-L2-H4-m4-ffn32 | ffn_cancellation_fraction | 12 | 0.271 | 0.472 | 0.201 | [0.164, 0.240] |
| adamw | C16-d16-L2-H4-m4-ffn32 | ffn_opposition_rate | 12 | 0.502 | 0.749 | 0.247 | [0.194, 0.301] |
| adamw | C16-d16-L2-H4-m4-ffn32 | ov_log_target_over_distractor_gain | 12 | -0.009 | 0.448 | 0.457 | [0.386, 0.526] |
| adamw | C16-d16-L2-H4-m4-ffn32 | qk_opposition_rate | 12 | 0.503 | 0.532 | 0.029 | [0.018, 0.042] |
| adamw | C16-d16-L2-H4-m4-ffn32 | qk_suppression_log_ratio | 12 | -0.266 | -0.649 | -0.383 | [-0.450, -0.312] |
| adamw | C16-d16-L2-H4-m4-ffnnone | ov_log_target_over_distractor_gain | 10 | -0.032 | 0.718 | 0.750 | [0.663, 0.826] |
| adamw | C16-d16-L2-H4-m4-ffnnone | qk_opposition_rate | 10 | 0.504 | 0.533 | 0.029 | [0.019, 0.038] |
| adamw | C16-d16-L2-H4-m4-ffnnone | qk_suppression_log_ratio | 10 | -0.235 | -0.717 | -0.482 | [-0.558, -0.410] |
| adamw | C64-d16-L2-H1-m4-ffn32 | ffn_cancellation_fraction | 12 | 0.292 | 0.419 | 0.127 | [0.082, 0.173] |
| adamw | C64-d16-L2-H1-m4-ffn32 | ffn_opposition_rate | 12 | 0.523 | 0.718 | 0.195 | [0.133, 0.253] |
| adamw | C64-d16-L2-H1-m4-ffn32 | ov_log_target_over_distractor_gain | 12 | 0.010 | 1.000 | 0.990 | [0.870, 1.108] |
| adamw | C64-d16-L2-H1-m4-ffn32 | qk_opposition_rate | 12 | 0.508 | 0.529 | 0.022 | [0.010, 0.033] |
| adamw | C64-d16-L2-H1-m4-ffn32 | qk_suppression_log_ratio | 12 | -0.270 | -0.851 | -0.582 | [-0.689, -0.476] |
| adamw | C64-d16-L2-H1-m4-ffnnone | ov_log_target_over_distractor_gain | 10 | -0.013 | 1.201 | 1.215 | [1.096, 1.331] |
| adamw | C64-d16-L2-H1-m4-ffnnone | qk_opposition_rate | 10 | 0.498 | 0.532 | 0.034 | [0.012, 0.056] |
| adamw | C64-d16-L2-H1-m4-ffnnone | qk_suppression_log_ratio | 10 | -0.343 | -0.899 | -0.556 | [-0.786, -0.373] |
| adamw | C64-d16-L2-H4-m4-ffn32 | ffn_cancellation_fraction | 9 | 0.304 | 0.488 | 0.184 | [0.138, 0.229] |
| adamw | C64-d16-L2-H4-m4-ffn32 | ffn_opposition_rate | 9 | 0.542 | 0.766 | 0.224 | [0.175, 0.272] |
| adamw | C64-d16-L2-H4-m4-ffn32 | ov_log_target_over_distractor_gain | 9 | -0.000 | 0.659 | 0.659 | [0.516, 0.811] |
| adamw | C64-d16-L2-H4-m4-ffn32 | qk_opposition_rate | 9 | 0.508 | 0.526 | 0.018 | [0.007, 0.028] |
| adamw | C64-d16-L2-H4-m4-ffn32 | qk_suppression_log_ratio | 9 | -0.224 | -0.606 | -0.383 | [-0.459, -0.293] |
| adamw | C64-d16-L2-H4-m4-ffnnone | ov_log_target_over_distractor_gain | 6 | -0.006 | 0.799 | 0.805 | [0.611, 0.967] |
| adamw | C64-d16-L2-H4-m4-ffnnone | qk_opposition_rate | 6 | 0.508 | 0.536 | 0.028 | [0.018, 0.040] |
| adamw | C64-d16-L2-H4-m4-ffnnone | qk_suppression_log_ratio | 6 | -0.254 | -0.593 | -0.339 | [-0.465, -0.195] |
| sgd | C16-d16-L2-H1-m4-ffn32 | ffn_cancellation_fraction | 12 | 0.268 | 0.328 | 0.060 | [0.011, 0.106] |
| sgd | C16-d16-L2-H1-m4-ffn32 | ffn_opposition_rate | 12 | 0.504 | 0.617 | 0.113 | [0.064, 0.158] |
| sgd | C16-d16-L2-H1-m4-ffn32 | ov_log_target_over_distractor_gain | 12 | -0.056 | 0.765 | 0.821 | [0.715, 0.929] |
| sgd | C16-d16-L2-H1-m4-ffn32 | qk_opposition_rate | 12 | 0.488 | 0.541 | 0.053 | [0.037, 0.067] |
| sgd | C16-d16-L2-H1-m4-ffn32 | qk_suppression_log_ratio | 12 | -0.364 | -1.004 | -0.640 | [-0.789, -0.513] |
| sgd | C16-d16-L2-H1-m4-ffnnone | ov_log_target_over_distractor_gain | 12 | -0.051 | 0.992 | 1.044 | [0.980, 1.112] |
| sgd | C16-d16-L2-H1-m4-ffnnone | qk_opposition_rate | 12 | 0.513 | 0.543 | 0.030 | [0.008, 0.055] |
| sgd | C16-d16-L2-H1-m4-ffnnone | qk_suppression_log_ratio | 12 | -0.193 | -0.980 | -0.787 | [-1.022, -0.560] |
| sgd | C16-d16-L2-H4-m4-ffn32 | ffn_cancellation_fraction | 12 | 0.271 | 0.354 | 0.083 | [0.051, 0.115] |
| sgd | C16-d16-L2-H4-m4-ffn32 | ffn_opposition_rate | 12 | 0.502 | 0.644 | 0.142 | [0.096, 0.185] |
| sgd | C16-d16-L2-H4-m4-ffn32 | ov_log_target_over_distractor_gain | 12 | -0.009 | 0.345 | 0.354 | [0.257, 0.456] |
| sgd | C16-d16-L2-H4-m4-ffn32 | qk_opposition_rate | 12 | 0.503 | 0.521 | 0.018 | [0.011, 0.027] |
| sgd | C16-d16-L2-H4-m4-ffn32 | qk_suppression_log_ratio | 12 | -0.266 | -0.684 | -0.418 | [-0.500, -0.342] |
| sgd | C16-d16-L2-H4-m4-ffnnone | ov_log_target_over_distractor_gain | 12 | -0.007 | 0.480 | 0.487 | [0.387, 0.574] |
| sgd | C16-d16-L2-H4-m4-ffnnone | qk_opposition_rate | 12 | 0.505 | 0.524 | 0.019 | [0.007, 0.031] |
| sgd | C16-d16-L2-H4-m4-ffnnone | qk_suppression_log_ratio | 12 | -0.246 | -0.744 | -0.498 | [-0.591, -0.397] |
| sgd | C64-d16-L2-H1-m4-ffn32 | ffn_cancellation_fraction | 12 | 0.292 | 0.296 | 0.005 | [-0.052, 0.061] |
| sgd | C64-d16-L2-H1-m4-ffn32 | ffn_opposition_rate | 12 | 0.523 | 0.579 | 0.056 | [-0.015, 0.127] |
| sgd | C64-d16-L2-H1-m4-ffn32 | ov_log_target_over_distractor_gain | 12 | 0.010 | 0.862 | 0.852 | [0.729, 0.983] |
| sgd | C64-d16-L2-H1-m4-ffn32 | qk_opposition_rate | 12 | 0.508 | 0.538 | 0.030 | [0.013, 0.047] |
| sgd | C64-d16-L2-H1-m4-ffn32 | qk_suppression_log_ratio | 12 | -0.270 | -0.787 | -0.518 | [-0.606, -0.425] |
| sgd | C64-d16-L2-H1-m4-ffnnone | ov_log_target_over_distractor_gain | 11 | -0.017 | 0.840 | 0.857 | [0.657, 1.061] |
| sgd | C64-d16-L2-H1-m4-ffnnone | qk_opposition_rate | 11 | 0.499 | 0.534 | 0.036 | [0.022, 0.049] |
| sgd | C64-d16-L2-H1-m4-ffnnone | qk_suppression_log_ratio | 11 | -0.347 | -1.052 | -0.705 | [-0.848, -0.571] |
| sgd | C64-d16-L2-H4-m4-ffn32 | ffn_cancellation_fraction | 9 | 0.302 | 0.332 | 0.030 | [-0.018, 0.074] |
| sgd | C64-d16-L2-H4-m4-ffn32 | ffn_opposition_rate | 9 | 0.535 | 0.624 | 0.089 | [0.016, 0.162] |
| sgd | C64-d16-L2-H4-m4-ffn32 | ov_log_target_over_distractor_gain | 9 | 0.008 | 0.456 | 0.448 | [0.294, 0.615] |
| sgd | C64-d16-L2-H4-m4-ffn32 | qk_opposition_rate | 9 | 0.508 | 0.510 | 0.002 | [-0.009, 0.013] |
| sgd | C64-d16-L2-H4-m4-ffn32 | qk_suppression_log_ratio | 9 | -0.231 | -0.552 | -0.321 | [-0.450, -0.220] |
| sgd | C64-d16-L2-H4-m4-ffnnone | ov_log_target_over_distractor_gain | 10 | -0.024 | 0.452 | 0.475 | [0.303, 0.628] |
| sgd | C64-d16-L2-H4-m4-ffnnone | qk_opposition_rate | 10 | 0.506 | 0.519 | 0.012 | [0.003, 0.023] |
| sgd | C64-d16-L2-H4-m4-ffnnone | qk_suppression_log_ratio | 10 | -0.226 | -0.593 | -0.367 | [-0.485, -0.262] |

## 优化器方向复制

`same` 只表示 AdamW 与 momentum-SGD 的 init→final 均值同号；`SGD-CI`
要求 SGD 的 95% CI 排除 0。即使二者都通过，也不能补上缺失的 finite
intervention 或 practical-floor gate。

| cell | metric | n common | AdamW Δ | SGD Δ | same | SGD-CI | desired | both desired | SGD-AdamW |
|---|---|---:|---:|---:|---|---|---|---|---:|
| C16-d16-L2-H1-m4-ffn32 | ffn_cancellation_fraction | 12 | 0.119 | 0.060 | yes | pass | pass | pass | -0.059 |
| C16-d16-L2-H1-m4-ffn32 | ffn_opposition_rate | 12 | 0.166 | 0.113 | yes | pass | pass | pass | -0.053 |
| C16-d16-L2-H1-m4-ffn32 | ov_log_target_over_distractor_gain | 12 | 0.910 | 0.821 | yes | pass | pass | pass | -0.089 |
| C16-d16-L2-H1-m4-ffn32 | qk_opposition_rate | 12 | 0.055 | 0.053 | yes | pass | pass | pass | -0.001 |
| C16-d16-L2-H1-m4-ffn32 | qk_suppression_log_ratio | 12 | -0.748 | -0.640 | yes | pass | fail | fail | 0.108 |
| C16-d16-L2-H1-m4-ffnnone | ov_log_target_over_distractor_gain | 12 | 1.002 | 1.044 | yes | pass | pass | pass | 0.041 |
| C16-d16-L2-H1-m4-ffnnone | qk_opposition_rate | 12 | 0.046 | 0.030 | yes | pass | pass | pass | -0.016 |
| C16-d16-L2-H1-m4-ffnnone | qk_suppression_log_ratio | 12 | -0.960 | -0.787 | yes | pass | fail | fail | 0.173 |
| C16-d16-L2-H4-m4-ffn32 | ffn_cancellation_fraction | 12 | 0.201 | 0.083 | yes | pass | pass | pass | -0.118 |
| C16-d16-L2-H4-m4-ffn32 | ffn_opposition_rate | 12 | 0.247 | 0.142 | yes | pass | pass | pass | -0.105 |
| C16-d16-L2-H4-m4-ffn32 | ov_log_target_over_distractor_gain | 12 | 0.457 | 0.354 | yes | pass | pass | pass | -0.102 |
| C16-d16-L2-H4-m4-ffn32 | qk_opposition_rate | 12 | 0.029 | 0.018 | yes | pass | pass | pass | -0.011 |
| C16-d16-L2-H4-m4-ffn32 | qk_suppression_log_ratio | 12 | -0.383 | -0.418 | yes | pass | fail | fail | -0.035 |
| C16-d16-L2-H4-m4-ffnnone | ov_log_target_over_distractor_gain | 10 | 0.750 | 0.499 | yes | pass | pass | pass | -0.251 |
| C16-d16-L2-H4-m4-ffnnone | qk_opposition_rate | 10 | 0.029 | 0.017 | yes | pass | pass | pass | -0.012 |
| C16-d16-L2-H4-m4-ffnnone | qk_suppression_log_ratio | 10 | -0.482 | -0.498 | yes | pass | fail | fail | -0.016 |
| C64-d16-L2-H1-m4-ffn32 | ffn_cancellation_fraction | 12 | 0.127 | 0.005 | yes | fail | fail | fail | -0.123 |
| C64-d16-L2-H1-m4-ffn32 | ffn_opposition_rate | 12 | 0.195 | 0.056 | yes | fail | fail | fail | -0.139 |
| C64-d16-L2-H1-m4-ffn32 | ov_log_target_over_distractor_gain | 12 | 0.990 | 0.852 | yes | pass | pass | pass | -0.138 |
| C64-d16-L2-H1-m4-ffn32 | qk_opposition_rate | 12 | 0.022 | 0.030 | yes | pass | pass | pass | 0.009 |
| C64-d16-L2-H1-m4-ffn32 | qk_suppression_log_ratio | 12 | -0.582 | -0.518 | yes | pass | fail | fail | 0.064 |
| C64-d16-L2-H1-m4-ffnnone | ov_log_target_over_distractor_gain | 10 | 1.215 | 0.859 | yes | pass | pass | pass | -0.356 |
| C64-d16-L2-H1-m4-ffnnone | qk_opposition_rate | 10 | 0.034 | 0.036 | yes | pass | pass | pass | 0.002 |
| C64-d16-L2-H1-m4-ffnnone | qk_suppression_log_ratio | 10 | -0.556 | -0.653 | yes | pass | fail | fail | -0.097 |
| C64-d16-L2-H4-m4-ffn32 | ffn_cancellation_fraction | 6 | 0.169 | 0.014 | yes | fail | fail | fail | -0.155 |
| C64-d16-L2-H4-m4-ffn32 | ffn_opposition_rate | 6 | 0.209 | 0.058 | yes | fail | fail | fail | -0.152 |
| C64-d16-L2-H4-m4-ffn32 | ov_log_target_over_distractor_gain | 6 | 0.637 | 0.443 | yes | fail | fail | fail | -0.194 |
| C64-d16-L2-H4-m4-ffn32 | qk_opposition_rate | 6 | 0.013 | -0.002 | no | fail | fail | fail | -0.015 |
| C64-d16-L2-H4-m4-ffn32 | qk_suppression_log_ratio | 6 | -0.395 | -0.389 | yes | fail | fail | fail | 0.006 |
| C64-d16-L2-H4-m4-ffnnone | ov_log_target_over_distractor_gain | 6 | 0.805 | 0.440 | yes | fail | fail | fail | -0.365 |
| C64-d16-L2-H4-m4-ffnnone | qk_opposition_rate | 6 | 0.028 | 0.010 | yes | fail | fail | fail | -0.019 |
| C64-d16-L2-H4-m4-ffnnone | qk_suppression_log_ratio | 6 | -0.339 | -0.379 | yes | fail | fail | fail | -0.040 |

## 哪些机制没有得到支持

- `ffn_cancellation_fraction`：4/4 个 matched cell 同号，2/4 个复制了观测方向，2/4 个在两优化器中都支持预注册机制方向。
- `ffn_opposition_rate`：4/4 个 matched cell 同号，2/4 个复制了观测方向，2/4 个在两优化器中都支持预注册机制方向。
- `ov_log_target_over_distractor_gain`：8/8 个 matched cell 同号，6/8 个复制了观测方向，6/8 个在两优化器中都支持预注册机制方向。
- `qk_opposition_rate`：7/8 个 matched cell 同号，6/8 个复制了观测方向，6/8 个在两优化器中都支持预注册机制方向。
- `qk_suppression_log_ratio`：8/8 个 matched cell 同号，6/8 个复制了观测方向，0/8 个在两优化器中都支持预注册机制方向。
- **确认性的 QK/OV/FFN compensation 数量仍为 0。** 原因不是把非显著结果
  当成反证，而是 v1 estimand 本身尚未包含协议规定的 finite output validation；
  FFN 还缺 practical floor，OV 指标也不是注册的 isotropic attenuation。
- `qk_suppression_log_ratio > 0` 表示在局部 output adjoint 上 route 比 content
  单独更小；本实验所有聚合终点与增量均小于 0，即 route 增强而非抑制。
  这会否定当前这个具体 QK-compensation 解释；只看 opposition rate 略高于
  0.5 不能挽救该命题。
- natural swap MSE 与 donor accuracy 是必要 gate：若某 seed 未通过，不能把其
  下游局部抵消解释为成功保持函数不变。

## 可复现文件

- `seed_step_metrics.csv`：每 optimizer/cell/seed/checkpoint 一行；
- `site_step_metrics.csv`：逐 layer/head 的长表；
- `cell_step_summary.csv`：跨 seed 的描述性轨迹；
- `paired_delta_summary.csv`：all-scheduled 与 gate-qualified 配对增量；
- `site_delta_summary.csv`：逐层/头配对增量；
- `optimizer_replication.csv`：共同合格 seed 上的优化器方向复制；
- `functional_gates.json/csv`：注册门槛及成功 seed。
