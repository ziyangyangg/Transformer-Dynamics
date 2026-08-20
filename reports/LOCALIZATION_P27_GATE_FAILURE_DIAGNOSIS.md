# Phase-II localization v2：P27 gate failure diagnosis

## 结论

`phase2-controlled-localization-h4-composites-v2` **不是 72/72 完整实验**：62 个
snapshot committed，10 个失败，root `_SUCCESS` 不存在。因此禁止对 62 个 complete cases
做 seed inference，也不报告 factorized/rank/dense 的机制效应。

这 10 个失败是确定的 runner gate-wiring bug，不是 P27 primitive identity 失败。

## 证据

primitive 对每一行 $i$ 使用同一行的联合门：

\[
F_i=\mathbf 1\{a_i>10^{-8}\}\mathbf 1\{r_i>10^{-5}\},
\qquad
a_i=\|\delta_i^{\mathrm{P27}}-\delta_i^{\mathrm{endpoint}}\|_2,
\]

\[
r_i=\frac{a_i}{
\max(\|\delta_i^{\mathrm{P27}}\|_2,
     \|\delta_i^{\mathrm{endpoint}}\|_2)+10^{-12}}.
\]

若任一 $F_i=1$，`localize_controlled_swap` 会立即抛错，因而不会返回完整 tables。实际
10 个失败都发生在 primitive 已完整返回之后；旧 runner 随后错误地单独检查

\[
\max_i r_i>10^{-5}.
\]

所以每个被旧 runner 拒绝的高-relative-gap 行均满足 $a_i\le10^{-8}$。这正是 near-zero
endpoint scale 下，relative error 可大而 absolute identity 仍通过的情形。不能把

\[
(\max_i a_i>10^{-8})\land(\max_i r_i>10^{-5})
\]

当成修复，因为两个最大值也可能来自不同的行。

失败分布：factorized 5 个、rank-matched direct 5 个、dense direct 0 个；全部在 step 6400，
失败日志中的 $\max_i r_i$ 从 $1.1863152\times10^{-5}$ 到
$1.9918527\times10^{-3}$。这只能描述 bug 触发位置；在 72/72 重放前，不能解释成架构差异。

## Prospectively frozen v3 remedy

v3 必须从 raw `qk_head` rows 重建：

\[
N_{\mathrm{joint}}=\sum_i
\mathbf 1\{a_i>10^{-8}\land r_i>10^{-5}\};
\qquad \text{pass}\iff N_{\mathrm{joint}}=0.
\]

每个 snapshot manifest 保存 `max_abs`、`max_rel`、`joint_violation_count` 和两个 tolerance；
resume 时从 raw NPZ 独立重建这些字段。v2 目录保持冻结，v3 使用新目录重放同一 72 个
checkpoint/pair populations。只有 v3 root `_SUCCESS` 且 72/72 后，分析层才运行 20,000 次
whole-seed paired/max-T。

## Receipts

- v2 `failures.jsonl` SHA256:
  `eb61a7632c15aef4540537e18b0c25755c1325c17846413c513acb79acebd7ed`
- v2 `manifest.json` SHA256:
  `d56346dde0453e3bd330dc82bb69a2af6af583325a074f7b3ac263e6bdb57c0e`
- 失败主键：
  - factorized: seeds 101, 102, 105, 108, 110 at step 6400;
  - rank-matched direct: seeds 102, 103, 105, 108, 110 at step 6400.
